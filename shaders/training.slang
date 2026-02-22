// training.slang

// Training workgroup dimensions (must match kernel dispatch)
static const uint TRAIN_WG_X = 4;
static const uint TRAIN_WG_Y = 4;
static const uint TRAIN_WG_Z = 4;
static const uint TRAIN_WG_SIZE = TRAIN_WG_X * TRAIN_WG_Y * TRAIN_WG_Z;

// DEFINITIONS
static const uint MAX_GAUSSIANS_PER_TILE = 64;

// PARAMETER LAYOUT CONFIGURATION
static const uint PARAM_POS_X  = 0;
static const uint PARAM_POS_Y  = 1;
static const uint PARAM_POS_Z  = 2;
static const uint PARAM_SCALE_X = 3;
static const uint PARAM_SCALE_Y = 4;
static const uint PARAM_SCALE_Z = 5;
static const uint PARAM_QUAT_X = 6;
static const uint PARAM_QUAT_Y = 7;
static const uint PARAM_QUAT_Z = 8;
static const uint PARAM_QUAT_W = 9;
static const uint PARAM_WEIGHT = 10;
static const uint PARAMS_PER_GAUSSIAN = 11;

static const uint LR_GROUP_POSITION = 0;
static const uint LR_GROUP_SCALE    = 1;
static const uint LR_GROUP_ROTATION = 2;
static const uint LR_GROUP_WEIGHT   = 3;
static const uint NUM_LR_GROUPS     = 4;

// GAUSSIAN STRUCTURE
struct Gaussian 
{
    float3 pos;
    float3 scale;   // per-axis standard deviations
    float4 quat;    // rotation quaternion (x,y,z,w), unit length
    float  weight;
    
    static float getParam(uint gaussianID, uint paramID)
    {
        return asfloat(gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID]);
    }
    
    static void setParam(uint gaussianID, uint paramID, float value)
    {
        gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID] = asuint(value);
    }
    
    static Gaussian load(uint gid)
    {
        Gaussian g;
        g.pos.x   = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_POS_X]);
        g.pos.y   = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_POS_Y]);
        g.pos.z   = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_POS_Z]);
        g.scale.x = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_SCALE_X]);
        g.scale.y = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_SCALE_Y]);
        g.scale.z = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_SCALE_Z]);
        g.quat.x  = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_X]);
        g.quat.y  = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_Y]);
        g.quat.z  = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_Z]);
        g.quat.w  = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_W]);
        g.weight  = asfloat(gGaussianParamsRaw[gid * PARAMS_PER_GAUSSIAN + PARAM_WEIGHT]);
        return g;
    }

};

// BINDINGS
[[vk::binding(0)]] Texture3D<float> ReferenceVol;
[[vk::binding(1)]] RWStructuredBuffer<uint> gGaussianParamsRaw;  // Raw floab buffer
[[vk::binding(2)]] RWStructuredBuffer<uint> gGradientsRaw;
[[vk::binding(3)]] RWStructuredBuffer<uint> gTileContent; 
[[vk::binding(4)]] RWStructuredBuffer<uint> gTileCounts; 
[[vk::binding(5)]] RWStructuredBuffer<float> gAdamFirstMoment;
[[vk::binding(6)]] RWStructuredBuffer<float> gAdamSecondMoment;
[[vk::binding(7)]] RWTexture3D<float4> gDebugVolume;

groupshared float reductionBuffer[TRAIN_WG_SIZE];


cbuffer TrainParams {
    uint3 volumeResolution;
    uint  gaussianCount;
    float3 minWorld;
    float3 maxWorld;
    
    // Generic learning rates (indexed by LR_GROUP_*)
    float learningRates[NUM_LR_GROUPS];
    
    uint3 tileResolution;
    uint totalTiles;
    uint tileSize;
    
    float adamBeta1;
    float adamBeta2;
    float adamEpsilon;
    uint adamIteration;

    // Debug
    uint debugMode;
    float debugScale;
};

// Debug
static const uint DEBUG_NONE = 0;
static const uint DEBUG_LOSS = 1;
static const uint DEBUG_GRADIENT_MAG = 2;
static const uint DEBUG_TILE_DENSITY = 3;
static const uint DEBUG_OVERLAP = 4;
static const uint DEBUG_PRED_VS_TARGET = 5;

// Scale factors for different debug modes
static const float DEBUG_LOSS_SCALE = 20.0;           // Loss values typically 0.0-0.05
static const float DEBUG_GRADIENT_SCALE = 5.0;        // Gradients typically 0.0-0.2
static const float DEBUG_TILE_SCALE = 1.0 / 32.0;    // Tile counts 0-64
static const float DEBUG_OVERLAP_SCALE = 2.0;         // Density typically 0.0-0.5
static const float DEBUG_ERROR_SCALE = 10.0;          // Prediction error typically -0.1 to +0.1

// Opacity scaling (how quickly debug vis becomes opaque)
static const float DEBUG_ALPHA_MULT_LOSS = 100.0;
static const float DEBUG_ALPHA_MULT_GRADIENT = 50.0;
static const float DEBUG_ALPHA_MULT_TILE = 2.0;
static const float DEBUG_ALPHA_MULT_OVERLAP = 5.0;
static const float DEBUG_ALPHA_MULT_ERROR = 20.0;

// Heatmap thresholds for color mapping
static const float HEATMAP_LOW_THRESHOLD = 0.2;    // Below this = mostly green
static const float HEATMAP_HIGH_THRESHOLD = 0.8;   // Above this = mostly red



// Maps each parameter to its learning rate group and constraints
struct ParamMetadata
{
    uint lrGroup;
    float minVal;
    float maxVal;
};

static const ParamMetadata PARAM_META[PARAMS_PER_GAUSSIAN] = {
    { LR_GROUP_POSITION, -1e6, 1e6  },  // pos.x
    { LR_GROUP_POSITION, -1e6, 1e6  },  // pos.y
    { LR_GROUP_POSITION, -1e6, 1e6  },  // pos.z
    { LR_GROUP_SCALE,    1e-4, 1e6  },  // scale.x (must stay positive)
    { LR_GROUP_SCALE,    1e-4, 1e6  },  // scale.y
    { LR_GROUP_SCALE,    1e-4, 1e6  },  // scale.z
    { LR_GROUP_ROTATION, -1.0, 1.0  },  // quat.x
    { LR_GROUP_ROTATION, -1.0, 1.0  },  // quat.y
    { LR_GROUP_ROTATION, -1.0, 1.0  },  // quat.z
    { LR_GROUP_ROTATION, -1.0, 1.0  },  // quat.w
    { LR_GROUP_WEIGHT,   0.0,  1.0  },  // weight
};

// MATH
// Build 3x3 rotation matrix from unit quaternion (x,y,z,w)
[Differentiable]
float3x3 quatToMatrix(float4 q)
{
    float x=q.x, y=q.y, z=q.z, w=q.w;
    return float3x3(
        1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
          2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
          2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)
    );
}

// Build precision matrix Sigma^{-1} = R * diag(1/s^2) * R^T
// This is what we use in the exponent: d^T * Sigma^{-1} * d
[Differentiable]
float3x3 buildPrecisionMatrix(float3 scale, float4 quat)
{
    float3x3 R = quatToMatrix(quat);
    float3 invS2 = 1.0 / (scale * scale);
    
    // P_ij = sum_k R[i][k] * invS2[k] * R[j][k]
    float3x3 P;
    P[0][0] = R[0][0]*R[0][0]*invS2.x + R[0][1]*R[0][1]*invS2.y + R[0][2]*R[0][2]*invS2.z;
    P[0][1] = R[0][0]*R[1][0]*invS2.x + R[0][1]*R[1][1]*invS2.y + R[0][2]*R[1][2]*invS2.z;
    P[0][2] = R[0][0]*R[2][0]*invS2.x + R[0][1]*R[2][1]*invS2.y + R[0][2]*R[2][2]*invS2.z;
    P[1][0] = P[0][1]; // symmetric
    P[1][1] = R[1][0]*R[1][0]*invS2.x + R[1][1]*R[1][1]*invS2.y + R[1][2]*R[1][2]*invS2.z;
    P[1][2] = R[1][0]*R[2][0]*invS2.x + R[1][1]*R[2][1]*invS2.y + R[1][2]*R[2][2]*invS2.z;
    P[2][0] = P[0][2]; // symmetric
    P[2][1] = P[1][2]; // symmetric
    P[2][2] = R[2][0]*R[2][0]*invS2.x + R[2][1]*R[2][1]*invS2.y + R[2][2]*R[2][2]*invS2.z;
    return P;
}

// Mahalanobis squared distance: d^T * P * d
[Differentiable]
float mahalanobis2(float3 d, float3x3 P)
{
    return dot(d, mul(P, d));
}

// HELPERS
int3 WorldToVoxel(float3 pos) {
    float3 uvw = (pos - minWorld) / (maxWorld - minWorld);
    return int3(floor(uvw * float3(volumeResolution)));
}

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + tileID.y * tileResolution.x + tileID.x;
}

// Compute local thread index within workgroup
uint GetLocalThreadIndex(uint3 tid) {
    uint3 local = tid % uint3(TRAIN_WG_X, TRAIN_WG_Y, TRAIN_WG_Z);
    return local.x + local.y * TRAIN_WG_X + local.z * TRAIN_WG_X * TRAIN_WG_Y;
}

float3 heatmapColor(float t) {
    // Improved heatmap with better distribution
    t = clamp(t, 0.0, 1.0);
    
    // Apply power curve for better visual distribution
    t = pow(t, 0.7);  // Makes mid-range more visible
    
    if (t < 0.33) {
        // Green to Yellow-Green
        return lerp(float3(0.0, 0.8, 0.0), float3(0.5, 1.0, 0.0), t * 3.0);
    } else if (t < 0.67) {
        // Yellow-Green to Orange
        return lerp(float3(0.5, 1.0, 0.0), float3(1.0, 0.6, 0.0), (t - 0.33) * 3.0);
    } else {
        // Orange to Red
        return lerp(float3(1.0, 0.6, 0.0), float3(1.0, 0.0, 0.0), (t - 0.67) * 3.0);
    }
}

float3 divergingColor(float t) {
    // Improved diverging colormap
    // Blue (under) → White (zero) → Red (over)
    t = clamp(t, -1.0, 1.0);
    
    if (t < -0.1) {
        // Strong blue for significant underprediction
        float blend = clamp((-t - 0.1) / 0.9, 0.0, 1.0);
        return lerp(float3(0.7, 0.9, 1.0), float3(0.0, 0.0, 1.0), blend);
    } else if (t < 0.1) {
        // White zone for small errors
        float blend = t / 0.1 * 0.5 + 0.5;  // 0.0 to 1.0
        return lerp(float3(0.7, 0.9, 1.0), float3(1.0, 0.9, 0.7), blend);
    } else {
        // Strong red for significant overprediction
        float blend = clamp((t - 0.1) / 0.9, 0.0, 1.0);
        return lerp(float3(1.0, 0.9, 0.7), float3(1.0, 0.0, 0.0), blend);
    }
}

[shader("compute")]
[numthreads(TRAIN_WG_X, TRAIN_WG_Y, TRAIN_WG_Z)]
void compute_debug(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= volumeResolution)) return;
    if (debugMode == DEBUG_NONE) return;
    
    float3 uvw = (float3(tid) + 0.5) / float3(volumeResolution);
    float3 worldPos = minWorld + uvw * (maxWorld - minWorld);
    
    float rawValue = 0.0;
    float normalizedValue = 0.0;
    float3 debugColor = float3(0.0);
    float debugAlpha = 0.0;
    
    if (debugMode == DEBUG_LOSS) {
        // Compute loss at this voxel
        float predicted = compute_density_tiled(tid, worldPos);
        float target = ReferenceVol[tid];
        float diff = predicted - target;
        float loss = diff * diff;
        
        rawValue = loss;
        normalizedValue = loss * DEBUG_LOSS_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = min(loss * DEBUG_ALPHA_MULT_LOSS * debugScale, 1.0);
    }
    else if (debugMode == DEBUG_GRADIENT_MAG) {
        // Sum gradient magnitudes for all Gaussians affecting this voxel
        uint tileIdx = GetTileIndex(tid);
        uint count = gTileCounts[tileIdx];
        count = min(count, MAX_GAUSSIANS_PER_TILE);
        
        float totalGradMag = 0.0;
        for (uint i = 0; i < count; ++i) {
            uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
            
            float gradMag = 0.0;
            for (uint p = 0; p < PARAMS_PER_GAUSSIAN; ++p) {
                float g = asfloat(gGradientsRaw[gid * PARAMS_PER_GAUSSIAN + p]);
                gradMag += g * g;  // L2 norm squared
            }
            totalGradMag += sqrt(gradMag);  // L2 norm
        }
        
        rawValue = totalGradMag;
        normalizedValue = totalGradMag * DEBUG_GRADIENT_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = min(totalGradMag * DEBUG_ALPHA_MULT_GRADIENT * debugScale, 1.0);
    }
    else if (debugMode == DEBUG_TILE_DENSITY) {
        // Visualize Gaussian count per tile
        uint tileIdx = GetTileIndex(tid);
        uint count = gTileCounts[tileIdx];
        
        rawValue = float(count);
        normalizedValue = float(count) * DEBUG_TILE_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = normalizedValue;
    }
    else if (debugMode == DEBUG_OVERLAP) {
        // Visualize Gaussian density
        float density = compute_density_tiled(tid, worldPos);
        
        rawValue = density;
        normalizedValue = density * DEBUG_OVERLAP_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = min(density * DEBUG_ALPHA_MULT_OVERLAP * debugScale, 1.0);
    }
    else if (debugMode == DEBUG_PRED_VS_TARGET) {
        // Visualize prediction error
        float predicted = compute_density_tiled(tid, worldPos);
        float target = ReferenceVol[tid];
        float error = predicted - target;
        
        rawValue = error;
        normalizedValue = error * DEBUG_ERROR_SCALE * debugScale;
        debugColor = divergingColor(normalizedValue);
        debugAlpha = min(abs(error) * DEBUG_ALPHA_MULT_ERROR * debugScale, 1.0);
    }
    
    // Store: RGB = color, A = alpha (for compositing)
    gDebugVolume[tid] = float4(debugColor, debugAlpha);
}

// CLEAR TILES
[shader("compute")]
[numthreads(256, 1, 1)]
void clear_tiles(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= totalTiles) return;
    gTileCounts[tid.x] = 0;
}

// BIN GAUSSIANS
[shader("compute")]
[numthreads(256, 1, 1)]
void bin_gaussians(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;

    Gaussian g = Gaussian::load(gid);
    
    float radius = max(g.scale.x, max(g.scale.y, g.scale.z)) * 3.0;
    int3 min_vox = WorldToVoxel(g.pos - radius);
    int3 max_vox = WorldToVoxel(g.pos + radius);
    
    uint3 min_tile = max(int3(0), min_vox) / tileSize;
    uint3 max_tile = min(int3(volumeResolution)/tileSize - 1, max_vox / tileSize);
    
    for (uint z = min_tile.z; z <= max_tile.z; ++z) {
        for (uint y = min_tile.y; y <= max_tile.y; ++y) {
            for (uint x = min_tile.x; x <= max_tile.x; ++x) {
                uint tileIdx = z * (tileResolution.x * tileResolution.y) + y * tileResolution.x + x;
                uint slot;
                InterlockedAdd(gTileCounts[tileIdx], 1, slot);
                if (slot < MAX_GAUSSIANS_PER_TILE) {
                    uint bufferIdx = tileIdx * MAX_GAUSSIANS_PER_TILE + slot;
                    gTileContent[bufferIdx] = gid;
                }
            }
        }
    }
}

// ATOMIC GRADIENT ACCUMULATION
void atomicAccumulate(uint idx, float val) {
    if (val == 0.0) return;
    for (;;) {
        uint oldInt = gGradientsRaw[idx];
        float oldFloat = asfloat(oldInt);
        float newFloat = oldFloat + val;
        uint newInt = asuint(newFloat);
        uint original = 0;
        InterlockedCompareExchange(gGradientsRaw[idx], oldInt, newInt, original);
        if (original == oldInt) break;
    }
}

// DIFFERENTIABLE PARAMETER LOADERS
[BackwardDerivative(loadParam_bwd)]
float loadParam(uint gaussianID, uint paramID, uint localThreadIdx) {
    return asfloat(gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID]);
}

void loadParam_bwd(uint gaussianID, uint paramID, uint localThreadIdx, float dOut) {
    if (localThreadIdx >= TRAIN_WG_SIZE) return;
    
    reductionBuffer[localThreadIdx] = dOut;
    GroupMemoryBarrierWithGroupSync();
    
    for (uint stride = TRAIN_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (localThreadIdx < stride) {
            reductionBuffer[localThreadIdx] += reductionBuffer[localThreadIdx + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (localThreadIdx == 0) {
        atomicAccumulate(gaussianID * PARAMS_PER_GAUSSIAN + paramID, reductionBuffer[0]);
    }
}

[Differentiable]
float gaussian_math(float3 p, float3 mu, float3x3 precision)
{
    float3 d = p - mu;
    return exp(-0.5 * mahalanobis2(d, precision));
}

[Differentiable]
float compute_density_tiled(no_diff uint3 tid, no_diff float3 worldPos) {
    float density = 0.0;
    
    uint tileIdx = GetTileIndex(tid);
    uint count = gTileCounts[tileIdx];
    count = min(count, MAX_GAUSSIANS_PER_TILE);

    uint localThreadIdx = GetLocalThreadIndex(tid);
    
    [ForceUnroll] 
    for(uint i=0; i<MAX_GAUSSIANS_PER_TILE; ++i) {
        if (i >= count) break;

        uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        float3 g_pos = float3(
            loadParam(gid, PARAM_POS_X, localThreadIdx),
            loadParam(gid, PARAM_POS_Y, localThreadIdx),
            loadParam(gid, PARAM_POS_Z, localThreadIdx)
        );
        
        // Load via generic accessor
        float3 g_scale = float3(
            loadParam(gid, PARAM_SCALE_X, localThreadIdx),
            loadParam(gid, PARAM_SCALE_Y, localThreadIdx),
            loadParam(gid, PARAM_SCALE_Z, localThreadIdx)
        );
        float4 g_quat = float4(
            loadParam(gid, PARAM_QUAT_X, localThreadIdx),
            loadParam(gid, PARAM_QUAT_Y, localThreadIdx),
            loadParam(gid, PARAM_QUAT_Z, localThreadIdx),
            loadParam(gid, PARAM_QUAT_W, localThreadIdx)
        );
        float g_weight = loadParam(gid, PARAM_WEIGHT, localThreadIdx);

        float3 d = worldPos - g_pos;
        // Cheap AABB cull using max scale radius
        float maxScale = max(g_scale.x, max(g_scale.y, g_scale.z));
        if (no_diff(dot(d,d) > (maxScale * 3.0) * (maxScale * 3.0))) continue;

        float3x3 precision = buildPrecisionMatrix(g_scale, g_quat);
        density += g_weight * gaussian_math(worldPos, g_pos, precision);
    }
    return density;
}

[Differentiable]
float compute_loss(
    no_diff uint3 tid, 
    no_diff float3 minW, 
    no_diff float3 maxW, 
    no_diff uint3 res
) {
    float3 uvw = (float3(tid) + 0.5) / float3(res);
    float3 worldPos = minW + uvw * (maxW - minW);

    float predicted = compute_density_tiled(tid, worldPos);
    float target = no_diff ReferenceVol[tid];

    float diff = predicted - target;
    return diff * diff;
}

// TRAINING
[shader("compute")]
[numthreads(TRAIN_WG_X, TRAIN_WG_Y, TRAIN_WG_Z)]
void train_main(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= volumeResolution)) return;
    
    bwd_diff(compute_loss)(tid, minWorld, maxWorld, volumeResolution, 1.0);
}

// CLEAR GRADIENTS
[shader("compute")]
[numthreads(256, 1, 1)]
void clear_gradients(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < gaussianCount * PARAMS_PER_GAUSSIAN) {
        gGradientsRaw[tid.x] = 0;
    }
}

// SGD OPTIMIZER (GENERIC)
[shader("compute")]
[numthreads(256, 1, 1)]
void optimizer_sgd(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;
    
    // Process all parameters generically
    for (uint paramID = 0; paramID < PARAMS_PER_GAUSSIAN; ++paramID) {
        uint idx = gid * PARAMS_PER_GAUSSIAN + paramID;
        float gradient = asfloat(gGradientsRaw[idx]);
        float currentVal = asfloat(gGaussianParamsRaw[idx]);
        
        // Get metadata
        ParamMetadata meta = PARAM_META[paramID];
        float lr = learningRates[meta.lrGroup];
        
        // Update
        float newVal = currentVal - gradient * lr;
        
        // Clamp
        newVal = clamp(newVal, meta.minVal, meta.maxVal);
        
        gGaussianParamsRaw[idx] = asuint(newVal);
    }

    // quat renormalization
    uint qBase = gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_X;
    float4 q = float4(
        asfloat(gGaussianParamsRaw[qBase + 0]),
        asfloat(gGaussianParamsRaw[qBase + 1]),
        asfloat(gGaussianParamsRaw[qBase + 2]),
        asfloat(gGaussianParamsRaw[qBase + 3])
    );
    float qLen = length(q);
    if (qLen > 1e-6) q /= qLen;
    else q = float4(0, 0, 0, 1);  // fallback to identity
    gGaussianParamsRaw[qBase + 0] = asuint(q.x);
    gGaussianParamsRaw[qBase + 1] = asuint(q.y);
    gGaussianParamsRaw[qBase + 2] = asuint(q.z);
    gGaussianParamsRaw[qBase + 3] = asuint(q.w);
}

// ADAM OPTIMIZER
[shader("compute")]
[numthreads(256, 1, 1)]
void optimizer_adam(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;
    
    // Bias correction terms (computed once per Gaussian)
    float beta1_pow = pow(adamBeta1, float(adamIteration));
    float beta2_pow = pow(adamBeta2, float(adamIteration));
    
    // Process all parameters generically
    for (uint paramID = 0; paramID < PARAMS_PER_GAUSSIAN; ++paramID) {
        uint idx = gid * PARAMS_PER_GAUSSIAN + paramID;
        
        float g_t = asfloat(gGradientsRaw[idx]);
        float currentVal = asfloat(gGaussianParamsRaw[idx]);
        if (paramID >= PARAM_SCALE_X && paramID <= PARAM_SCALE_Z) {
            g_t += currentVal * 0.001;  // L2 penalty — tune this value
        }
        
        // Load previous moments
        float m_t_prev = gAdamFirstMoment[idx];
        float v_t_prev = gAdamSecondMoment[idx];
        
        // Update moments
        float m_t = adamBeta1 * m_t_prev + (1.0 - adamBeta1) * g_t;
        float v_t = adamBeta2 * v_t_prev + (1.0 - adamBeta2) * (g_t * g_t);
        
        // Store moments
        gAdamFirstMoment[idx] = m_t;
        gAdamSecondMoment[idx] = v_t;
        
        // Bias correction
        float m_t_hat = m_t / (1.0 - beta1_pow);
        float v_t_hat = v_t / (1.0 - beta2_pow);
        
        // Get metadata
        ParamMetadata meta = PARAM_META[paramID];
        float lr = learningRates[meta.lrGroup];
        
        // Compute update
        float update = (lr / (sqrt(v_t_hat) + adamEpsilon)) * m_t_hat;
        float newVal = currentVal - update;
        
        // Clamp
        newVal = clamp(newVal, meta.minVal, meta.maxVal);
        
        gGaussianParamsRaw[idx] = asuint(newVal);
    }
    // quat renormalization
    uint qBase = gid * PARAMS_PER_GAUSSIAN + PARAM_QUAT_X;
    float4 q = float4(
        asfloat(gGaussianParamsRaw[qBase + 0]),
        asfloat(gGaussianParamsRaw[qBase + 1]),
        asfloat(gGaussianParamsRaw[qBase + 2]),
        asfloat(gGaussianParamsRaw[qBase + 3])
    );
    float qLen = length(q);
    if (qLen > 1e-6) q /= qLen;
    else q = float4(0, 0, 0, 1);  // fallback to identity
    gGaussianParamsRaw[qBase + 0] = asuint(q.x);
    gGaussianParamsRaw[qBase + 1] = asuint(q.y);
    gGaussianParamsRaw[qBase + 2] = asuint(q.z);
    gGaussianParamsRaw[qBase + 3] = asuint(q.w);
}

// INITIALIZE ADAM STATE
[shader("compute")]
[numthreads(256, 1, 1)]
void init_adam_state(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gaussianCount * PARAMS_PER_GAUSSIAN) return;
    
    gAdamFirstMoment[idx] = 0.0;
    gAdamSecondMoment[idx] = 0.0;
}