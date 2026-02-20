static const uint TRAIN_WG_X = 4;
static const uint TRAIN_WG_Y = 4;
static const uint TRAIN_WG_Z = 4;
static const uint TRAIN_WG_SIZE = TRAIN_WG_X * TRAIN_WG_Y * TRAIN_WG_Z;

static const uint MAX_GAUSSIANS_PER_TILE = 64;

static const uint PARAM_POS_X = 0;
static const uint PARAM_POS_Y = 1;
static const uint PARAM_POS_Z = 2;
static const uint PARAM_SCALE_X = 3;
static const uint PARAM_SCALE_Y = 4;
static const uint PARAM_SCALE_Z = 5;
static const uint PARAM_QUAT_W = 6;
static const uint PARAM_QUAT_X = 7;
static const uint PARAM_QUAT_Y = 8;
static const uint PARAM_QUAT_Z = 9;
static const uint PARAM_WEIGHT = 10;
static const uint PARAMS_PER_GAUSSIAN = 11;

static const uint LR_GROUP_POSITION = 0;
static const uint LR_GROUP_SCALE = 1;
static const uint LR_GROUP_ROTATION = 2;
static const uint LR_GROUP_WEIGHT = 3;
static const uint NUM_LR_GROUPS = 4;

struct Gaussian 
{
    float3 pos;
    float3 scale;
    float4 rotation;
    float weight;
    
    static float getParam(uint gaussianID, uint paramID)
    {
        return asfloat(gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID]);
    }
    
    static void setParam(uint gaussianID, uint paramID, float value)
    {
        gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID] = asuint(value);
    }
    
    static Gaussian load(uint gaussianID)
    {
        Gaussian g;
        g.pos.x = getParam(gaussianID, PARAM_POS_X);
        g.pos.y = getParam(gaussianID, PARAM_POS_Y);
        g.pos.z = getParam(gaussianID, PARAM_POS_Z);
        g.scale.x = getParam(gaussianID, PARAM_SCALE_X);
        g.scale.y = getParam(gaussianID, PARAM_SCALE_Y);
        g.scale.z = getParam(gaussianID, PARAM_SCALE_Z);
        g.rotation.w = getParam(gaussianID, PARAM_QUAT_W);
        g.rotation.x = getParam(gaussianID, PARAM_QUAT_X);
        g.rotation.y = getParam(gaussianID, PARAM_QUAT_Y);
        g.rotation.z = getParam(gaussianID, PARAM_QUAT_Z);
        g.weight = getParam(gaussianID, PARAM_WEIGHT);
        return g;
    }
    
    static void store(uint gaussianID, Gaussian g)
    {
        setParam(gaussianID, PARAM_POS_X, g.pos.x);
        setParam(gaussianID, PARAM_POS_Y, g.pos.y);
        setParam(gaussianID, PARAM_POS_Z, g.pos.z);
        setParam(gaussianID, PARAM_SCALE_X, g.scale.x);
        setParam(gaussianID, PARAM_SCALE_Y, g.scale.y);
        setParam(gaussianID, PARAM_SCALE_Z, g.scale.z);
        setParam(gaussianID, PARAM_QUAT_W, g.rotation.w);
        setParam(gaussianID, PARAM_QUAT_X, g.rotation.x);
        setParam(gaussianID, PARAM_QUAT_Y, g.rotation.y);
        setParam(gaussianID, PARAM_QUAT_Z, g.rotation.z);
        setParam(gaussianID, PARAM_WEIGHT, g.weight);
    }
};

[[vk::binding(0)]] Texture3D<float> ReferenceVol;
[[vk::binding(1)]] RWStructuredBuffer<uint> gGaussianParamsRaw;
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
    
    float learningRates[NUM_LR_GROUPS];
    
    uint3 tileResolution;
    uint totalTiles;
    uint tileSize;
    
    float adamBeta1;
    float adamBeta2;
    float adamEpsilon;
    uint adamIteration;

    uint debugMode;
    float debugScale;
};

static const uint DEBUG_NONE = 0;
static const uint DEBUG_LOSS = 1;
static const uint DEBUG_GRADIENT_MAG = 2;
static const uint DEBUG_TILE_DENSITY = 3;
static const uint DEBUG_OVERLAP = 4;
static const uint DEBUG_PRED_VS_TARGET = 5;

static const float DEBUG_LOSS_SCALE = 20.0;
static const float DEBUG_GRADIENT_SCALE = 5.0;
static const float DEBUG_TILE_SCALE = 1.0 / 32.0;
static const float DEBUG_OVERLAP_SCALE = 2.0;
static const float DEBUG_ERROR_SCALE = 10.0;

static const float DEBUG_ALPHA_MULT_LOSS = 100.0;
static const float DEBUG_ALPHA_MULT_GRADIENT = 50.0;
static const float DEBUG_ALPHA_MULT_TILE = 2.0;
static const float DEBUG_ALPHA_MULT_OVERLAP = 5.0;
static const float DEBUG_ALPHA_MULT_ERROR = 20.0;

static const float HEATMAP_LOW_THRESHOLD = 0.2;
static const float HEATMAP_HIGH_THRESHOLD = 0.8;

struct ParamMetadata
{
    uint lrGroup;
    float minVal;
    float maxVal;
};

static const ParamMetadata PARAM_META[PARAMS_PER_GAUSSIAN] = {
    { LR_GROUP_POSITION, -1e6, 1e6 },
    { LR_GROUP_POSITION, -1e6, 1e6 },
    { LR_GROUP_POSITION, -1e6, 1e6 },
    { LR_GROUP_SCALE, 0.01, 1e6 },
    { LR_GROUP_SCALE, 0.01, 1e6 },
    { LR_GROUP_SCALE, 0.01, 1e6 },
    { LR_GROUP_ROTATION, -1.0, 1.0 },
    { LR_GROUP_ROTATION, -1.0, 1.0 },
    { LR_GROUP_ROTATION, -1.0, 1.0 },
    { LR_GROUP_ROTATION, -1.0, 1.0 },
    { LR_GROUP_WEIGHT, 0.0, 1.0 },
};

int3 WorldToVoxel(float3 pos) {
    float3 uvw = (pos - minWorld) / (maxWorld - minWorld);
    return int3(floor(uvw * float3(volumeResolution)));
}

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + tileID.y * tileResolution.x + tileID.x;
}

uint GetLocalThreadIndex(uint3 tid) {
    uint3 local = tid % uint3(TRAIN_WG_X, TRAIN_WG_Y, TRAIN_WG_Z);
    return local.x + local.y * TRAIN_WG_X + local.z * TRAIN_WG_X * TRAIN_WG_Y;
}

float3 heatmapColor(float t) {
    t = clamp(t, 0.0, 1.0);
    t = pow(t, 0.7);
    
    if (t < 0.33) {
        return lerp(float3(0.0, 0.8, 0.0), float3(0.5, 1.0, 0.0), t * 3.0);
    } else if (t < 0.67) {
        return lerp(float3(0.5, 1.0, 0.0), float3(1.0, 0.6, 0.0), (t - 0.33) * 3.0);
    } else {
        return lerp(float3(1.0, 0.6, 0.0), float3(1.0, 0.0, 0.0), (t - 0.67) * 3.0);
    }
}

float3 divergingColor(float t) {
    t = clamp(t, -1.0, 1.0);
    
    if (t < -0.1) {
        float blend = clamp((-t - 0.1) / 0.9, 0.0, 1.0);
        return lerp(float3(0.7, 0.9, 1.0), float3(0.0, 0.0, 1.0), blend);
    } else if (t < 0.1) {
        float blend = t / 0.1 * 0.5 + 0.5;
        return lerp(float3(0.7, 0.9, 1.0), float3(1.0, 0.9, 0.7), blend);
    } else {
        float blend = clamp((t - 0.1) / 0.9, 0.0, 1.0);
        return lerp(float3(1.0, 0.9, 0.7), float3(1.0, 0.0, 0.0), blend);
    }
}

float3x3 quat_to_matrix(float4 q) {
    q = normalize(q);
    
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;
    
    float3x3 R;
    R[0] = float3(1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy));
    R[1] = float3(2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx));
    R[2] = float3(2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy));
    
    return R;
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
        uint tileIdx = GetTileIndex(tid);
        uint count = gTileCounts[tileIdx];
        count = min(count, MAX_GAUSSIANS_PER_TILE);
        
        float totalGradMag = 0.0;
        for (uint i = 0; i < count; ++i) {
            uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
            
            float gradMag = 0.0;
            for (uint p = 0; p < PARAMS_PER_GAUSSIAN; ++p) {
                float g = asfloat(gGradientsRaw[gid * PARAMS_PER_GAUSSIAN + p]);
                gradMag += g * g;
            }
            totalGradMag += sqrt(gradMag);
        }
        
        rawValue = totalGradMag;
        normalizedValue = totalGradMag * DEBUG_GRADIENT_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = min(totalGradMag * DEBUG_ALPHA_MULT_GRADIENT * debugScale, 1.0);
    }
    else if (debugMode == DEBUG_TILE_DENSITY) {
        uint tileIdx = GetTileIndex(tid);
        uint count = gTileCounts[tileIdx];
        
        rawValue = float(count);
        normalizedValue = float(count) * DEBUG_TILE_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = normalizedValue;
    }
    else if (debugMode == DEBUG_OVERLAP) {
        float density = compute_density_tiled(tid, worldPos);
        
        rawValue = density;
        normalizedValue = density * DEBUG_OVERLAP_SCALE * debugScale;
        debugColor = heatmapColor(normalizedValue);
        debugAlpha = min(density * DEBUG_ALPHA_MULT_OVERLAP * debugScale, 1.0);
    }
    else if (debugMode == DEBUG_PRED_VS_TARGET) {
        float predicted = compute_density_tiled(tid, worldPos);
        float target = ReferenceVol[tid];
        float error = predicted - target;
        
        rawValue = error;
        normalizedValue = error * DEBUG_ERROR_SCALE * debugScale;
        debugColor = divergingColor(normalizedValue);
        debugAlpha = min(abs(error) * DEBUG_ALPHA_MULT_ERROR * debugScale, 1.0);
    }
    
    gDebugVolume[tid] = float4(debugColor, debugAlpha);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clear_tiles(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= totalTiles) return;
    gTileCounts[tid.x] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bin_gaussians(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;

    Gaussian g = Gaussian::load(gid);
    
    float max_scale = max(max(g.scale.x, g.scale.y), g.scale.z);
    float radius = max_scale * 3.0;
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
float gaussian_math_aniso(float3 p, float3 mu, float3 scale, float4 quat) {
    float3 d = p - mu;
    
    float3x3 R = no_diff quat_to_matrix(quat);
    float3x3 R_T = transpose(R);
    float3 d_local = mul(R_T, d);
    
    float3 d_scaled = d_local / scale;
    float dist2 = dot(d_scaled, d_scaled);
    
    return exp(-0.5 * dist2);
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
        float3 g_scale = float3(
            loadParam(gid, PARAM_SCALE_X, localThreadIdx),
            loadParam(gid, PARAM_SCALE_Y, localThreadIdx),
            loadParam(gid, PARAM_SCALE_Z, localThreadIdx)
        );
        float4 g_quat = float4(
            loadParam(gid, PARAM_QUAT_W, localThreadIdx),
            loadParam(gid, PARAM_QUAT_X, localThreadIdx),
            loadParam(gid, PARAM_QUAT_Y, localThreadIdx),
            loadParam(gid, PARAM_QUAT_Z, localThreadIdx)
        );
        float g_weight = loadParam(gid, PARAM_WEIGHT, localThreadIdx);
        
        float max_scale = max(max(g_scale.x, g_scale.y), g_scale.z);
        float3 d = worldPos - g_pos;
        if (no_diff(dot(d,d) > (max_scale * 3.0)*(max_scale * 3.0))) continue;
        
        density += g_weight * gaussian_math_aniso(worldPos, g_pos, g_scale, g_quat);
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

[shader("compute")]
[numthreads(TRAIN_WG_X, TRAIN_WG_Y, TRAIN_WG_Z)]
void train_main(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= volumeResolution)) return;
    
    bwd_diff(compute_loss)(tid, minWorld, maxWorld, volumeResolution, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clear_gradients(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < gaussianCount * PARAMS_PER_GAUSSIAN) {
        gGradientsRaw[tid.x] = 0;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void optimizer_sgd(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;
    
    for (uint paramID = 0; paramID < PARAMS_PER_GAUSSIAN; ++paramID) {
        uint idx = gid * PARAMS_PER_GAUSSIAN + paramID;
        float gradient = asfloat(gGradientsRaw[idx]);
        float currentVal = asfloat(gGaussianParamsRaw[idx]);
        
        ParamMetadata meta = PARAM_META[paramID];
        float lr = learningRates[meta.lrGroup];
        
        float newVal = currentVal - gradient * lr;
        newVal = clamp(newVal, meta.minVal, meta.maxVal);
        
        gGaussianParamsRaw[idx] = asuint(newVal);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void optimizer_adam(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;
    
    float beta1_pow = pow(adamBeta1, float(adamIteration));
    float beta2_pow = pow(adamBeta2, float(adamIteration));
    
    for (uint paramID = 0; paramID < PARAMS_PER_GAUSSIAN; ++paramID) {
        uint idx = gid * PARAMS_PER_GAUSSIAN + paramID;
        
        float g_t = asfloat(gGradientsRaw[idx]);
        float currentVal = asfloat(gGaussianParamsRaw[idx]);
        
        float m_t_prev = gAdamFirstMoment[idx];
        float v_t_prev = gAdamSecondMoment[idx];
        
        float m_t = adamBeta1 * m_t_prev + (1.0 - adamBeta1) * g_t;
        float v_t = adamBeta2 * v_t_prev + (1.0 - adamBeta2) * (g_t * g_t);
        
        gAdamFirstMoment[idx] = m_t;
        gAdamSecondMoment[idx] = v_t;
        
        float m_t_hat = m_t / (1.0 - beta1_pow);
        float v_t_hat = v_t / (1.0 - beta2_pow);
        
        ParamMetadata meta = PARAM_META[paramID];
        float lr = learningRates[meta.lrGroup];
        
        float update = (lr / (sqrt(v_t_hat) + adamEpsilon)) * m_t_hat;
        float newVal = currentVal - update;
        
        newVal = clamp(newVal, meta.minVal, meta.maxVal);
        
        gGaussianParamsRaw[idx] = asuint(newVal);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void init_adam_state(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gaussianCount * PARAMS_PER_GAUSSIAN) return;
    
    gAdamFirstMoment[idx] = 0.0;
    gAdamSecondMoment[idx] = 0.0;
}