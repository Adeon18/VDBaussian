// training.slang

// Training workgroup dimensions (must match kernel dispatch)
static const uint TRAIN_WG_X = 4;
static const uint TRAIN_WG_Y = 4;
static const uint TRAIN_WG_Z = 4;
static const uint TRAIN_WG_SIZE = TRAIN_WG_X * TRAIN_WG_Y * TRAIN_WG_Z;

// DEFINITIONS
static const uint MAX_GAUSSIANS_PER_TILE = 64;

// PARAMETER LAYOUT CONFIGURATION
static const uint PARAM_POS_X = 0;
static const uint PARAM_POS_Y = 1;
static const uint PARAM_POS_Z = 2;
static const uint PARAM_SIGMA = 3;
static const uint PARAM_WEIGHT = 4;
static const uint PARAMS_PER_GAUSSIAN = 5;

// Learning rate groups (indices to learningRates arr)
static const uint LR_GROUP_POSITION = 0;
static const uint LR_GROUP_SIGMA = 1;
static const uint LR_GROUP_WEIGHT = 2;
static const uint NUM_LR_GROUPS = 3;

// GAUSSIAN STRUCTURE
struct Gaussian 
{
    float3 pos;
    float sigma;
    float weight;
    
    static float getParam(uint gaussianID, uint paramID)
    {
        return asfloat(gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID]);
    }
    
    static void setParam(uint gaussianID, uint paramID, float value)
    {
        gGaussianParamsRaw[gaussianID * PARAMS_PER_GAUSSIAN + paramID] = asuint(value);
    }
    
    // Convenience loaders
    static Gaussian load(uint gaussianID)
    {
        Gaussian g;
        g.pos.x = getParam(gaussianID, PARAM_POS_X);
        g.pos.y = getParam(gaussianID, PARAM_POS_Y);
        g.pos.z = getParam(gaussianID, PARAM_POS_Z);
        g.sigma = getParam(gaussianID, PARAM_SIGMA);
        g.weight = getParam(gaussianID, PARAM_WEIGHT);
        return g;
    }
    
    static void store(uint gaussianID, Gaussian g)
    {
        setParam(gaussianID, PARAM_POS_X, g.pos.x);
        setParam(gaussianID, PARAM_POS_Y, g.pos.y);
        setParam(gaussianID, PARAM_POS_Z, g.pos.z);
        setParam(gaussianID, PARAM_SIGMA, g.sigma);
        setParam(gaussianID, PARAM_WEIGHT, g.weight);
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
};

// Maps each parameter to its learning rate group and constraints
struct ParamMetadata
{
    uint lrGroup;
    float minVal;
    float maxVal;
};

static const ParamMetadata PARAM_META[PARAMS_PER_GAUSSIAN] = {
    { LR_GROUP_POSITION, -1e6, 1e6 },   // pos.x
    { LR_GROUP_POSITION, -1e6, 1e6 },   // pos.y
    { LR_GROUP_POSITION, -1e6, 1e6 },   // pos.z
    { LR_GROUP_SIGMA,    0.01, 1e6 },   // sigma
    { LR_GROUP_WEIGHT,   0.0,  1.0 },   // weight
};

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
    
    float radius = g.sigma * 3.0;
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
float gaussian_math(float3 p, float3 mu, float sigma) {
    float3 d = p - mu;
    float r2 = dot(d, d);
    return exp(-0.5 * r2 / (sigma * sigma));
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
        
        // Load via generic accessor
        float3 g_pos = float3(
            loadParam(gid, PARAM_POS_X, localThreadIdx),
            loadParam(gid, PARAM_POS_Y, localThreadIdx),
            loadParam(gid, PARAM_POS_Z, localThreadIdx)
        );
        float g_sigma = loadParam(gid, PARAM_SIGMA, localThreadIdx);
        float g_weight = loadParam(gid, PARAM_WEIGHT, localThreadIdx);
        
        float3 d = worldPos - g_pos;
        if (no_diff(dot(d,d) > (g_sigma * 3.0)*(g_sigma * 3.0))) continue;

        density += g_weight * gaussian_math(worldPos, g_pos, g_sigma);
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