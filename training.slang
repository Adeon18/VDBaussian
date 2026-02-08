// training.slang

// --- DEFINITIONS ---
static const uint MAX_GAUSSIANS_PER_TILE = 64; 

// --- BINDINGS ---
struct Gaussian { float3 pos; float sigma; float weight; };

[[vk::binding(0)]] Texture3D<float> ReferenceVol;
[[vk::binding(1)]] RWStructuredBuffer<Gaussian> gGaussians;
[[vk::binding(2)]] RWStructuredBuffer<uint> gGradientsRaw;

// --- NEW: THE TILE BUFFER ---
// flattened size = (Res/4)^3 * MAX_GAUSSIANS_PER_TILE
// Content: List of Gaussian Indices. 
// We also need a counter buffer to know how many are in each tile.
[[vk::binding(3)]] RWStructuredBuffer<uint> gTileContent; 
[[vk::binding(4)]] RWStructuredBuffer<uint> gTileCounts; 

cbuffer TrainParams {
    uint3 volumeResolution;
    uint  gaussianCount;
    float3 minWorld;
    float3 maxWorld;
    float learningRatePos;
    float learningRateSigma;
    float learningRateWeight;
    uint3 tileResolution; // volumeResolution / TILE_SIZE
    uint totalTiles;
    uint tileSize;
};

// --- HELPERS ---
int3 WorldToVoxel(float3 pos) {
    float3 uvw = (pos - minWorld) / (maxWorld - minWorld);
    return int3(floor(uvw * float3(volumeResolution)));
}

// Map a Voxel Coordinate to a Flattened Tile Index
uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    // Clamp to be safe
    tileID = min(tileID, tileResolution - 1);
    // Flatten: z * (w*h) + y * w + x
    return tileID.z * (tileResolution.x * tileResolution.y) + tileID.y * tileResolution.x + tileID.x;
}

// --- KERNEL 1: CLEAR TILES ---
[shader("compute")]
[numthreads(256, 1, 1)]
void clear_tiles(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= totalTiles) return;
    gTileCounts[tid.x] = 0;
}

// --- KERNEL 2: BIN GAUSSIANS (The Tiler) ---
// One thread per Gaussian. Writes itself into the tiles it touches.
[shader("compute")]
[numthreads(256, 1, 1)]
void bin_gaussians(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= gaussianCount) return;

    Gaussian g = gGaussians[gid];
    
    // 1. Calculate Bounding Box in Tile Space
    float radius = g.sigma;
    int3 min_vox = WorldToVoxel(g.pos - radius);
    int3 max_vox = WorldToVoxel(g.pos + radius);
    
    uint3 min_tile = max(int3(0), min_vox) / tileSize;
    uint3 max_tile = min(int3(volumeResolution)/tileSize - 1, max_vox / tileSize);
    
    // 2. Loop over touched tiles and append ID
    for (uint z = min_tile.z; z <= max_tile.z; ++z) {
        for (uint y = min_tile.y; y <= max_tile.y; ++y) {
            for (uint x = min_tile.x; x <= max_tile.x; ++x) {
                
                uint tileIdx = z * (tileResolution.x * tileResolution.y) + y * tileResolution.x + x;
                
                // Atomic Increment to get slot
                uint slot;
                InterlockedAdd(gTileCounts[tileIdx], 1, slot);
                
                // If tile is not full, write ID
                if (slot < MAX_GAUSSIANS_PER_TILE) {
                    uint bufferIdx = tileIdx * MAX_GAUSSIANS_PER_TILE + slot;
                    gTileContent[bufferIdx] = gid;
                }
            }
        }
    }
}

// --- STANDARD ATOMIC ACCUMULATORS ---
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

[BackwardDerivative(load_pos_bwd)]
float3 load_pos(uint index) { return gGaussians[index].pos; }
void load_pos_bwd(uint index, float3 dOut) {
    uint base = index * 5;
    atomicAccumulate(base+0, dOut.x);
    atomicAccumulate(base+1, dOut.y);
    atomicAccumulate(base+2, dOut.z);
}
[BackwardDerivative(load_sigma_bwd)]
float load_sigma(uint index) { return gGaussians[index].sigma; }
void load_sigma_bwd(uint index, float dOut) { atomicAccumulate(index*5+3, dOut); }
[BackwardDerivative(load_weight_bwd)]
float load_weight(uint index) { return gGaussians[index].weight; }
void load_weight_bwd(uint index, float dOut) { atomicAccumulate(index*5+4, dOut); }

// --- MATH ---
[Differentiable]
float gaussian_math(float3 p, float3 mu, float sigma) {
    float3 d = p - mu;
    float r2 = dot(d, d);
    return exp(-0.5 * r2 / (sigma * sigma));
}

// --- DENSITY CALCULATION (TILED) ---
[Differentiable]
float compute_density_tiled(no_diff uint3 tid, no_diff float3 worldPos) {
    float density = 0.0;
    
    uint tileIdx = GetTileIndex(tid);
    uint count = gTileCounts[tileIdx];
    count = min(count, MAX_GAUSSIANS_PER_TILE);
    
    // FIX 1: [ForceUnroll]
    // This forces the compiler to generate the code for all 64 iterations,
    // which ensures the custom derivative (load_pos_bwd) is correctly linked.
    [ForceUnroll] 
    for(uint i=0; i<MAX_GAUSSIANS_PER_TILE; ++i) {
        // Dynamic check to break early (simulates dynamic loop)
        if (i >= count) break;

        uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        
        float3 g_pos = load_pos(gid);
        float  g_sigma = load_sigma(gid);
        float  g_weight = load_weight(gid);
        
        float3 d = worldPos - g_pos;
        if (no_diff(dot(d,d) > (g_sigma)*(g_sigma))) continue;

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

// --- TRAIN KERNEL ---
[shader("compute")]
[numthreads(4, 4, 4)]
void train_main(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= volumeResolution)) return;
    // Note: We removed 'gaussianCount' from signature as it's not needed for the tiled loop
    bwd_diff(compute_loss)(tid, minWorld, maxWorld, volumeResolution, 1.0);
}

// --- GRADIENT CLEAR & OPTIMIZER REMAIN THE SAME ---
[shader("compute")]
[numthreads(256, 1, 1)]
void clear_gradients(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < gaussianCount * 5) gGradientsRaw[tid.x] = 0;
}
[shader("compute")]
[numthreads(256, 1, 1)]
void optimizer_main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gaussianCount) return;
    uint base = idx * 5;
    float3 dPos   = float3(asfloat(gGradientsRaw[base]), asfloat(gGradientsRaw[base+1]), asfloat(gGradientsRaw[base+2]));
    float dSigma  = asfloat(gGradientsRaw[base+3]);
    float dWeight = asfloat(gGradientsRaw[base+4]);
    Gaussian g = gGaussians[idx];
    g.pos    -= dPos * learningRatePos;
    g.sigma  -= dSigma * learningRateSigma;
    g.weight -= dWeight * learningRateWeight;
    g.sigma = max(0.01, g.sigma);
    g.weight = max(0.0, g.weight);
    gGaussians[idx] = g;
}