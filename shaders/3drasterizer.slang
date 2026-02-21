struct Gaussian
{
    float3 pos;
    float  sigma;
    float  weight;
};

[[vk::binding(0)]] StructuredBuffer<Gaussian> gGaussians;
[[vk::binding(1)]] RWTexture3D<float> gGaussianVolume;
[[vk::binding(2)]] StructuredBuffer<uint> gTileContent;
[[vk::binding(3)]] StructuredBuffer<uint> gTileCounts;

cbuffer RasterParams
{
    uint3 volumeResolution;
    uint3 tileResolution;
    uint  tileSize;
    float3 volumeMinWorld;
    float3 volumeMaxWorld;
};

static const uint MAX_GAUSSIANS_PER_TILE = 64;
static const float CUTOFF_RADIUS = 3.0;

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + 
           tileID.y * tileResolution.x + tileID.x;
}

[shader("compute")]
[numthreads(4, 4, 4)]  // Match training workgroup size
void rasterize_tiled(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= volumeResolution))
        return;

    // Calculate world position
    float3 uvw = (float3(tid) + 0.5) / float3(volumeResolution);
    float3 worldPos = volumeMinWorld + uvw * (volumeMaxWorld - volumeMinWorld);

    // Get tile index
    uint tileIdx = GetTileIndex(tid);
    uint count = gTileCounts[tileIdx];
    count = min(count, MAX_GAUSSIANS_PER_TILE);

    float density = 0.0;

    // ONLY loop through Gaussians in this tile!
    for (uint i = 0; i < count; ++i)
    {
        uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        Gaussian g = gGaussians[gid];

        float3 d = worldPos - g.pos;
        float maxDist = g.sigma * CUTOFF_RADIUS;

        // Distance check
        float r2 = dot(d, d);
        if (r2 > maxDist * maxDist)
            continue;

        // Gaussian contribution
        float invSigma2 = 1.0 / (g.sigma * g.sigma);
        float contribution = exp(-0.5 * r2 * invSigma2);
        density += g.weight * contribution;
    }

    gGaussianVolume[tid] = density;
}