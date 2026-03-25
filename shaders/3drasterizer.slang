// 3drasterizer.slang

struct Gaussian
{
    float3 pos;
    float3 scale;
    float4 quat;
    float weight;
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
    uint  maxGaussiansPerTile;
    float3 volumeMinWorld;
    float3 volumeMaxWorld;
};

static const float CUTOFF_RADIUS = 3.0;

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + 
           tileID.y * tileResolution.x + tileID.x;
}

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
float mahalanobis2(float3 d, float3x3 P)
{
    return dot(d, mul(P, d));
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
    count = min(count, maxGaussiansPerTile);

    float density = 0.0;

    // ONLY loop through Gaussians in this tile!
    for (uint i = 0; i < count; ++i)
    {
        uint gid = gTileContent[tileIdx * maxGaussiansPerTile + i];
        Gaussian g = gGaussians[gid];

        float3 d = worldPos - g.pos;
        float maxScale = max(g.scale.x, max(g.scale.y, g.scale.z));
        float r2 = dot(d, d);
        if (r2 > (maxScale * CUTOFF_RADIUS) * (maxScale * CUTOFF_RADIUS)) continue;

        float3x3 precision = buildPrecisionMatrix(g.scale, g.quat);
        float maha2 = dot(d, mul(precision, d));
        float contribution = exp(-0.5 * maha2);
        density += g.weight * contribution;
    }

    gGaussianVolume[tid] = density;
}