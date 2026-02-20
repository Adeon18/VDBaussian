struct Gaussian
{
    float3 pos;
    float3 scale;
    float4 rotation;
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

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + 
           tileID.y * tileResolution.x + tileID.x;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void rasterize_tiled(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= volumeResolution))
        return;

    float3 uvw = (float3(tid) + 0.5) / float3(volumeResolution);
    float3 worldPos = volumeMinWorld + uvw * (volumeMaxWorld - volumeMinWorld);

    uint tileIdx = GetTileIndex(tid);
    uint count = gTileCounts[tileIdx];
    count = min(count, MAX_GAUSSIANS_PER_TILE);

    float density = 0.0;

    for (uint i = 0; i < count; ++i)
    {
        uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        Gaussian g = gGaussians[gid];

        float3 d = worldPos - g.pos;

        float max_scale = max(max(g.scale.x, g.scale.y), g.scale.z);
        float maxDist = max_scale * CUTOFF_RADIUS;

        float r2 = dot(d, d);
        if (r2 > maxDist * maxDist)
            continue;

        float3x3 R = quat_to_matrix(g.rotation);
        float3x3 R_T = transpose(R);
        float3 d_local = mul(R_T, d);

        float3 safe_scale = max(g.scale, float3(0.001, 0.001, 0.001));
        float3 d_scaled = d_local / safe_scale;

        float dist2 = dot(d_scaled, d_scaled);
        float contribution = exp(-0.5 * dist2);
        density += g.weight * contribution;
    }

    gGaussianVolume[tid] = density;
}