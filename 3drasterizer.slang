// 3drasterizer.slang - Unified rasterizer for both Gaussian types

// ==========================================
// GAUSSIAN TYPE CONFIGURATION
// ==========================================

#ifndef GAUSSIAN_TYPE
#define GAUSSIAN_TYPE 1  // 0 = Isotropic, 1 = Anisotropic
#endif

#if GAUSSIAN_TYPE == 0
    // ISOTROPIC
    static const uint PARAMS_PER_GAUSSIAN = 5;
    
    struct Gaussian {
        float3 pos;
        float sigma;
        float weight;
        
        static Gaussian loadFromRaw(uint gaussianID, StructuredBuffer<uint> buffer) {
            Gaussian g;
            uint base = gaussianID * 5;
            g.pos.x = asfloat(buffer[base + 0]);
            g.pos.y = asfloat(buffer[base + 1]);
            g.pos.z = asfloat(buffer[base + 2]);
            g.sigma = asfloat(buffer[base + 3]);
            g.weight = asfloat(buffer[base + 4]);
            return g;
        }
    };
    
    float evaluate_gaussian(Gaussian g, float3 worldPos) {
        float3 d = worldPos - g.pos;
        float r2 = dot(d, d);
        
        // 3σ cutoff
        float maxDist = g.sigma * 3.0;
        if (r2 > maxDist * maxDist)
            return 0.0;
        
        float invSigma2 = 1.0 / (g.sigma * g.sigma);
        return g.weight * exp(-0.5 * r2 * invSigma2);
    }
    
#elif GAUSSIAN_TYPE == 1
    // ANISOTROPIC
    static const uint PARAMS_PER_GAUSSIAN = 11;
    
    struct Gaussian {
        float3 pos;
        float3 scale;
        float4 rotation;  // quaternion
        float weight;
        
        static Gaussian loadFromRaw(uint gaussianID, StructuredBuffer<uint> buffer) {
            Gaussian g;
            uint base = gaussianID * 11;
            g.pos.x = asfloat(buffer[base + 0]);
            g.pos.y = asfloat(buffer[base + 1]);
            g.pos.z = asfloat(buffer[base + 2]);
            g.scale.x = asfloat(buffer[base + 3]);
            g.scale.y = asfloat(buffer[base + 4]);
            g.scale.z = asfloat(buffer[base + 5]);
            g.rotation.w = asfloat(buffer[base + 6]);
            g.rotation.x = asfloat(buffer[base + 7]);
            g.rotation.y = asfloat(buffer[base + 8]);
            g.rotation.z = asfloat(buffer[base + 9]);
            g.weight = asfloat(buffer[base + 10]);
            return g;
        }
    };
    
    // Quaternion to rotation matrix
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
    
    float evaluate_gaussian(Gaussian g, float3 worldPos) {
        float3 d = worldPos - g.pos;
        
        // Conservative culling (largest axis)
        float max_scale = max(max(g.scale.x, g.scale.y), g.scale.z);
        float maxDist = max_scale * 3.0;
        if (dot(d, d) > maxDist * maxDist)
            return 0.0;
        
        // Build rotation matrix
        float3x3 R = quat_to_matrix(g.rotation);
        
        // Transform to local space: d_local = R^T * d
        float3 d_local = mul(d, R);
        
        // Apply inverse scale
        float3 d_scaled = d_local / g.scale;
        
        // Mahalanobis distance
        float dist2 = dot(d_scaled, d_scaled);
        
        return g.weight * exp(-0.5 * dist2);
    }
    
#endif

// ==========================================
// BINDINGS
// ==========================================

[[vk::binding(0)]] StructuredBuffer<uint> gGaussians;  // Raw uint buffer now!
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

uint GetTileIndex(uint3 voxelID) {
    uint3 tileID = voxelID / tileSize;
    tileID = min(tileID, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y) + 
           tileID.y * tileResolution.x + tileID.x;
}

// ==========================================
// RASTERIZATION KERNEL
// ==========================================

[shader("compute")]
[numthreads(4, 4, 4)]
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

    // Loop through Gaussians in this tile
    for (uint i = 0; i < count; ++i)
    {
        uint gid = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        
        // Load Gaussian (type-specific)
        Gaussian g = Gaussian::loadFromRaw(gid, gGaussians);
        
        // Evaluate (type-specific)
        density += evaluate_gaussian(g, worldPos);
    }

    gGaussianVolume[tid] = density;
}