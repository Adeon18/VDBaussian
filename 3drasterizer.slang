struct Gaussian
{
    float3 pos;     // world space
    float  sigma;
    float  weight;
};

StructuredBuffer<Gaussian> gGaussians;
RWTexture3D<float> gGaussianVolume;

cbuffer GaussianParams
{
    uint3 volumeResolution;
    uint  gaussianCount;

    float voxelSize;       // Not strictly used if you use min/max world, but good to have
    float3 volumeMinWorld;
    float3 volumeMaxWorld;
};

// Optimization: Pre-calculate constants if possible, or define them here
static const float CUTOFF_RADIUS = 3.0; // Evaluate out to 3 sigmas (99.7% of mass)

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= volumeResolution))
        return;

    // 1. Calculate World Position of this Voxel
    float3 uvw = (float3(tid) + 0.5) / float3(volumeResolution);
    float3 worldPos = volumeMinWorld + uvw * (volumeMaxWorld - volumeMinWorld);

    float density = 0.0;

    // 2. Optimized Loop
    for (uint i = 0; i < gaussianCount; ++i)
    {
        Gaussian g = gGaussians[i];

        // OPTIMIZATION: AABB Check
        // If this voxel is too far from the gaussian center, skip the expensive exp()
        float3 d = worldPos - g.pos;
        float maxDist = g.sigma * CUTOFF_RADIUS;

        // Quick axis-aligned rejection (manhattan or bounding box) is faster than dot(d,d)
        if (abs(d.x) > maxDist || abs(d.y) > maxDist || abs(d.z) > maxDist) 
            continue;

        float r2 = dot(d, d);
        
        // Exact radius check
        if (r2 > maxDist * maxDist) 
            continue;

        float invSigma2 = 1.0 / (g.sigma * g.sigma);
        
        // Standard Gaussian Falloff
        float contribution = exp(-0.5 * r2 * invSigma2);
        
        // Accumulate
        density += g.weight * contribution;
    }

    // 3. Write Output
    gGaussianVolume[tid] = density;
}