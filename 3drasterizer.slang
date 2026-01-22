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

    float voxelSize;
    float3 volumeMinWorld;
    float3 volumeMaxWorld;
};

float gaussian_eval(float3 p, float3 mu, float sigma)
{
    float3 d = p - mu;
    float r2 = dot(d, d);
    float invSigma2 = 1.0 / (sigma * sigma);
    return exp(-0.5 * r2 * invSigma2);
}

[numthreads(4, 4, 4)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid >= volumeResolution))
        return;

    // Voxel center in world space
    float3 uvw = (float3(tid) + 0.5) / volumeResolution; // [0,1]
    float3 worldPos = volumeMinWorld + uvw * (volumeMaxWorld - volumeMinWorld);


    float density = 0.0;

    // Brute-force gather
    for (uint i = 0; i < gaussianCount; ++i)
    {
        Gaussian g = gGaussians[i];
        density += g.weight * gaussian_eval(worldPos, g.pos, g.sigma);
    }

    gGaussianVolume[tid] = density;
}
