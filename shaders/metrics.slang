// metrics.slang
// Two-tier GPU metrics using tile-accelerated Gaussian density evaluation.
// Same density path as training
//
// Result buffer layout (8 floats, indices below):
//   [0] mse_sum
//   [1] l1_sum
//   [2] iou_intersect
//   [3] iou_union
//   [4] ssim_sum_xy   (accumulated over dispatched slices)
//   [5] ssim_sum_xz
//   [6] ssim_sum_yz
//   [7] voxel_count
//
// Tier 1 — fast_metrics   : PSNR/L1/IoU, one dispatch, every-frame capable
// Tier 2 — ssim_slice     : per-axis 2D SSIM, N slices x 3 axes, every N frames
// Utility — clear_metrics : zero all 8 slots before each collection

[[vk::binding(0)]] Texture3D<float>          gReference;
[[vk::binding(1)]] RWStructuredBuffer<uint>  gGaussianParamsRaw;
[[vk::binding(2)]] RWStructuredBuffer<uint>  gTileContent;
[[vk::binding(3)]] RWStructuredBuffer<uint>  gTileCounts;
[[vk::binding(4)]] RWStructuredBuffer<float> gMetricsResult;

// CONSTANTS
static const uint MAX_GAUSSIANS_PER_TILE = 256;
static const uint PARAMS_PER_GAUSSIAN    = 11;

static const uint PARAM_POS_X   = 0;
static const uint PARAM_POS_Y   = 1;
static const uint PARAM_POS_Z   = 2;
static const uint PARAM_SCALE_X = 3;
static const uint PARAM_SCALE_Y = 4;
static const uint PARAM_SCALE_Z = 5;
static const uint PARAM_QUAT_X  = 6;
static const uint PARAM_QUAT_Y  = 7;
static const uint PARAM_QUAT_Z  = 8;
static const uint PARAM_QUAT_W  = 9;
static const uint PARAM_WEIGHT  = 10;

// Output slot indices in sync with MetricsCollector
static const uint SLOT_MSE_SUM       = 0;
static const uint SLOT_L1_SUM        = 1;
static const uint SLOT_IOU_INTERSECT = 2;
static const uint SLOT_IOU_UNION     = 3;
static const uint SLOT_SSIM_XY       = 4;
static const uint SLOT_SSIM_XZ       = 5;
static const uint SLOT_SSIM_YZ       = 6;
static const uint SLOT_VOXEL_COUNT   = 7;

cbuffer MetricsParams
{
    uint3  volumeResolution;
    uint3  tileResolution;
    float3 minWorld;
    float3 maxWorld;
    uint   tileSize;
    float  iouThreshold;
    float  ssimC1;
    float  ssimC2;
    uint   ssimWindowHalf;
    uint   ssimSliceAxis;    // 0=XY  1=XZ  2=YZ
    uint   ssimSliceIndex;   // which slice along the axis
    float  _pad0;
};

// MATH  (duplicated from training.slang for now)
float3x3 quatToMatrix(float4 q)
{
    float x=q.x, y=q.y, z=q.z, w=q.w;
    return float3x3(
        1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
          2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
          2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)
    );
}

float3x3 buildPrecisionMatrix(float3 scale, float4 quat)
{
    float3x3 R     = quatToMatrix(quat);
    float3   invS2 = 1.0 / (scale * scale);
    float3x3 P;
    P[0][0] = R[0][0]*R[0][0]*invS2.x + R[0][1]*R[0][1]*invS2.y + R[0][2]*R[0][2]*invS2.z;
    P[0][1] = R[0][0]*R[1][0]*invS2.x + R[0][1]*R[1][1]*invS2.y + R[0][2]*R[1][2]*invS2.z;
    P[0][2] = R[0][0]*R[2][0]*invS2.x + R[0][1]*R[2][1]*invS2.y + R[0][2]*R[2][2]*invS2.z;
    P[1][0] = P[0][1];
    P[1][1] = R[1][0]*R[1][0]*invS2.x + R[1][1]*R[1][1]*invS2.y + R[1][2]*R[1][2]*invS2.z;
    P[1][2] = R[1][0]*R[2][0]*invS2.x + R[1][1]*R[2][1]*invS2.y + R[1][2]*R[2][2]*invS2.z;
    P[2][0] = P[0][2];
    P[2][1] = P[1][2];
    P[2][2] = R[2][0]*R[2][0]*invS2.x + R[2][1]*R[2][1]*invS2.y + R[2][2]*R[2][2]*invS2.z;
    return P;
}

// TILE HELPERS
uint getTileIndex(uint3 voxelID)
{
    uint3 tileID = min(voxelID / tileSize, tileResolution - 1);
    return tileID.z * (tileResolution.x * tileResolution.y)
         + tileID.y *  tileResolution.x
         + tileID.x;
}

// Evaluate the true Gaussian field density at a world-space position
// using only the Gaussians registered in the tile for voxelID.
// This is the forward-only version of gsBatch_computeDensity from training.slang.
float evaluateDensityTiled(float3 worldPos, uint3 voxelID)
{
    uint tileIdx = getTileIndex(voxelID);
    uint count   = min(gTileCounts[tileIdx], MAX_GAUSSIANS_PER_TILE);
    float density = 0.0;

    for (uint i = 0; i < count; ++i)
    {
        uint gid  = gTileContent[tileIdx * MAX_GAUSSIANS_PER_TILE + i];
        uint base = gid * PARAMS_PER_GAUSSIAN;

        float3 pos   = float3(asfloat(gGaussianParamsRaw[base + PARAM_POS_X]),
                              asfloat(gGaussianParamsRaw[base + PARAM_POS_Y]),
                              asfloat(gGaussianParamsRaw[base + PARAM_POS_Z]));
        float3 scale = float3(asfloat(gGaussianParamsRaw[base + PARAM_SCALE_X]),
                              asfloat(gGaussianParamsRaw[base + PARAM_SCALE_Y]),
                              asfloat(gGaussianParamsRaw[base + PARAM_SCALE_Z]));
        float4 quat  = float4(asfloat(gGaussianParamsRaw[base + PARAM_QUAT_X]),
                              asfloat(gGaussianParamsRaw[base + PARAM_QUAT_Y]),
                              asfloat(gGaussianParamsRaw[base + PARAM_QUAT_Z]),
                              asfloat(gGaussianParamsRaw[base + PARAM_QUAT_W]));
        float weight = asfloat(gGaussianParamsRaw[base + PARAM_WEIGHT]);

        float3 d        = worldPos - pos;
        float  maxScale = max(scale.x, max(scale.y, scale.z));
        if (dot(d, d) > (maxScale * 3.0) * (maxScale * 3.0)) continue;

        float3x3 P = buildPrecisionMatrix(scale, quat);
        density += weight * exp(-0.5 * dot(d, mul(P, d)));
    }
    return density;
}


void atomicAddFloat(uint slot, float value)
{
    if (value == 0.0) return;
    InterlockedAdd(gMetricsResult[slot], value);
}

// TIER 1 — fast_metrics
// One thread per voxel. Each workgroup reduces into groupshared memory,
// then lane 0 flushes partial sums to global via CAS.
// Workgroup size 512 → one dispatch covers 196³ ≈ 7.5M voxels in ~14k groups.
static const uint FAST_WG = 512;

groupshared float gs_mse [FAST_WG];
groupshared float gs_l1  [FAST_WG];
groupshared float gs_iou_i[FAST_WG];
groupshared float gs_iou_u[FAST_WG];
groupshared float gs_cnt [FAST_WG];

[shader("compute")]
[numthreads(FAST_WG, 1, 1)]
void fast_metrics(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint total = volumeResolution.x * volumeResolution.y * volumeResolution.z;
    uint idx   = tid.x;

    float mse_v=0, l1_v=0, ii_v=0, uu_v=0, cnt_v=0;

    if (idx < total)
    {
        uint z = idx / (volumeResolution.x * volumeResolution.y);
        uint r = idx % (volumeResolution.x * volumeResolution.y);
        uint y = r / volumeResolution.x;
        uint x = r % volumeResolution.x;
        uint3 voxelID = uint3(x, y, z);

        float3 uvw      = (float3(voxelID) + 0.5) / float3(volumeResolution);
        float3 worldPos = minWorld + uvw * (maxWorld - minWorld);

        float pred = evaluateDensityTiled(worldPos, voxelID);
        float ref  = gReference[voxelID];

        float diff = pred - ref;
        mse_v  = diff * diff;
        l1_v   = abs(diff);
        float predBin = (pred > iouThreshold) ? 1.0 : 0.0;
        float refBin  = (ref  > iouThreshold) ? 1.0 : 0.0;
        ii_v   = predBin * refBin;
        uu_v   = max(predBin, refBin);
        cnt_v  = 1.0;
    }

    gs_mse [lane] = mse_v;
    gs_l1  [lane] = l1_v;
    gs_iou_i[lane]= ii_v;
    gs_iou_u[lane]= uu_v;
    gs_cnt [lane] = cnt_v;
    GroupMemoryBarrierWithGroupSync();

    [unroll(9)]
    for (uint stride = FAST_WG >> 1; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            gs_mse [lane] += gs_mse [lane + stride];
            gs_l1  [lane] += gs_l1  [lane + stride];
            gs_iou_i[lane]+= gs_iou_i[lane + stride];
            gs_iou_u[lane]+= gs_iou_u[lane + stride];
            gs_cnt [lane] += gs_cnt [lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0)
    {
        atomicAddFloat(SLOT_MSE_SUM,       gs_mse[0]);
        atomicAddFloat(SLOT_L1_SUM,        gs_l1[0]);
        atomicAddFloat(SLOT_IOU_INTERSECT, gs_iou_i[0]);
        atomicAddFloat(SLOT_IOU_UNION,     gs_iou_u[0]);
        atomicAddFloat(SLOT_VOXEL_COUNT,   gs_cnt[0]);
    }
}

// TIER 2 — ssim_slice
// Computes mean SSIM over one 2D slice of the volume using tile-evaluated
// Gaussian density
// CPU dispatches this 3 × ssim_slices_per_axis times.
// Uses a box window of side (2*ssimWindowHalf+1)^2 for speed.
static const uint SSIM_WG_X     = 16;
static const uint SSIM_WG_Y     = 16;
static const uint SSIM_WG_TOTAL = SSIM_WG_X * SSIM_WG_Y;

groupshared float gs_ssim_acc[SSIM_WG_TOTAL];
groupshared float gs_ssim_cnt[SSIM_WG_TOTAL];

// Map a 2D slice pixel to a 3D voxelID based on axis and slice index
uint3 slicePixelToVoxel(uint axis, uint sliceIdx, int2 uv,
                         uint sliceW, uint sliceH)
{
    int3 coord;
    if      (axis == 0) coord = int3(uv.x, uv.y, int(sliceIdx)); // XY
    else if (axis == 1) coord = int3(uv.x, int(sliceIdx), uv.y); // XZ
    else                coord = int3(int(sliceIdx), uv.x, uv.y); // YZ
    int3 res = int3(volumeResolution);
    return uint3(clamp(coord, int3(0,0,0), res - 1));
}

// Sample predicted density for a slice pixel using tile evaluation
float samplePredSlice(uint axis, uint sliceIdx, int2 uv, uint sliceW, uint sliceH)
{
    uint3  vox      = slicePixelToVoxel(axis, sliceIdx, uv, sliceW, sliceH);
    float3 uvw      = (float3(vox) + 0.5) / float3(volumeResolution);
    float3 worldPos = minWorld + uvw * (maxWorld - minWorld);
    return evaluateDensityTiled(worldPos, vox);
}

// Sample reference density for a slice pixel directly from texture
float sampleRefSlice(uint axis, uint sliceIdx, int2 uv,
                      uint sliceW, uint sliceH)
{
    uint3 vox = slicePixelToVoxel(axis, sliceIdx, uv, sliceW, sliceH);
    return gReference[vox];
}

[shader("compute")]
[numthreads(SSIM_WG_X, SSIM_WG_Y, 1)]
void ssim_slice(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint sliceW, sliceH;
    if      (ssimSliceAxis == 0) { sliceW = volumeResolution.x; sliceH = volumeResolution.y; }
    else if (ssimSliceAxis == 1) { sliceW = volumeResolution.x; sliceH = volumeResolution.z; }
    else                         { sliceW = volumeResolution.y; sliceH = volumeResolution.z; }

    int2 pixel = int2(tid.xy);
    bool valid = (pixel.x < int(sliceW)) && (pixel.y < int(sliceH));

    float ssim_v = 0.0, cnt_v = 0.0;

    if (valid)
    {
        int half = int(ssimWindowHalf);
        float mu_x=0, mu_y=0, n=0;

        for (int dy = -half; dy <= half; ++dy)
        for (int dx = -half; dx <= half; ++dx)
        {
            int2 s = clamp(pixel + int2(dx,dy),
                           int2(0,0), int2(int(sliceW)-1, int(sliceH)-1));
            mu_x += samplePredSlice(ssimSliceAxis, ssimSliceIndex, s, sliceW, sliceH);
            mu_y += sampleRefSlice (ssimSliceAxis, ssimSliceIndex, s, sliceW, sliceH);
            n    += 1.0;
        }
        mu_x /= n; mu_y /= n;

        float sx2=0, sy2=0, sxy=0;
        for (int dy = -half; dy <= half; ++dy)
        for (int dx = -half; dx <= half; ++dx)
        {
            int2 s = clamp(pixel + int2(dx,dy),
                           int2(0,0), int2(int(sliceW)-1, int(sliceH)-1));
            float px = samplePredSlice(ssimSliceAxis, ssimSliceIndex, s, sliceW, sliceH) - mu_x;
            float py = sampleRefSlice (ssimSliceAxis, ssimSliceIndex, s, sliceW, sliceH) - mu_y;
            sx2 += px*px; sy2 += py*py; sxy += px*py;
        }
        sx2 /= n; sy2 /= n; sxy /= n;

        float num = (2.0*mu_x*mu_y + ssimC1) * (2.0*sxy + ssimC2);
        float den = (mu_x*mu_x + mu_y*mu_y + ssimC1) * (sx2 + sy2 + ssimC2);
        ssim_v = (den > 0.0) ? num / den : 1.0;
        cnt_v  = 1.0;
    }

    gs_ssim_acc[lane] = ssim_v;
    gs_ssim_cnt[lane] = cnt_v;
    GroupMemoryBarrierWithGroupSync();

    [unroll(8)]
    for (uint stride = SSIM_WG_TOTAL >> 1; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            gs_ssim_acc[lane] += gs_ssim_acc[lane + stride];
            gs_ssim_cnt[lane] += gs_ssim_cnt[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0 && gs_ssim_cnt[0] > 0.0)
        atomicAddFloat(SLOT_SSIM_XY + ssimSliceAxis, gs_ssim_acc[0]);
}

// Zero all 8 result slots
[shader("compute")]
[numthreads(8, 1, 1)]
void clear_metrics(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < 8)
        gMetricsResult[tid.x] = 0.0;
}