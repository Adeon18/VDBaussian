// screen_metrics.slang
// Renders reference and prediction volumes from multiple canonical cameras,
// then computes 2D PSNR / SSIM / IoU / L1 pixel-wise.
// This file is essentially the raybmarcher copy but with functionality to get the metrics
//
// Result buffer layout:
//   [0]  mse_sum
//   [1]  l1_sum
//   [2]  iou_intersect
//   [3]  iou_union
//   [4]  ssim_sum
//   [5]  ssim_count
//   [6]  pixel_count
//   [7]  _pad

[[vk::binding(0)]] Texture3D<float>          gRefVolume;
[[vk::binding(1)]] Texture3D<float>          gPredVolume;
[[vk::binding(2)]] SamplerState              gLinearSampler;
[[vk::binding(3)]] RWStructuredBuffer<float> gScreenMetrics;
[[vk::binding(4)]] RWTexture2D<float4>       gRefImage;
[[vk::binding(5)]] RWTexture2D<float4>       gPredImage;

static const uint SLOT_MSE_SUM       = 0;
static const uint SLOT_L1_SUM        = 1;
static const uint SLOT_IOU_INTERSECT = 2;
static const uint SLOT_IOU_UNION     = 3;
static const uint SLOT_SSIM_SUM      = 4;
static const uint SLOT_SSIM_COUNT    = 5;
static const uint SLOT_PIXEL_COUNT   = 6;

static const float SHADOW_BIAS       = 0.01;
static const int   MAX_ITERATIONS    = 2000;
static const float DENSITY_THRESHOLD = 0.001;
static const float OPACITY_THRESHOLD = 0.99;

cbuffer ScreenMetricsParams
{
    float3 camPos;
    float  _pad0;
    float3 camFront;
    float  _pad1;
    float3 camRight;
    float  _pad2;
    float3 camUp;
    float  _pad3;

    float  stepSize;
    float  densityScale;
    float  densityCurve;
    int    maxSteps;
    float3 smokeAlbedo;
    float  _pad4;
    float  lightPenetration;
    float  phaseG;
    float  _pad5;
    float  _pad6;
    float3 sunDir;
    float  _pad7;
    float3 sunColor;
    float  _pad8;
    float3 ambientColor;
    float  _pad9;
    int    shadowSteps;
    float  shadowStepMult;

    uint   screenW;
    uint   screenH;

    float  iouLumaThreshold;
    float  ssimC1;
    float  ssimC2;
    int    ssimHalf;
};

float henyeyGreenstein(float g, float costh)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * costh, 1.5));
}

float2 hitBox(float3 orig, float3 dir)
{
    const float3 box_min = float3(-1.0);
    const float3 box_max = float3( 1.0);
    float3 inv_dir = 1.0 / dir;
    float3 t0 = (box_min - orig) * inv_dir;
    float3 t1 = (box_max - orig) * inv_dir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    return float2(max(max(tmin.x, tmin.y), tmin.z),
                  min(min(tmax.x, tmax.y), tmax.z));
}

float shadowMarch(Texture3D<float> vol, float3 p)
{
    float t   = SHADOW_BIAS;
    float acc = 0.0;
    for (int j = 0; j < shadowSteps; ++j)
    {
        float3 lp = p + sunDir * t;
        if (any(abs(lp) > 1.0)) break;
        acc += vol.SampleLevel(gLinearSampler, lp * 0.5 + 0.5, 0);
        t   += stepSize * shadowStepMult;
    }
    return exp(-acc * densityScale * lightPenetration * stepSize * shadowStepMult);
}

float3 raymarch(Texture3D<float> vol, float3 ro, float3 rd, float2 uv, float tNear, float tFar)
{
    float3 bg       = lerp(float3(0.1, 0.1, 0.1), float3(0.0, 0.0, 0.0), uv.y);
    float  phaseVal = henyeyGreenstein(phaseG, dot(rd, sunDir));
    float  noise    = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
    float  t        = max(0.0, tNear) + noise * stepSize;
    float4 color    = float4(0.0, 0.0, 0.0, 0.0);

    for (int i = 0; i < maxSteps; ++i)
    {
        if (t > tFar || i >= MAX_ITERATIONS) break;
        if (color.a > OPACITY_THRESHOLD)     break;

        float3 p = ro + rd * t;
        float  d = vol.SampleLevel(gLinearSampler, p * 0.5 + 0.5, 0);

        if (d > DENSITY_THRESHOLD)
        {
            float  shape = pow(d, densityCurve);
            float  alpha = shape * densityScale * stepSize;
            float  trans = shadowMarch(vol, p);
            float3 light = sunColor * trans * phaseVal + ambientColor;
            color.rgb   += (1.0 - color.a) * smokeAlbedo * light * alpha;
            color.a     += (1.0 - color.a) * alpha;
        }
        t += stepSize;
    }

    return color.rgb + bg * (1.0 - color.a);
}

// PASS 1: render reference and prediction into off-screen textures
[shader("compute")]
[numthreads(8, 8, 1)]
void render_both(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= screenW || tid.y >= screenH) return;

    float2 uv  = (float2(tid.xy) + 0.5) / float2(screenW, screenH);
    float2 scr = uv * 2.0 - 1.0;
    float3 rd  = normalize(camFront + camRight * scr.x + camUp * scr.y);

    float2 bounds = hitBox(camPos, rd);
    bool   missed = (bounds.x > bounds.y || bounds.y < 0.0);
    float3 bg     = lerp(float3(0.1, 0.1, 0.1), float3(0.0, 0.0, 0.0), uv.y);

    float3 refCol  = missed ? bg : raymarch(gRefVolume,  camPos, rd, uv, bounds.x, bounds.y);
    float3 predCol = missed ? bg : raymarch(gPredVolume, camPos, rd, uv, bounds.x, bounds.y);
    gRefImage[tid.xy]  = float4(saturate(refCol),  1.0);
    gPredImage[tid.xy] = float4(saturate(predCol), 1.0);
}

// PASS 2: pixel-wise metrics with groupshared reduction
static const uint METRIC_WG = 256;
groupshared float gs_mse  [METRIC_WG];
groupshared float gs_l1   [METRIC_WG];
groupshared float gs_iou_i[METRIC_WG];
groupshared float gs_iou_u[METRIC_WG];
groupshared float gs_ssim [METRIC_WG];
groupshared float gs_cnt  [METRIC_WG];

float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[shader("compute")]
[numthreads(METRIC_WG, 1, 1)]
void compute_screen_metrics(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint total = screenW * screenH;
    uint idx   = tid.x;

    float mse_v=0, l1_v=0, ii_v=0, uu_v=0, ssim_v=0, cnt_v=0;

    if (idx < total)
    {
        int2 px    = int2(idx % screenW, idx / screenW);
        float refL = luma(gRefImage[px].rgb);
        float predL= luma(gPredImage[px].rgb);
        float diff = predL - refL;

        mse_v = diff * diff;
        l1_v  = abs(diff);
        float predBin = (predL > iouLumaThreshold) ? 1.0 : 0.0;
        float refBin  = (refL  > iouLumaThreshold) ? 1.0 : 0.0;
        ii_v  = predBin * refBin;
        uu_v  = max(predBin, refBin);
        cnt_v = 1.0;

        // SSIM box window
        float mu_x=0, mu_y=0, n=0;
        for (int dy = -ssimHalf; dy <= ssimHalf; ++dy)
        for (int dx = -ssimHalf; dx <= ssimHalf; ++dx)
        {
            int2 s = clamp(px + int2(dx, dy),
                           int2(0, 0),
                           int2(int(screenW)-1, int(screenH)-1));
            mu_x += luma(gRefImage[s].rgb);
            mu_y += luma(gPredImage[s].rgb);
            n    += 1.0;
        }
        mu_x /= n; mu_y /= n;

        float sx2=0, sy2=0, sxy=0;
        for (int dy = -ssimHalf; dy <= ssimHalf; ++dy)
        for (int dx = -ssimHalf; dx <= ssimHalf; ++dx)
        {
            int2  s  = clamp(px + int2(dx, dy),
                             int2(0, 0),
                             int2(int(screenW)-1, int(screenH)-1));
            float px2 = luma(gRefImage[s].rgb)  - mu_x;
            float py2 = luma(gPredImage[s].rgb) - mu_y;
            sx2 += px2*px2; sy2 += py2*py2; sxy += px2*py2;
        }
        sx2 /= n; sy2 /= n; sxy /= n;

        float num = (2.0*mu_x*mu_y + ssimC1) * (2.0*sxy     + ssimC2);
        float den = (mu_x*mu_x + mu_y*mu_y + ssimC1) * (sx2 + sy2 + ssimC2);
        ssim_v = (den > 0.0) ? num / den : 1.0;
    }

    gs_mse  [lane] = mse_v;
    gs_l1   [lane] = l1_v;
    gs_iou_i[lane] = ii_v;
    gs_iou_u[lane] = uu_v;
    gs_ssim [lane] = ssim_v;
    gs_cnt  [lane] = cnt_v;
    GroupMemoryBarrierWithGroupSync();

    [unroll(8)]
    for (uint stride = METRIC_WG >> 1; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            gs_mse  [lane] += gs_mse  [lane + stride];
            gs_l1   [lane] += gs_l1   [lane + stride];
            gs_iou_i[lane] += gs_iou_i[lane + stride];
            gs_iou_u[lane] += gs_iou_u[lane + stride];
            gs_ssim [lane] += gs_ssim [lane + stride];
            gs_cnt  [lane] += gs_cnt  [lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0)
    {
        InterlockedAdd(gScreenMetrics[SLOT_MSE_SUM],       gs_mse[0]);
        InterlockedAdd(gScreenMetrics[SLOT_L1_SUM],        gs_l1[0]);
        InterlockedAdd(gScreenMetrics[SLOT_IOU_INTERSECT], gs_iou_i[0]);
        InterlockedAdd(gScreenMetrics[SLOT_IOU_UNION],     gs_iou_u[0]);
        InterlockedAdd(gScreenMetrics[SLOT_SSIM_SUM],      gs_ssim[0]);
        InterlockedAdd(gScreenMetrics[SLOT_SSIM_COUNT],    gs_cnt[0]);
        InterlockedAdd(gScreenMetrics[SLOT_PIXEL_COUNT],   gs_cnt[0]);
    }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void clear_screen_metrics(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < 8) gScreenMetrics[tid.x] = 0.0;
}