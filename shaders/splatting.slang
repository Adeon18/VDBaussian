// splatting.slang
// Volumetric Gaussian Splatting renderer
// Uses closed-form ray integral through 3D Gaussians for physically-based volumetric opacity.
//
// The line integral of a 3D Gaussian N(mu, Sigma) with amplitude w along ray r(t) = o + t*d is:
//   integral = w * sqrt(2*PI / (d^T P d)) * exp(-0.5 * q_perp)
// where P = Sigma^{-1} and q_perp is the perpendicular Mahalanobis distance.
// The exp(-0.5 * q_perp) term is exactly the 2D projected Gaussian evaluated at the pixel.

// Constants
static const uint PARAMS_PER_GAUSSIAN = 11;
static const uint SCREEN_TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_SCREEN_TILE = 2048;
static const float CUTOFF_SIGMA = 3.0;
static const float MIN_DEPTH = 0.2;
static const float OPACITY_THRESHOLD = 0.999;
static const float PI = 3.14159265358979;

// Data structures

struct Gaussian
{
    float3 pos;
    float3 scale;
    float4 quat;
    float weight;
};

// Intermediate projected data
// Layout: [center.x, center.y, depth, opacityScale, conic.a, conic.b, conic.c,
//          bbox_minTX, bbox_minTY, bbox_maxTX, bbox_maxTY]
// = 11 floats = 44 bytes
static const uint PROJ_CENTER_X    = 0;
static const uint PROJ_CENTER_Y    = 1;
static const uint PROJ_DEPTH       = 2;
static const uint PROJ_OPACITY     = 3;
static const uint PROJ_CONIC_A     = 4;
static const uint PROJ_CONIC_B     = 5;
static const uint PROJ_CONIC_C     = 6;
static const uint PROJ_BBOX_MIN_TX = 7;
static const uint PROJ_BBOX_MIN_TY = 8;
static const uint PROJ_BBOX_MAX_TX = 9;
static const uint PROJ_BBOX_MAX_TY = 10;
static const uint PROJ_STRIDE      = 11;

// Bindings
[[vk::binding(0)]] StructuredBuffer<Gaussian> gGaussians;
[[vk::binding(1)]] StructuredBuffer<float> gCamera;
[[vk::binding(2)]] StructuredBuffer<float> gSettings;
[[vk::binding(3)]] RWStructuredBuffer<float> gProjected;            // flat float array
[[vk::binding(4)]] RWStructuredBuffer<uint> gSortedIndices;    
[[vk::binding(5)]] RWStructuredBuffer<uint> gScreenTileContent;
[[vk::binding(6)]] RWStructuredBuffer<uint> gScreenTileCounts;
[[vk::binding(7)]] RWTexture2D<float4> gOutputTexture;
[[vk::binding(8)]] RWStructuredBuffer<uint> gSortKeys;              // depth as sortable uint
[[vk::binding(9)]] RWStructuredBuffer<float> gLightTransmittance;   // per-gaussian self-shadowing
[[vk::binding(10)]] RWTexture2D<float4> gBlurOutput;                // temp texture for blur pass

cbuffer SplatParams
{
    uint  gaussianCount;
    uint  screenWidth;
    uint  screenHeight;
    uint  paddedSortCount;  // gaussianCount padded to next power of 2 (for sort init in projection)
    uint2 tileResolution;
    float densityScale;
    float lightPenetration;  // controls shadow density (matches raymarcher usage)
    float shadowStrength;    // multiplier on self-shadow transmittance (1.0 = physical, higher = darker shadows)
    float splatSoftness;     // low-pass filter on 2D covariance (0.3 = sharp/default, higher = softer edges)
    uint  useDepthDarkening; // 0 = off, 1 = view-dependent depth darkening
    uint  blurRadius;        // screen-space blur radius (0 = off, 1-4 = kernel half-size)
    float3 volumeMin;   // world-space AABB of the gaussian volume
    float3 volumeMax;
};

// ============================================================
// Math helpers
// ============================================================

float3x3 quatToMatrix(float4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
          2*(x*y + w*z),   1 - 2*(x*x + z*z),     2*(y*z - w*x),
          2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x + y*y)
    );
}

float3x3 buildCovarianceMatrix(float3 scale, float4 quat)
{
    float3x3 R = quatToMatrix(quat);
    float3 s2 = scale * scale;
    float3x3 S;
    S[0][0] = R[0][0]*R[0][0]*s2.x + R[0][1]*R[0][1]*s2.y + R[0][2]*R[0][2]*s2.z;
    S[0][1] = R[0][0]*R[1][0]*s2.x + R[0][1]*R[1][1]*s2.y + R[0][2]*R[1][2]*s2.z;
    S[0][2] = R[0][0]*R[2][0]*s2.x + R[0][1]*R[2][1]*s2.y + R[0][2]*R[2][2]*s2.z;
    S[1][0] = S[0][1];
    S[1][1] = R[1][0]*R[1][0]*s2.x + R[1][1]*R[1][1]*s2.y + R[1][2]*R[1][2]*s2.z;
    S[1][2] = R[1][0]*R[2][0]*s2.x + R[1][1]*R[2][1]*s2.y + R[1][2]*R[2][2]*s2.z;
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[2][2] = R[2][0]*R[2][0]*s2.x + R[2][1]*R[2][1]*s2.y + R[2][2]*R[2][2]*s2.z;
    return S;
}

float3x3 buildPrecisionMatrix(float3 scale, float4 quat)
{
    float3x3 R = quatToMatrix(quat);
    float3 invS2 = 1.0 / (scale * scale);
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

float henyeyGreenstein(float g, float costh)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * costh, 1.5));
}

// ============================================================
// Projected-buffer helpers (avoid struct padding issues)
// ============================================================

void projWrite(uint gid, uint field, float val)
{
    gProjected[gid * PROJ_STRIDE + field] = val;
}

float projRead(uint gid, uint field)
{
    return gProjected[gid * PROJ_STRIDE + field];
}

// ============================================================
// Pass 1: Project gaussians to screen space
// ============================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void project_gaussians(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;

    // Handle sort padding elements (gaussianCount <= gid < paddedSortCount)
    if (gid >= gaussianCount) {
        if (gid < paddedSortCount) {
            gSortKeys[gid] = 0xFFFFFFFFu;
            gSortedIndices[gid] = gid;
        }
        return;
    }

    Gaussian g = gGaussians[gid];

    // Mark invalid by default
    projWrite(gid, PROJ_DEPTH, -1.0);
    gSortKeys[gid] = 0xFFFFFFFFu;
    gSortedIndices[gid] = gid;

    if (g.weight < 1e-5)
        return;

    // Remap gaussian from VDB world space to [-1,1] cube space (where the camera lives)
    float3 volExtent = volumeMax - volumeMin;
    g.pos   = (g.pos - volumeMin) / volExtent * 2.0 - 1.0;
    g.scale = g.scale / volExtent * 2.0;

    // Camera
    float3 camPos   = float3(gCamera[0], gCamera[1], gCamera[2]);
    float3 camFront = float3(gCamera[4], gCamera[5], gCamera[6]);
    float3 camRight = float3(gCamera[8], gCamera[9], gCamera[10]);  // right_hat * aspect * 0.5
    float3 camUp    = float3(gCamera[12], gCamera[13], gCamera[14]); // up_hat * 0.5

    float3 rightHat = normalize(camRight);
    float3 upHat    = normalize(camUp);
    // Focal length = height (derived from up_scale = 0.5 → tan(half_fov) = 0.5 → f = h/(2*0.5) = h)
    float focal = float(screenHeight);

    float3 delta = g.pos - camPos;
    float cx = dot(delta, rightHat);
    float cy = dot(delta, upHat);
    float cz = dot(delta, camFront);

    if (cz < MIN_DEPTH)
        return;

    // Project to pixel coords.
    // Y is negated so that camera-up maps to smaller pixel.y (= screen top)
    // matching the convention where pixel (0,0) = top-left in compute shaders.
    float2 center;
    center.x =  focal * cx / cz + float(screenWidth)  * 0.5;
    center.y = -focal * cy / cz + float(screenHeight) * 0.5;

    // ---- 2D covariance via EWA splatting ----
    // World-to-camera rotation
    float3x3 W = float3x3(
        rightHat.x, rightHat.y, rightHat.z,
        upHat.x,    upHat.y,    upHat.z,
        camFront.x, camFront.y, camFront.z
    );

    float3x3 Sigma = buildCovarianceMatrix(g.scale, g.quat);
    float3x3 SigmaCam = mul(mul(W, Sigma), transpose(W));

    // Jacobian of perspective projection (camera-space → pixel-space)
    // Note: j12 sign is negated to match the Y-flip in the projection above
    float inv_cz  = 1.0 / cz;
    float inv_cz2 = inv_cz * inv_cz;
    float j00 =  focal * inv_cz;
    float j02 = -focal * cx * inv_cz2;
    float j11 = -focal * inv_cz;          // negative because of Y-flip
    float j12 =  focal * cy * inv_cz2;    // sign follows from chain rule on -cy/cz

    // J * SigmaCam  (2x3)
    float3 js0 = float3(j00 * SigmaCam[0][0] + j02 * SigmaCam[2][0],
                         j00 * SigmaCam[0][1] + j02 * SigmaCam[2][1],
                         j00 * SigmaCam[0][2] + j02 * SigmaCam[2][2]);
    float3 js1 = float3(j11 * SigmaCam[1][0] + j12 * SigmaCam[2][0],
                         j11 * SigmaCam[1][1] + j12 * SigmaCam[2][1],
                         j11 * SigmaCam[1][2] + j12 * SigmaCam[2][2]);

    // (J * SigmaCam) * J^T → 2x2 covariance
    float cov_xx = js0.x * j00 + js0.z * j02;
    float cov_xy = js0.y * j11 + js0.z * j12;
    float cov_yy = js1.y * j11 + js1.z * j12;

    // Low-pass filter — higher splatSoftness blurs gaussian footprints, hiding hard edges
    cov_xx += splatSoftness;
    cov_yy += splatSoftness;

    // Invert 2x2 → conic
    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 1e-6)
        return;
    float inv_det = 1.0 / det;
    float conic_a =  cov_yy * inv_det;
    float conic_b = -cov_xy * inv_det;
    float conic_c =  cov_xx * inv_det;

    // Bounding box (3-sigma ellipse → AABB in tile coords)
    float radius_x = ceil(CUTOFF_SIGMA * sqrt(cov_xx));
    float radius_y = ceil(CUTOFF_SIGMA * sqrt(cov_yy));

    int bb_min_tx = int(max(0.0, center.x - radius_x)) / int(SCREEN_TILE_SIZE);
    int bb_min_ty = int(max(0.0, center.y - radius_y)) / int(SCREEN_TILE_SIZE);
    int bb_max_tx = int(min(float(screenWidth  - 1), center.x + radius_x)) / int(SCREEN_TILE_SIZE);
    int bb_max_ty = int(min(float(screenHeight - 1), center.y + radius_y)) / int(SCREEN_TILE_SIZE);

    if (bb_min_tx > int(tileResolution.x - 1) || bb_max_tx < 0 ||
        bb_min_ty > int(tileResolution.y - 1) || bb_max_ty < 0)
        return;

    bb_min_tx = max(bb_min_tx, 0);
    bb_min_ty = max(bb_min_ty, 0);
    bb_max_tx = min(bb_max_tx, int(tileResolution.x - 1));
    bb_max_ty = min(bb_max_ty, int(tileResolution.y - 1));

    // Volumetric ray-integral scale factor
    // The closed-form line integral of G(r(t)) = w * exp(-0.5 * (r(t)-mu)^T P (r(t)-mu)) is:
    //   integral = w * sqrt(2*PI / (d^T P d)) * exp(-0.5 * q_perp)
    // After coordinate remapping to [-1,1], the integral shrinks proportionally.
    // densityScale compensates and lets the user control overall opacity.
    float3x3 P = buildPrecisionMatrix(g.scale, g.quat);
    float2 screenUV = center / float2(screenWidth, screenHeight) * 2.0 - 1.0;
    float3 rayDir = normalize(camFront + camRight * screenUV.x + camUp * screenUV.y);
    float dPd = max(dot(rayDir, mul(P, rayDir)), 1e-6);
    float opacityScale = g.weight * sqrt(2.0 * PI / dPd) * densityScale;

    // Write all fields
    projWrite(gid, PROJ_CENTER_X,    center.x);
    projWrite(gid, PROJ_CENTER_Y,    center.y);
    projWrite(gid, PROJ_DEPTH,       cz);
    projWrite(gid, PROJ_OPACITY,     opacityScale);
    projWrite(gid, PROJ_CONIC_A,     conic_a);
    projWrite(gid, PROJ_CONIC_B,     conic_b);
    projWrite(gid, PROJ_CONIC_C,     conic_c);
    projWrite(gid, PROJ_BBOX_MIN_TX, asfloat(bb_min_tx));
    projWrite(gid, PROJ_BBOX_MIN_TY, asfloat(bb_min_ty));
    projWrite(gid, PROJ_BBOX_MAX_TX, asfloat(bb_max_tx));
    projWrite(gid, PROJ_BBOX_MAX_TY, asfloat(bb_max_ty));

    // Camera sort key — written here to avoid a separate init dispatch
    gSortKeys[gid] = depthToSortKey(cz);
}

// ============================================================
// GPU Sort: Local bitonic sort in shared memory + global merge
// ============================================================

// Convert float depth to a uint that sorts in the same order (positive depths only, invalid → max).
uint depthToSortKey(float depth)
{
    if (depth <= 0.0)
        return 0xFFFFFFFFu;
    uint bits = asuint(depth);
    return bits ^ 0x80000000u;
}

// Convert any float (including negative) to a sortable uint preserving total order.
uint floatToSortKey(float val)
{
    uint bits = asuint(val);
    uint mask = (bits >> 31) != 0 ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

cbuffer SortParams
{
    uint sortCount;     // padded to next power of 2
    uint blockSize;     // for global merge: current bitonic block size
    uint subBlockSize;  // for global merge: current comparison distance
    float lightDirX;    // for light-direction sorting
    float lightDirY;
    float lightDirZ;
};

// LOCAL_SORT_SIZE must be power of 2 and match numthreads * 2
static const uint LOCAL_SORT_SIZE = 4096;
static const uint LOCAL_SORT_THREADS = 1024;

groupshared uint gs_keys[LOCAL_SORT_SIZE];
groupshared uint gs_vals[LOCAL_SORT_SIZE];

// ============================================================
// Sort key initialization kernels
// ============================================================

// Init sort keys from camera depth (call before local_bitonic_sort for camera-order sorting)
[shader("compute")]
[numthreads(256, 1, 1)]
void init_sort_keys_depth(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= sortCount) return;

    if (idx < gaussianCount) {
        gSortKeys[idx] = depthToSortKey(projRead(idx, PROJ_DEPTH));
    } else {
        gSortKeys[idx] = 0xFFFFFFFFu;
    }
    gSortedIndices[idx] = idx;
}

// Init sort keys from light-direction depth (call before local_bitonic_sort for light-order sorting)
[shader("compute")]
[numthreads(256, 1, 1)]
void init_sort_keys_light(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= sortCount) return;

    if (idx < gaussianCount) {
        Gaussian g = gGaussians[idx];
        float3 volExtent = volumeMax - volumeMin;
        float3 pos = (g.pos - volumeMin) / volExtent * 2.0 - 1.0;
        float3 ld = float3(lightDirX, lightDirY, lightDirZ);
        float lightDepth = dot(pos, ld);
        gSortKeys[idx] = floatToSortKey(lightDepth);
    } else {
        gSortKeys[idx] = 0xFFFFFFFFu;
    }
    gSortedIndices[idx] = idx;
}

// ============================================================
// Transmittance: parallel tau compute + lightweight serial scan
// ============================================================

// Step 1: Compute per-gaussian tau in parallel (one thread per gaussian).
// Writes tau into gLightTransmittance[] indexed by gaussian ID (reused as temp storage).
[shader("compute")]
[numthreads(256, 1, 1)]
void compute_gaussian_tau(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gaussianCount) return;

    Gaussian g = gGaussians[idx];
    if (g.weight < 1e-5) {
        gLightTransmittance[idx] = 0.0;
        return;
    }

    float3 ld = normalize(float3(lightDirX, lightDirY, lightDirZ));
    float3 volExtent = volumeMax - volumeMin;

    // Remap to [-1,1] cube space
    g.scale = g.scale / volExtent * 2.0;

    // Line integral through gaussian center along light direction
    float3x3 P = buildPrecisionMatrix(g.scale, g.quat);
    float dPd = max(dot(ld, mul(P, ld)), 1e-6);
    float lineIntegral = g.weight * sqrt(2.0 * PI / dPd);

    // Mean-field correction: weight by projected cross-section / volume cross-section
    float3x3 R = quatToMatrix(g.quat);
    float3 localLD = float3(
        R[0][0]*ld.x + R[1][0]*ld.y + R[2][0]*ld.z,
        R[0][1]*ld.x + R[1][1]*ld.y + R[2][1]*ld.z,
        R[0][2]*ld.x + R[1][2]*ld.y + R[2][2]*ld.z
    );
    float dSd = localLD.x*localLD.x * g.scale.x*g.scale.x
               + localLD.y*localLD.y * g.scale.y*g.scale.y
               + localLD.z*localLD.z * g.scale.z*g.scale.z;
    float projArea = g.scale.x * g.scale.y * g.scale.z / sqrt(max(dSd, 1e-10));
    float coverage = min(PI * projArea / 4.0, 1.0);

    gLightTransmittance[idx] = lineIntegral * densityScale * lightPenetration * coverage * shadowStrength;
}

// Step 2: Serial exclusive prefix sum + exp — reads precomputed taus, writes final transmittance.
// Lightweight: just float additions and one exp per gaussian, no matrix math.
[shader("compute")]
[numthreads(1, 1, 1)]
void scan_transmittance(uint3 tid : SV_DispatchThreadID)
{
    float accumulated_tau = 0.0;

    for (uint i = 0; i < gaussianCount; i++)
    {
        uint gid = gSortedIndices[i];
        float tau = gLightTransmittance[gid];
        gLightTransmittance[gid] = exp(-accumulated_tau);
        accumulated_tau += tau;
    }
}

// ============================================================
// Phase 1: Sort blocks of LOCAL_SORT_SIZE entirely in shared memory (ONE dispatch).
// Reads pre-filled gSortKeys/gSortedIndices (call init_sort_keys_* first).
// ============================================================
[shader("compute")]
[numthreads(LOCAL_SORT_THREADS, 1, 1)]
void local_bitonic_sort(uint3 groupID : SV_GroupID, uint localIdx : SV_GroupIndex)
{
    uint blockStart = groupID.x * LOCAL_SORT_SIZE;
    static const uint ELEMS_PER_THREAD = LOCAL_SORT_SIZE / LOCAL_SORT_THREADS; // 4

    // Load 4 elements per thread into shared memory
    [ForceUnroll]
    for (uint e = 0; e < ELEMS_PER_THREAD; e++)
    {
        uint ldsIdx = localIdx + e * LOCAL_SORT_THREADS;
        uint globalIdx = blockStart + ldsIdx;
        gs_keys[ldsIdx] = (globalIdx < sortCount) ? gSortKeys[globalIdx] : 0xFFFFFFFFu;
        gs_vals[ldsIdx] = (globalIdx < sortCount) ? gSortedIndices[globalIdx] : globalIdx;
    }

    GroupMemoryBarrierWithGroupSync();

    // Full bitonic sort within shared memory
    for (uint blockSz = 2; blockSz <= LOCAL_SORT_SIZE; blockSz <<= 1)
    {
        for (uint subBlock = blockSz >> 1; subBlock >= 1; subBlock >>= 1)
        {
            [ForceUnroll]
            for (uint pass = 0; pass < ELEMS_PER_THREAD; pass++)
            {
                uint idx = pass * LOCAL_SORT_THREADS + localIdx;
                uint grpIdx = idx / subBlock;
                uint grpOff = idx % subBlock;
                uint left  = grpIdx * subBlock * 2 + grpOff;
                uint right = left + subBlock;

                if (right < LOCAL_SORT_SIZE)
                {
                    bool ascending = ((left & blockSz) == 0);
                    uint kL = gs_keys[left];
                    uint kR = gs_keys[right];
                    if (ascending ? (kL > kR) : (kL < kR))
                    {
                        gs_keys[left] = kR;
                        gs_keys[right] = kL;
                        uint vL = gs_vals[left];
                        gs_vals[left] = gs_vals[right];
                        gs_vals[right] = vL;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write back
    [ForceUnroll]
    for (uint e = 0; e < ELEMS_PER_THREAD; e++)
    {
        uint ldsIdx = localIdx + e * LOCAL_SORT_THREADS;
        uint globalIdx = blockStart + ldsIdx;
        if (globalIdx < sortCount) {
            gSortKeys[globalIdx] = gs_keys[ldsIdx];
            gSortedIndices[globalIdx] = gs_vals[ldsIdx];
        }
    }
}

// Phase 2: Global bitonic merge for blocks > LOCAL_SORT_SIZE.
// One dispatch per (blockSize, subBlockSize) pair where subBlockSize >= LOCAL_SORT_SIZE.
[shader("compute")]
[numthreads(256, 1, 1)]
void global_bitonic_merge(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= sortCount)
        return;

    uint partner = idx ^ subBlockSize;
    if (partner <= idx || partner >= sortCount)
        return;

    uint keyA = gSortKeys[idx];
    uint keyB = gSortKeys[partner];

    bool ascending = ((idx & blockSize) == 0);
    bool needSwap = ascending ? (keyA > keyB) : (keyA < keyB);

    if (needSwap)
    {
        gSortKeys[idx] = keyB;
        gSortKeys[partner] = keyA;

        uint valA = gSortedIndices[idx];
        uint valB = gSortedIndices[partner];
        gSortedIndices[idx] = valB;
        gSortedIndices[partner] = valA;
    }
}

// Phase 2b: After global merge brings subBlockSize below LOCAL_SORT_SIZE,
// finish the remaining sub-steps in shared memory (ONE dispatch).
[shader("compute")]
[numthreads(LOCAL_SORT_THREADS, 1, 1)]
void local_merge_finish(uint3 groupID : SV_GroupID, uint localIdx : SV_GroupIndex)
{
    uint blockStart = groupID.x * LOCAL_SORT_SIZE;
    static const uint ELEMS_PER_THREAD = LOCAL_SORT_SIZE / LOCAL_SORT_THREADS;

    // Load
    [ForceUnroll]
    for (uint e = 0; e < ELEMS_PER_THREAD; e++)
    {
        uint ldsIdx = localIdx + e * LOCAL_SORT_THREADS;
        uint globalIdx = blockStart + ldsIdx;
        gs_keys[ldsIdx] = (globalIdx < sortCount) ? gSortKeys[globalIdx] : 0xFFFFFFFFu;
        gs_vals[ldsIdx] = (globalIdx < sortCount) ? gSortedIndices[globalIdx] : globalIdx;
    }

    GroupMemoryBarrierWithGroupSync();

    // Finish merge steps for subBlock sizes from LOCAL_SORT_SIZE/2 down to 1
    for (uint subBlock = LOCAL_SORT_SIZE >> 1; subBlock >= 1; subBlock >>= 1)
    {
        [ForceUnroll]
        for (uint pass = 0; pass < ELEMS_PER_THREAD; pass++)
        {
            uint idx = pass * LOCAL_SORT_THREADS + localIdx;
            uint grpIdx = idx / subBlock;
            uint grpOff = idx % subBlock;
            uint left  = grpIdx * subBlock * 2 + grpOff;
            uint right = left + subBlock;

            if (right < LOCAL_SORT_SIZE)
            {
                // Direction is determined by the global blockSize
                bool ascending = (((blockStart + left) & blockSize) == 0);
                uint kL = gs_keys[left];
                uint kR = gs_keys[right];
                if (ascending ? (kL > kR) : (kL < kR))
                {
                    gs_keys[left] = kR;
                    gs_keys[right] = kL;
                    uint vL = gs_vals[left];
                    gs_vals[left] = gs_vals[right];
                    gs_vals[right] = vL;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write back
    [ForceUnroll]
    for (uint e = 0; e < ELEMS_PER_THREAD; e++)
    {
        uint ldsIdx = localIdx + e * LOCAL_SORT_THREADS;
        uint globalIdx = blockStart + ldsIdx;
        if (globalIdx < sortCount) {
            gSortKeys[globalIdx] = gs_keys[ldsIdx];
            gSortedIndices[globalIdx] = gs_vals[ldsIdx];
        }
    }
}

// ============================================================
// Pass 2: Clear screen tile counters
// ============================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void clear_screen_tiles(uint3 tid : SV_DispatchThreadID)
{
    uint totalTiles = tileResolution.x * tileResolution.y;
    if (tid.x >= totalTiles)
        return;
    gScreenTileCounts[tid.x] = 0;
}

// ============================================================
// Pass 3: Assign sorted gaussians to screen tiles
// ============================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_assign_screen(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gaussianCount)
        return;

    uint gid = gSortedIndices[tid.x];
    float depth = projRead(gid, PROJ_DEPTH);
    if (depth < 0.0)
        return;

    int bb_min_tx = asint(asuint(projRead(gid, PROJ_BBOX_MIN_TX)));
    int bb_min_ty = asint(asuint(projRead(gid, PROJ_BBOX_MIN_TY)));
    int bb_max_tx = asint(asuint(projRead(gid, PROJ_BBOX_MAX_TX)));
    int bb_max_ty = asint(asuint(projRead(gid, PROJ_BBOX_MAX_TY)));

    for (int ty = bb_min_ty; ty <= bb_max_ty; ty++)
    {
        for (int tx = bb_min_tx; tx <= bb_max_tx; tx++)
        {
            uint tileIdx = uint(ty) * tileResolution.x + uint(tx);
            uint slot;
            InterlockedAdd(gScreenTileCounts[tileIdx], 1, slot);
            if (slot < MAX_GAUSSIANS_PER_SCREEN_TILE)
            {
                gScreenTileContent[tileIdx * MAX_GAUSSIANS_PER_SCREEN_TILE + slot] = gid;
            }
        }
    }
}

// ============================================================
// Pass 4: Render splats (one workgroup = one tile)
// ============================================================

groupshared uint gs_tileCount;
groupshared uint gs_gaussianIDs[MAX_GAUSSIANS_PER_SCREEN_TILE];

[shader("compute")]
[numthreads(SCREEN_TILE_SIZE, SCREEN_TILE_SIZE, 1)]
void render_splats(uint3 tid : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint localIdx : SV_GroupIndex)
{
    uint2 pixel = tid.xy;
    uint tileIdx = groupID.y * tileResolution.x + groupID.x;

    // Load tile data cooperatively
    if (localIdx == 0)
        gs_tileCount = min(gScreenTileCounts[tileIdx], MAX_GAUSSIANS_PER_SCREEN_TILE);
    GroupMemoryBarrierWithGroupSync();

    uint count = gs_tileCount;
    for (uint i = localIdx; i < count; i += SCREEN_TILE_SIZE * SCREEN_TILE_SIZE)
        gs_gaussianIDs[i] = gScreenTileContent[tileIdx * MAX_GAUSSIANS_PER_SCREEN_TILE + i];
    GroupMemoryBarrierWithGroupSync();

    if (pixel.x >= screenWidth || pixel.y >= screenHeight)
        return;

    // Lighting — per-gaussian self-shadowing via analytic transmittance
    float3 smokeAlbedo  = float3(gSettings[4], gSettings[5], gSettings[6]);
    float  phaseG       = gSettings[8];
    float3 sunDir       = normalize(float3(gSettings[9], gSettings[10], gSettings[11]));
    float3 sunColorBase = float3(gSettings[12], gSettings[13], gSettings[14]);
    float3 ambientColor = float3(gSettings[15], gSettings[16], gSettings[17]);

    float3 camFront = float3(gCamera[4], gCamera[5], gCamera[6]);
    float3 camRight = float3(gCamera[8], gCamera[9], gCamera[10]);
    float3 camUp    = float3(gCamera[12], gCamera[13], gCamera[14]);

    float2 screenUV = (float2(pixel) + 0.5) / float2(screenWidth, screenHeight) * 2.0 - 1.0;
    float3 rayDir = normalize(camFront + camRight * screenUV.x + camUp * screenUV.y);
    float phase = henyeyGreenstein(phaseG, dot(rayDir, sunDir));

    // Per-pixel base light (transmittance applied per-gaussian below)
    // lightPenetration now controls shadow density in the sweep, not sun brightness
    float3 sunLight = sunColorBase * phase;

    // Background
    float bgT = float(pixel.y) / float(screenHeight);
    float3 bg = lerp(float3(0.1, 0.1, 0.1), float3(0.0, 0.0, 0.0), bgT);

    // Front-to-back compositing
    float3 accColor = float3(0.0);
    float  accAlpha = 0.0;
    float2 pixelCenter = float2(pixel) + 0.5;

    for (uint i = 0; i < count; i++)
    {
        if (accAlpha > OPACITY_THRESHOLD)
            break;

        uint gIdx = gs_gaussianIDs[i];

        float2 projCenter = float2(projRead(gIdx, PROJ_CENTER_X),
                                   projRead(gIdx, PROJ_CENTER_Y));
        float  opScale    = projRead(gIdx, PROJ_OPACITY);
        float  ca         = projRead(gIdx, PROJ_CONIC_A);
        float  cb         = projRead(gIdx, PROJ_CONIC_B);
        float  cc         = projRead(gIdx, PROJ_CONIC_C);

        float2 dp = pixelCenter - projCenter;
        float power = -0.5 * (ca * dp.x * dp.x + 2.0 * cb * dp.x * dp.y + cc * dp.y * dp.y);

        if (power < -0.5 * CUTOFF_SIGMA * CUTOFF_SIGMA)
            continue;

        float G = exp(power);
        float tau = opScale * G;
        float alpha = 1.0 - exp(-tau);
        alpha = min(alpha, 0.99);

        if (alpha < 1.0 / 255.0)
            continue;

        float lightTransmit = gLightTransmittance[gIdx];

        // Optional: view-dependent depth darkening — deeper gaussians along the view ray
        // receive less light, simulating in-scattering attenuation through the medium
        if (useDepthDarkening != 0)
            lightTransmit *= (1.0 - accAlpha);

        float3 light = sunLight * lightTransmit + ambientColor;

        float w = (1.0 - accAlpha) * alpha;
        accColor += w * smokeAlbedo * light;
        accAlpha += w;
    }

    float3 finalColor = accColor + bg * (1.0 - accAlpha);
    finalColor = saturate(finalColor);
    gOutputTexture[pixel] = float4(finalColor, 1.0);
}

// ============================================================
// Post-process: screen-space Gaussian blur to soften splat edges
// ============================================================

[shader("compute")]
[numthreads(16, 16, 1)]
void blur_splats(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= screenWidth || pixel.y >= screenHeight)
        return;

    int r = int(blurRadius);
    float4 acc = float4(0.0);
    float weightSum = 0.0;
    float sigma = float(r) * 0.5;
    float invSigma2 = 1.0 / (2.0 * sigma * sigma);

    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            int2 sp = int2(pixel) + int2(dx, dy);
            if (sp.x < 0 || sp.x >= int(screenWidth) || sp.y < 0 || sp.y >= int(screenHeight))
                continue;

            float w = exp(-float(dx*dx + dy*dy) * invSigma2);
            acc += gOutputTexture[uint2(sp)] * w;
            weightSum += w;
        }
    }

    gBlurOutput[pixel] = acc / weightSum;
}
