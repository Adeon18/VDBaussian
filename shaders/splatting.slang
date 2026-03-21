// splatting.slang
// Volumetric Gaussian Splatting renderer
// Uses closed-form ray integral through 3D Gaussians for physically-based volumetric opacity.
//
// The line integral of a 3D Gaussian N(mu, Sigma) with amplitude w along ray r(t) = o + t*d is:
//   integral = w * sqrt(2*PI / (d^T P d)) * exp(-0.5 * q_perp)
// where P = Sigma^{-1} and q_perp is the perpendicular Mahalanobis distance.
// The exp(-0.5 * q_perp) term is exactly the 2D projected Gaussian evaluated at the pixel.

// ============================================================
// Constants
// ============================================================

static const uint PARAMS_PER_GAUSSIAN = 11;
static const uint SCREEN_TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_SCREEN_TILE = 512;
static const float CUTOFF_SIGMA = 3.0;
static const float MIN_DEPTH = 0.2;
static const float OPACITY_THRESHOLD = 0.999;
static const float PI = 3.14159265358979;

// ============================================================
// Data structures
// ============================================================

// Matches the raw 44-byte buffer layout used by the rest of the project
struct Gaussian
{
    float3 pos;
    float3 scale;
    float4 quat;
    float weight;
};

// Intermediate projected data — stored as flat floats to avoid
// any struct-padding surprises across GPU vendors.
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

// ============================================================
// Bindings
// ============================================================

[[vk::binding(0)]] StructuredBuffer<Gaussian> gGaussians;
[[vk::binding(1)]] StructuredBuffer<float> gCamera;
[[vk::binding(2)]] StructuredBuffer<float> gSettings;
[[vk::binding(3)]] RWStructuredBuffer<float> gProjected;       // flat float array
[[vk::binding(4)]] StructuredBuffer<uint> gSortedIndices;
[[vk::binding(5)]] RWStructuredBuffer<uint> gScreenTileContent;
[[vk::binding(6)]] RWStructuredBuffer<uint> gScreenTileCounts;
[[vk::binding(7)]] RWTexture2D<float4> gOutputTexture;

cbuffer SplatParams
{
    uint  gaussianCount;
    uint  screenWidth;
    uint  screenHeight;
    uint2 tileResolution;
    float densityScale;
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
    if (gid >= gaussianCount)
        return;

    Gaussian g = gGaussians[gid];

    // Mark invalid by default
    projWrite(gid, PROJ_DEPTH, -1.0);

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

    // Low-pass filter to prevent aliasing (matches 3DGS)
    cov_xx += 0.3;
    cov_yy += 0.3;

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

    // Lighting — simplified for splatting (no shadow marching)
    float3 smokeAlbedo  = float3(gSettings[4], gSettings[5], gSettings[6]);
    float  phaseG       = gSettings[8];
    float3 sunDir       = normalize(float3(gSettings[9], gSettings[10], gSettings[11]));
    float3 sunColorBase = float3(gSettings[12], gSettings[13], gSettings[14]);
    float  lightPen     = gSettings[7];
    float3 ambientColor = float3(gSettings[15], gSettings[16], gSettings[17]);

    float3 camFront = float3(gCamera[4], gCamera[5], gCamera[6]);
    float3 camRight = float3(gCamera[8], gCamera[9], gCamera[10]);
    float3 camUp    = float3(gCamera[12], gCamera[13], gCamera[14]);

    float2 screenUV = (float2(pixel) + 0.5) / float2(screenWidth, screenHeight) * 2.0 - 1.0;
    float3 rayDir = normalize(camFront + camRight * screenUV.x + camUp * screenUV.y);
    float phase = henyeyGreenstein(phaseG, dot(rayDir, sunDir));

    float3 light = sunColorBase * lightPen * phase + ambientColor;

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

        float w = (1.0 - accAlpha) * alpha;
        accColor += w * smokeAlbedo * light;
        accAlpha += w;
    }

    float3 finalColor = accColor + bg * (1.0 - accAlpha);
    finalColor = saturate(finalColor);
    gOutputTexture[pixel] = float4(finalColor, 1.0);
}
