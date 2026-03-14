[[vk::binding(1, 0)]] Texture3D<float4> debugVolume;  // RGBA debug data
[[vk::binding(2, 0)]] StructuredBuffer<float> camera; 
[[vk::binding(3, 0)]] SamplerState linearSampler;

struct VertexOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const float STEP_SIZE = 0.01;
static const int MAX_STEPS = 256;

float2 hitBox(float3 orig, float3 dir) {
    const float3 box_min = float3(-1.0);
    const float3 box_max = float3( 1.0);
    float3 inv_dir = 1.0 / dir;
    float3 t0 = (box_min - orig) * inv_dir;
    float3 t1 = (box_max - orig) * inv_dir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float t_near = max(max(tmin.x, tmin.y), tmin.z);
    float t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return float2(t_near, t_far);
}

[shader("vertex")]
VertexOut vertex_main(uint id : SV_VertexID)
{
    VertexOut output;
    output.uv = float2((id << 1) & 2, id & 2);
    output.pos = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOut pin) : SV_Target
{
    // Camera Setup
    float3 ro    = float3(camera[0], camera[1], camera[2]);
    float3 front = float3(camera[4], camera[5], camera[6]);
    float3 right = float3(camera[8], camera[9], camera[10]);
    float3 up    = float3(camera[12], camera[13], camera[14]);

    float2 screenUV = pin.uv * 2.0 - 1.0; 
    float3 rd = normalize(front + (right * screenUV.x) + (up * screenUV.y));
    
    // Background
    float3 bg = lerp(float3(0.1, 0.1, 0.15), float3(0.2, 0.2, 0.3), pin.uv.y);
    float4 color = float4(0.0);

    // Box Test
    float2 bounds = hitBox(ro, rd);
    if (bounds.x > bounds.y || bounds.y < 0.0) return float4(bg, 1.0);

    float t = max(0.0, bounds.x);
    float t_end = bounds.y;
    
    // Raymarch with compositing
    for(int i = 0; i < MAX_STEPS; i++) {
        if (t > t_end) break;
        if (color.a > 0.99) break;

        float3 p = ro + rd * t;
        float3 uvw = p * 0.5 + 0.5;
        uvw.y = 1.0 - uvw.y;  // VDB Y-up fix
        float4 sample = debugVolume.SampleLevel(linearSampler, uvw, 0);
        
        // Front-to-back compositing
        if (sample.a > 0.01) {
            float alpha = sample.a * STEP_SIZE * 50.0; // Scale for visibility
            color.rgb += (1.0 - color.a) * sample.rgb * alpha;
            color.a += (1.0 - color.a) * alpha;
        }
        
        t += STEP_SIZE;
    }
    
    color.rgb = color.rgb + bg * (1.0 - color.a);
    return float4(color.rgb, 1.0);
}