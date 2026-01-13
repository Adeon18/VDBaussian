// hybrid.slang

[[vk::binding(1, 0)]] Texture3D<float> inVolume;
[[vk::binding(2, 0)]] StructuredBuffer<float> camera; 
[[vk::binding(3, 0)]] SamplerState linearSampler;
[[vk::binding(4, 0)]] StructuredBuffer<float> settings; 

struct VertexOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

[shader("vertex")]
VertexOut vertex_main(uint id : SV_VertexID)
{
    VertexOut output;
    output.uv = float2((id << 1) & 2, id & 2);
    output.pos = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// --- NEW: RAY-BOX INTERSECTION ---
// Returns float2(t_near, t_far). 
// If t_near > t_far, the ray missed the box.
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

[shader("fragment")]
float4 fragment_main(VertexOut pin) : SV_Target
{
    // 1. Camera Setup
    float3 ro    = float3(camera[0], camera[1], camera[2]);
    float3 front = float3(camera[4], camera[5], camera[6]);
    float3 right = float3(camera[8], camera[9], camera[10]);
    float3 up    = float3(camera[12], camera[13], camera[14]);

    float2 screenUV = pin.uv * 2.0 - 1.0; 
    float3 rd = normalize(front + (right * screenUV.x) + (up * screenUV.y));
    
    // 2. Settings
    float step_size = settings[0];
    if (step_size < 0.001) step_size = 0.015;
    
    float density_scale = settings[1];
    float density_curve = settings[2];
    int max_steps = int(settings[3]);
    if (max_steps < 10) max_steps = 256;
    float3 smoke_albedo = float3(settings[4], settings[5], settings[6]);

    // 3. BACKGROUND
    float3 top = float3(0.5, 0.6, 0.7);
    float3 bot = float3(0.5, 0.6, 0.8);
    float3 bg = lerp(bot, top, pin.uv.y);
    float4 color = float4(0.0);

    // 4. INTERSECTION TEST
    float2 bounds = hitBox(ro, rd);
    
    // If ray misses box (near > far) or box is behind camera (far < 0), draw background
    if (bounds.x > bounds.y || bounds.y < 0.0) {
        return float4(bg, 1.0);
    }

    // Clamp start to 0.0 (in case camera is INSIDE the volume)
    float t = max(0.0, bounds.x);
    float t_end = bounds.y;

    // Jitter start position to remove "wood grain" banding artifacts
    // Simple hash based on pixel position
    float noise = frac(sin(dot(pin.uv, float2(12.9898, 78.233))) * 43758.5453);
    t += noise * step_size;

    // 5. MARCH LOOP
    // Loop until we hit the max steps OR we exit the box (t > t_end)
    for(int i=0; i<2000; i++) {
        if (i >= max_steps || t > t_end) break;

        float3 p = ro + rd * t;
        float3 uvw = p * 0.5 + 0.5;
        
        // No need for 'if(all(abs(p) < 1.0))' because t_near/t_far guarantees we are inside!
        float d = inVolume.SampleLevel(linearSampler, uvw, 0);
        
        if(d > 0.001) {
            float shape = pow(d, density_curve); 
            float alpha = shape * density_scale * step_size;
            
            // Accumulate
            color.rgb += (1.0 - color.a) * alpha * smoke_albedo;
            color.a += (1.0 - color.a) * alpha;
        }
        
        if(color.a > 0.99) break;
        t += step_size;
    }
    
    // Composite over background
    color.rgb = color.rgb + bg * (1.0 - color.a);
    return float4(color.rgb, 1.0);
}