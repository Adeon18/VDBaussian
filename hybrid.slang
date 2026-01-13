// hybrid.slang

// --- INPUTS ---
[[vk::binding(1, 0)]] Texture3D<float> inVolume;
[[vk::binding(2, 0)]] StructuredBuffer<float> camera; 
[[vk::binding(3, 0)]] SamplerState linearSampler; // <--- The Linear Sampler

struct VertexOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

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
    // 1. CAMERA
    float3 ro    = float3(camera[0], camera[1], camera[2]);
    float3 front = float3(camera[4], camera[5], camera[6]);
    float3 right = float3(camera[8], camera[9], camera[10]);
    float3 up    = float3(camera[12], camera[13], camera[14]);

    float2 screenUV = pin.uv * 2.0 - 1.0; 
    float3 rd = normalize(front + (right * screenUV.x) + (up * screenUV.y));
    
    // 2. RAYMARCH SETTINGS
    float4 color = float4(0.0);
    float t = 0.0;
    
    // TWEAK THESE TO CHANGE LOOK:
    float step_size = 0.02;      // Smaller = smoother, slower
    float density_scale = 25.0;  // How "thick" the cloud is (High value = solid)
    float density_curve = 0.5;   // < 1.0 boosts faint wisps. > 1.0 hides them.
    float3 smoke_albedo = float3(0.9, 0.95, 1.0); // Blue-white smoke

    // 3. MARCH
    for(int i=0; i<512; i++) {
        float3 p = ro + rd * t;
        
        if(all(abs(p) < 1.0)) {
            float3 uvw = p * 0.5 + 0.5;
            
            // Linear Sampling (Smooths the blockiness)
            float d = inVolume.SampleLevel(linearSampler, uvw, 0);
            
            if(d > 0.001) {
                // VISIBILITY BOOST:
                // Raise density to a power < 1.0 (e.g. 0.5) to make 0.1 become 0.3
                float shape = pow(d, density_curve); 
                
                // Beer's Law approximation
                float alpha = shape * density_scale * step_size;
                
                // Blend
                color.rgb += (1.0 - color.a) * alpha * smoke_albedo;
                color.a += (1.0 - color.a) * alpha;
            }
        }
        
        if(color.a > 0.99) break;
        t += step_size;
        if(t > 7.0) break;
    }
    
    // Background (Dark)
    float3 bg = float3(0.5, 0.6, 0.6);
    color.rgb = color.rgb + bg * (1.0 - color.a);
    
    return float4(color.rgb, 1.0);
}