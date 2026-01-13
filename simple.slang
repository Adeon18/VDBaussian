// hybrid.slang

// --- INPUTS ---
[[vk::binding(1, 0)]] Texture3D<float> inVolume;
// We use a simple float buffer for camera data to avoid struct/dict errors
// [0..2]=Pos, [4..6]=Front, [8..10]=Right, [12..14]=Up
[[vk::binding(2, 0)]] StructuredBuffer<float> camera; 
SamplerState linearSampler; 

struct VertexOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

[shader("vertex")]
VertexOut vertex_main(uint id : SV_VertexID)
{
    VertexOut output;
    output.uv = float2((id << 1) & 2, id & 2);
    output.pos = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Helper: SDF for a Box (to see the volume bounds)
float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

[shader("fragment")]
float4 fragment_main(VertexOut pin) : SV_Target
{
    // 1. READ CAMERA FROM BUFFER
    float3 ro    = float3(camera[0], camera[1], camera[2]);
    float3 front = float3(camera[4], camera[5], camera[6]);
    float3 right = float3(camera[8], camera[9], camera[10]);
    float3 up    = float3(camera[12], camera[13], camera[14]);

    // 2. GENERATE RAY
    // Map UV [0..2] to [-1..1] (aspect ratio correction handled in Python)
    float2 screenUV = pin.uv * 2.0 - 1.0; 
    
    // Ray Direction = Front + (Right * x) + (Up * y)
    float3 rd = normalize(front + (right * screenUV.x) + (up * screenUV.y));
    
    float4 color = float4(0.0);
    float t = 0.0;
    
    // 3. RAYMARCH
    for(int i=0; i<128; i++) {
        float3 p = ro + rd * t;
        
        // Debug Box (White Cage)
        float boxDist = sdBox(p, float3(1.0));
        if (abs(boxDist) < 0.01) color += float4(0.1, 0.1, 0.1, 0.05);

        // Volume Sampling
        if(all(abs(p) < 1.0)) {
            // Remap [-1, 1] -> [0, 1]
            float3 uvw = p * 0.5 + 0.5;
            float d = inVolume.SampleLevel(linearSampler, uvw, 0);
            
            if(d > 0.01) {
                // Density coloring
                float3 heat = lerp(float3(0,1,0), float3(1,0,0), d * 3.0);
                float alpha = d * 0.4; 
                color.rgb += (1.0 - color.a) * alpha * heat;
                color.a += (1.0 - color.a) * alpha;
            }
        }
        
        if(color.a > 0.98) break;
        t += 0.04;
    }
    
    // Background gradient based on ray direction Y
    float3 bg = lerp(float3(0.05, 0.05, 0.1), float3(0.0, 0.0, 0.0), rd.y * 0.5 + 0.5);
    color.rgb = color.rgb + bg * (1.0 - color.a);
    
    return float4(color.rgb, 1.0);
}