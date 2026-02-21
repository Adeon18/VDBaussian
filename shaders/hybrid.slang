// hybrid.slang

[[vk::binding(1, 0)]] Texture3D<float> inVolume;
[[vk::binding(2, 0)]] StructuredBuffer<float> camera; 
[[vk::binding(3, 0)]] SamplerState linearSampler;
[[vk::binding(4, 0)]] StructuredBuffer<float> settings; 

struct VertexOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// HARDCODED OPTIMIZATION CONSTANTS
static const float SHADOW_BIAS = 0.01; 
static const int MAX_ITERATIONS = 2000;
static const float DENSITY_THRESHOLD = 0.001;
static const float OPACITY_THRESHOLD = 0.99;

// HELPER FUNCTIONS
float henyeyGreenstein(float g, float costh) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * costh, 1.5));
}

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

// Shadow Marching
float getLightAttenuation(float3 p, float3 lightDir, float step_size, float density_scale, float light_penetration, int shadow_steps, float shadow_step_mult) {
    float t = SHADOW_BIAS; 
    float accumulated_density = 0.0;
    
    // We cast shadow_steps to int loop
    for(int j=0; j<shadow_steps; j++) {
        float3 lightP = p + lightDir * t;
        
        if (abs(lightP.x) > 1.0 || abs(lightP.y) > 1.0 || abs(lightP.z) > 1.0) break;

        float3 uvw = lightP * 0.5 + 0.5;
        float d = inVolume.SampleLevel(linearSampler, uvw, 0);
        
        accumulated_density += d;
        t += step_size * shadow_step_mult;
    }
    
    float effective_density = accumulated_density * density_scale * light_penetration;
    return exp(-effective_density * step_size * shadow_step_mult);
}

// MAIN SHADER
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
    
    // Load Settings from Buffer
    // Basic
    float step_size       = (settings[0] < 0.001) ? 0.015 : settings[0];
    float density_scale   = settings[1];
    float density_curve   = settings[2];
    int   max_steps       = int(settings[3]);
    float3 smoke_albedo   = float3(settings[4], settings[5], settings[6]);
    
    // Lighting Params (Indices [7-19])
    float  light_penetration = settings[7];
    float  phase_g           = settings[8];
    float3 sun_dir           = normalize(float3(settings[9], settings[10], settings[11]));
    float3 sun_color         = float3(settings[12], settings[13], settings[14]);
    float3 ambient_color     = float3(settings[15], settings[16], settings[17]);
    int    shadow_steps      = int(settings[18]);
    float  shadow_step_mult  = settings[19];

    // Background
    float3 top = float3(0.5, 0.6, 0.7);
    float3 bot = float3(0.5, 0.6, 0.8);
    float3 bg = lerp(bot, top, pin.uv.y);
    float4 color = float4(0.0);

    // Box Test
    float2 bounds = hitBox(ro, rd);
    if (bounds.x > bounds.y || bounds.y < 0.0) return float4(bg, 1.0);

    float t = max(0.0, bounds.x);
    float t_end = bounds.y;
    
    float noise = frac(sin(dot(pin.uv, float2(12.9898, 78.233))) * 43758.5453);
    t += noise * step_size;

    // Phase Function
    float sun_cos_angle = dot(rd, sun_dir);
    float phase_val = henyeyGreenstein(phase_g, sun_cos_angle);

    // Raymarch Loop
    for(int i=0; i<max_steps; i++) {
        if (t > t_end || i >= MAX_ITERATIONS) break;
        if (color.a > OPACITY_THRESHOLD) break;

        float3 p = ro + rd * t;
        float3 uvw = p * 0.5 + 0.5;
        
        float d = inVolume.SampleLevel(linearSampler, uvw, 0);
        
        if(d > DENSITY_THRESHOLD) {
            float shape = pow(d, density_curve); 
            float segment_alpha = shape * density_scale * step_size;
            
            // Lighting calculation using dynamic settings
            float transmittance = getLightAttenuation(p, sun_dir, step_size, density_scale, light_penetration, shadow_steps, shadow_step_mult);
            
            float3 light = sun_color * transmittance * phase_val + ambient_color;
            float3 finalRGB = smoke_albedo * light * segment_alpha;
            
            color.rgb += (1.0 - color.a) * finalRGB;
            color.a   += (1.0 - color.a) * segment_alpha;
        }
        
        t += step_size;
    }
    
    color.rgb = color.rgb + bg * (1.0 - color.a);
    return float4(color.rgb, 1.0);
}