#include <metal_stdlib>
using namespace metal;
struct SavedGaussian { float4 position, color, axisX, axisY, axisZ; };
struct SavedUniforms { float4x4 view, projection; float4 viewport; };
struct SavedVertex { float4 position [[position]]; float2 local; float4 color; };
vertex SavedVertex savedSplatVertex(uint vid [[vertex_id]], uint instance [[instance_id]],
    device const SavedGaussian* splats [[buffer(0)]], device const uint* order [[buffer(1)]],
    constant SavedUniforms& u [[buffer(2)]]) {
    SavedGaussian g = splats[order[instance]];
    float4 p = u.view * g.position;
    float4 clip = u.projection * p;
    SavedVertex out;
    out.color = g.color;
    float2 corners[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),float2(-1,1),float2(1,-1),float2(1,1)};
    out.local = corners[vid] * 3.0;
    if (p.z >= -0.03 || clip.w <= 0) { out.position = float4(2,2,2,1); out.color = 0; return out; }
    float2 center = clip.xy / clip.w;
    float2 axes[3];
    float4 worldAxes[3] = {g.axisX,g.axisY,g.axisZ};
    for (uint i=0; i<3; ++i) {
        float4 a = u.projection * u.view * worldAxes[i];
        axes[i] = (a.xy * clip.w - clip.xy * a.w) / (clip.w * clip.w);
        axes[i] *= u.viewport.xy * 0.5;
    }
    float xx = 0.3, yy = 0.3, xy = 0;
    for (uint i=0; i<3; ++i) { xx += axes[i].x*axes[i].x; yy += axes[i].y*axes[i].y; xy += axes[i].x*axes[i].y; }
    float middle = 0.5*(xx+yy);
    float delta = sqrt(max(0.0, 0.25*(xx-yy)*(xx-yy)+xy*xy));
    float2 major = abs(xy)>0.00001 ? normalize(float2(xy,middle+delta-xx)) : (xx>=yy ? float2(1,0) : float2(0,1));
    float2 minor = float2(-major.y,major.x);
    float2 offset = major * sqrt(max(0.3,middle+delta)) * out.local.x;
    offset += minor * sqrt(max(0.3,middle-delta)) * out.local.y;
    // Bound pathological near-camera splats without changing the exported result.
    offset = clamp(offset, -u.viewport.xy, u.viewport.xy);
    float2 ndc = center + offset * 2.0 / u.viewport.xy;
    out.position = float4(ndc * clip.w, clip.z, clip.w);
    out.color.a *= u.viewport.z;
    return out;
}
fragment float4 savedSplatFragment(SavedVertex in [[stage_in]]) {
    float a = in.color.a * exp(-0.5 * dot(in.local,in.local));
    if (a < 0.0039) discard_fragment();
    return float4(in.color.rgb*a,a);
}
