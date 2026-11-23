#include <metal_stdlib>
using namespace metal;

// Compiled into this package's default Metal library. GPU appearance and texture
// coordinates require an iPhone acceptance run; there is no Linux rendering substitute.
kernel void subjectHighlight(texture2d<float, access::read> source [[texture(0)]],
                             texture2d<float, access::write> target [[texture(1)]],
                             texture2d<float, access::sample> mask [[texture(2)]],
                             constant uint &enabled [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= target.get_width() || gid.y >= target.get_height()) return;
    float4 color = source.read(gid);
    if (enabled == 0) { target.write(color, gid); return; }
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid)+0.5)/float2(target.get_width(), target.get_height());
    float inside = mask.sample(s, uv).r;
    float edge = 0.0, glow = 0.0;
    // Width is measured in output screen pixels, independent of mask resolution.
    float2 texel = 1.0/float2(target.get_width(), target.get_height());
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            float m = mask.sample(s, uv+float2(x,y)*texel).r;
            float delta = abs(m-inside);
            if (abs(x) <= 1 && abs(y) <= 1) edge = max(edge, delta);
            glow = max(glow, delta * exp(-0.3*float(x*x+y*y)));
        }
    }
    color.rgb *= mix(0.28, 1.0, inside);
    color.rgb += float3(0.12, 0.85, 0.65)*(0.8*edge+0.45*glow);
    target.write(color, gid);
}
