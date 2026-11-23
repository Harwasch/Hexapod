#include <metal_stdlib>
using namespace metal;

// Exact 64-byte cells; see SurfelReference.swift. No model readback or per-frame buffers.
struct Surfel { int4 key; float4 positionRadius; float4 color; uint4 state; };
struct Uniforms {
  float4x4 world, view;
  float4 intrinsics, displayX, displayY, inverseX, inverseY, dimensions;
  uint4 settings, flags;
};
constant uint capacity = 262144;
uint bucket(int4 key) {
  return ((uint(key.x) * 73856093u) ^ (uint(key.y) * 19349663u)
    ^ (uint(key.z) * 83492791u) ^ uint(key.w)) % (capacity / 4);
}
int4 voxel(float3 p, float d) {
  int level = d < 3 ? 0 : 1;
  return int4(int3(floor(p / (level == 0 ? 0.02f : 0.04f))), level);
}
float radius(float depth, float focal, float stride) {
  return max(0.85f * (depth < 3 ? 0.02f : 0.04f), min(0.12f, 0.85f * depth * stride / focal));
}
float3 worldPoint(float2 uv, float d, constant Uniforms &u) {
  float2 xy = (uv - u.intrinsics.zw) * d / u.intrinsics.xy;
  return (u.world * float4(xy.x, -xy.y, -d, 1)).xyz;
}
bool validDepth(float d, float confidence) { return isfinite(d) && d >= 0.1f && d <= 8 && confidence > 1.5f / 255; }
float3 rgb(float y, float2 cbcr, constant Uniforms &u) {
  float yy = u.flags.y ? (y * 255 - 16) / 219 : y;
  float2 c = (cbcr * 255 - 128) / (u.flags.y ? 224.0f : 255.0f);
  return clamp(float3(yy + (u.flags.z ? 1.5748f : 1.402f) * c.y,
    yy - (u.flags.z ? 0.187324f : 0.344136f) * c.x - (u.flags.z ? 0.468124f : 0.714136f) * c.y,
    yy + (u.flags.z ? 1.8556f : 1.772f) * c.x), 0.0f, 1.0f);
}
// One winner per bucket per keyframe. Claims reset with a blit before the next fusion pass.
// No lock release within a dispatch: no cross-thread publication or spin waiting.
// GPU memory is only accessed by subsequent encoders after fusion completes.
kernel void fuseSurfels(device Surfel *cells [[buffer(0)]], device atomic_uint *locks [[buffer(1)]],
  device atomic_uint *counts [[buffer(2)]], constant Uniforms &u [[buffer(3)]],
  texture2d<float, access::read> depth [[texture(0)]], texture2d<float, access::read> confidence [[texture(1)]],
  texture2d<float> yPlane [[texture(2)]], texture2d<float> cbcrPlane [[texture(3)]], uint id [[thread_position_in_grid]]) {
  if (id >= u.settings.x) return;
  uint stride = u.settings.z, columns = (depth.get_width() + stride - 1) / stride;
  uint phase = u.settings.y % (stride * stride);
  uint2 pixel = uint2(id % columns, id / columns) * stride + uint2(phase % stride, phase / stride);
  if (pixel.x >= depth.get_width() || pixel.y >= depth.get_height()) return;
  float d = depth.read(pixel).r;
  if (!validDepth(d, confidence.read(pixel).r)) return;
  atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
  float2 uv = float2(pixel) / float2(depth.get_width(), depth.get_height());
  float3 p = worldPoint(uv, d, u);
  if (!all(isfinite(p)) || any(abs(p) >= 1000000)) return;
  int4 key = voxel(p, d);
  uint b = bucket(key), expected = 0;
  if (!atomic_compare_exchange_weak_explicit(locks + b, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) return;
  uint start = b * 4, victim = start;
  bool found = false;
  for (uint i = start; i < start + 4; ++i) {
    Surfel cell = cells[i];
    if (cell.state.x == 0) { victim = i; break; }
    if (all(cell.key == key)) { victim = i; found = true; break; }
    Surfel old = cells[victim];
    if (cell.state.y < old.state.y || (cell.state.y == old.state.y && cell.state.x < old.state.x)) victim = i;
  }
  Surfel cell = cells[victim];
  if (found && cell.state.x >= 2) atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
  constexpr sampler sample(filter::linear, address::clamp_to_edge);
  float3 color = rgb(yPlane.sample(sample, uv).r, cbcrPlane.sample(sample, uv).rg, u);
  float r = radius(d, u.intrinsics.x * u.dimensions.z, float(stride));
  if (found) {
    float weight = 1.0f / float(min(32u, cell.state.x + 1));
    cell.positionRadius.xyz = mix(cell.positionRadius.xyz, p, weight);
    cell.positionRadius.w = max(cell.positionRadius.w, r);
    cell.color.xyz = mix(cell.color.xyz, color, weight);
    cell.state.x = min(32u, cell.state.x + 1);
  } else {
    cell.key = key; cell.positionRadius = float4(p, r); cell.color = float4(color, 1); cell.state.x = 1;
  }
  cell.state.y = u.settings.y;
  cells[victim] = cell;

}
struct VertexOut { float4 position [[position]]; float size [[point_size]]; float3 color; float distance; };
vertex VertexOut surfelVertex(device const Surfel *cells [[buffer(0)]], constant Uniforms &u [[buffer(1)]], uint id [[vertex_id]]) {
  Surfel cell = cells[id * u.flags.x];
  float3 p = (u.view * float4(cell.positionRadius.xyz, 1)).xyz;
  float d = -p.z;
  VertexOut out;
  out.position = float4(2, 2, 2, 1); out.size = 1; out.color = cell.color.xyz; out.distance = d;
  if (cell.state.x == 0 || d < 0.05f || d > 100) return out;
  float2 uv = float2(p.x, -p.y) * u.intrinsics.xy / d + u.intrinsics.zw;
  float3 q = float3(uv, 1);
  float2 screen = float2(dot(u.displayX.xyz, q), dot(u.displayY.xyz, q));
  out.position = float4(screen.x * 2 - 1, 1 - screen.y * 2, (d - 0.05f) / 99.95f, 1);
  float2 dx = float2(u.displayX.x, u.displayY.x) * u.dimensions.xy * u.intrinsics.x;
  float2 dy = float2(u.displayX.y, u.displayY.y) * u.dimensions.xy * u.intrinsics.y;
  out.size = clamp(2 * cell.positionRadius.w * max(length(dx), length(dy)) / d, 1.0f, 64.0f);
  return out;
}
fragment float4 surfelFragment(VertexOut in [[stage_in]], float2 point [[point_coord]],
  constant Uniforms &u [[buffer(1)]], texture2d<float> depth [[texture(0)]], texture2d<float> confidence [[texture(1)]]) {
  float r2 = dot(point * 2 - 1, point * 2 - 1);
  if (r2 >= 1) discard_fragment();
  float3 screen = float3(in.position.xy / u.dimensions.xy, 1);
  float2 uv = float2(dot(u.inverseX.xyz, screen), dot(u.inverseY.xyz, screen));
  constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
  float d = depth.sample(nearest, uv).r;
  if (validDepth(d, confidence.sample(nearest, uv).r) && in.distance > d + max(0.03f, d * 0.02f)) discard_fragment();
  float alpha = 0.94f * (1 - smoothstep(0.55f, 1.0f, r2));
  if (alpha < 0.05f) discard_fragment();
  // Camera colour is gamma encoded; RealityKit's source/target are linear.
  float3 linear = select(in.color / 12.92f, pow((in.color + 0.055f) / 1.055f, float3(2.4f)), in.color > 0.04045f);
  return float4(linear * alpha, alpha);
}
// Coverage runs at the adaptive INTERNAL resolution, never a full-resolution hash walk.
kernel void coverageScanPreview(device const Surfel *cells [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
  texture2d<float, access::write> target [[texture(0)]], texture2d<float> depth [[texture(1)]],
  texture2d<float> confidence [[texture(2)]], uint2 pixel [[thread_position_in_grid]]) {
  if (pixel.x >= target.get_width() || pixel.y >= target.get_height()) return;
  float2 screen = (float2(pixel) + 0.5f) / float2(target.get_width(), target.get_height());
  float3 q = float3(screen, 1);
  float2 uv = float2(dot(u.inverseX.xyz, q), dot(u.inverseY.xyz, q));
  constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
  float d = depth.sample(nearest, uv).r;
  float4 tint = 0;
  if (all(uv >= 0) && all(uv < 1) && validDepth(d, confidence.sample(nearest, uv).r)) {
    // Match the integer depth pixel centres used by fusion.
    float2 depthUV = floor(uv * u.dimensions.zw) / u.dimensions.zw;
    int4 key = voxel(worldPoint(depthUV, d, u), d);
    uint start = bucket(key) * 4;
    bool covered = false;
    for (uint i = start; i < start + 4; ++i) {
      if (cells[i].state.x >= 2 && all(cells[i].key == key)) covered = true;
    }
    if (!covered) tint = float4(float3(0.03f, 0.45f, 0.40f) * 0.12f, 0.12f);
  }
  target.write(tint, pixel);
}
kernel void compositeScanPreview(texture2d<float, access::read> source [[texture(0)]],
  texture2d<float, access::write> target [[texture(1)]], texture2d<float> surfels [[texture(2)]],
  uint2 pixel [[thread_position_in_grid]]) {
  if (pixel.x >= target.get_width() || pixel.y >= target.get_height()) return;
  float2 screen = (float2(pixel) + 0.5f) / float2(target.get_width(), target.get_height());
  constexpr sampler linear(filter::linear, address::clamp_to_edge);
  float4 camera = source.read(pixel), preview = surfels.sample(linear, screen);
  target.write(float4(preview.rgb + camera.rgb * (1 - preview.a), 1), pixel);
}
