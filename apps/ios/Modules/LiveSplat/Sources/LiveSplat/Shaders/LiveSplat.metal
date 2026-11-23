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
struct OrbitUniforms { float4 controls, rect, phoneIntrinsics; };
struct SceneBounds { float4 sumCount, lower, upper; };
struct SortEntry { float depth; uint index; };
float4x4 previewView(constant Uniforms &u, device const SceneBounds *scene, constant OrbitUniforms &orbit) {
  if (!(u.flags.w & 1u)) return u.view;
  float4 centerRadius = scene[1024].sumCount;
  float3 center = centerRadius.xyz;
  float radius = max(0.5f, centerRadius.w);
  float angle = orbit.controls.x, pitch = orbit.controls.y;
  float3 eye = center + radius * 2.5f * float3(cos(pitch) * sin(angle), sin(pitch), cos(pitch) * cos(angle));
  float3 z = normalize(eye - center), x = normalize(cross(float3(0, 1, 0), z)), y = cross(z, x);
  return float4x4(float4(x.x, y.x, z.x, 0), float4(x.y, y.y, z.y, 0),
    float4(x.z, y.z, z.z, 0), float4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1));
}
float4 previewPosition(float3 p, constant Uniforms &u) {
  float d = -p.z;
  float2 uv = float2(p.x, -p.y) * u.intrinsics.xy / max(d, 0.05f) + u.intrinsics.zw;
  float3 q = float3(uv, 1);
  float2 screen = float2(dot(u.displayX.xyz, q), dot(u.displayY.xyz, q));
  return float4(screen.x * 2 - 1, 1 - screen.y * 2, (d - 0.05f) / 99.95f, 1);
}
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
  device float4 *gaussians [[buffer(4)]],
  texture2d<float, access::read> depth [[texture(0)]], texture2d<float, access::read> confidence [[texture(1)]],
  texture2d<float> yPlane [[texture(2)]], texture2d<float> cbcrPlane [[texture(3)]], uint id [[thread_position_in_grid]]) {
  if (id >= u.settings.x) return;
  uint stride = u.settings.z, columns = (depth.get_width() + stride - 1) / stride;
  uint phase = u.settings.y % (stride * stride);
  uint2 pixel = uint2(id % columns, id / columns) * stride + uint2(phase % stride, phase / stride);
  if (pixel.x >= depth.get_width() || pixel.y >= depth.get_height()) return;
  float d = depth.read(pixel).r;
  float certainty = confidence.read(pixel).r * 255.0f;
  // Discs preserve the original high-confidence admission. Splats also admit
  // medium-confidence depth, with lower opacity; coverage still needs high confidence.
  if (!isfinite(d) || d < 0.1f || d > 8 || certainty < ((u.flags.w & 2u) ? 0.5f : 1.5f)) return;
  if (certainty > 1.5f) atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
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
  if (found && cell.state.z >= 2 && certainty > 1.5f) atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
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
  cell.state.z = certainty > 1.5f ? (found ? min(32u, cell.state.z + 1) : 1u) : 0u;
  cells[victim] = cell;
  if (u.flags.w & 2u) {
    // Estimate the local tangent plane from adjacent valid depth samples. At image
    // edges/depth discontinuities use the viewing direction rather than bridging surfaces.
    float3 normal = normalize(u.world[3].xyz - p);
    uint2 px = uint2(min(pixel.x + 1, depth.get_width() - 1), pixel.y);
    uint2 py = uint2(pixel.x, min(pixel.y + 1, depth.get_height() - 1));
    float dx = depth.read(px).r, dy = depth.read(py).r;
    if (pixel.x != px.x && pixel.y != py.y && validDepth(dx, confidence.read(px).r)
      && validDepth(dy, confidence.read(py).r) && abs(dx - d) < d * 0.05f && abs(dy - d) < d * 0.05f) {
      float3 tangentX = worldPoint(float2(px) / u.dimensions.zw, dx, u) - p;
      float3 tangentY = worldPoint(float2(py) / u.dimensions.zw, dy, u) - p;
      float3 n = cross(tangentX, tangentY);
      if (dot(n, n) > 1e-12f) normal = normalize(n);
    }
    gaussians[victim] = float4(normal, certainty > 1.5f ? 0.85f : 0.45f);
  }

}
struct VertexOut { float4 position [[position]]; float size [[point_size]]; float3 color; float distance; };
vertex VertexOut surfelVertex(device const Surfel *cells [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
  device const SceneBounds *scene [[buffer(4)]], constant OrbitUniforms &orbit [[buffer(5)]], uint id [[vertex_id]]) {
  Surfel cell = cells[id * u.flags.x];
  float3 p = (previewView(u, scene, orbit) * float4(cell.positionRadius.xyz, 1)).xyz;
  float d = -p.z;
  VertexOut out;
  out.position = float4(2, 2, 2, 1); out.size = 1; out.color = cell.color.xyz; out.distance = d;
  if ((u.flags.w & 1u) && cell.state.z < 2) out.color = mix(out.color, float3(0.05f, 0.9f, 0.75f), 0.65f);
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
  if (!(u.flags.w & 1u) && validDepth(d, confidence.sample(nearest, uv).r) && in.distance > d + max(0.03f, d * 0.02f)) discard_fragment();
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
      if (cells[i].state.z >= 2 && all(cells[i].key == key)) covered = true;
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

// Fixed work: 1024 partials of 256 cells, then a single small reduction. Geometry
// never returns to the CPU. Bounds include all fused cells, independent of draw thinning.
kernel void reduceScanScene(device const Surfel *cells [[buffer(0)]], device SceneBounds *scene [[buffer(1)]],
  uint id [[thread_position_in_grid]]) {
  if (id >= 1024) return;
  float4 sum = 0; float3 lo = float3(INFINITY), hi = float3(-INFINITY);
  for (uint i = id * 256; i < (id + 1) * 256; ++i) {
    if (!cells[i].state.x) continue;
    float3 p = cells[i].positionRadius.xyz;
    sum += float4(p, 1); lo = min(lo, p); hi = max(hi, p);
  }
  scene[id].sumCount = sum; scene[id].lower = float4(lo, 0); scene[id].upper = float4(hi, 0);
}
kernel void finishScanScene(device SceneBounds *scene [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  if (id) return;
  float4 sum = 0; float3 lo = float3(INFINITY), hi = float3(-INFINITY);
  for (uint i = 0; i < 1024; ++i) {
    sum += scene[i].sumCount;
    lo = min(lo, scene[i].lower.xyz); hi = max(hi, scene[i].upper.xyz);
  }
  scene[1024].sumCount = sum.w > 0 ? float4(sum.xyz / sum.w, max(0.5f, length(hi - lo) * 0.5f)) : float4(0, 0, 0, 1);
}
kernel void prepareGaussianSort(device const Surfel *cells [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
  device SortEntry *order [[buffer(3)]], device const SceneBounds *scene [[buffer(4)]],
  constant OrbitUniforms &orbit [[buffer(5)]], constant uint &count [[buffer(6)]], uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  uint index = id * (capacity / count);
  float d = -(previewView(u, scene, orbit) * float4(cells[index].positionRadius.xyz, 1)).z;
  order[id] = {cells[index].state.x && isfinite(d) && d > 0.05f && d <= 100 ? d : -INFINITY, index};
}
bool precedes(SortEntry a, SortEntry b) { return a.depth > b.depth || (a.depth == b.depth && a.index < b.index); }
kernel void bitonicGaussianSort(device SortEntry *order [[buffer(0)]], constant uint4 &stage [[buffer(1)]],
  uint id [[thread_position_in_grid]]) {
  uint other = id ^ stage.y;
  if (id >= stage.z || other <= id) return;
  SortEntry a = order[id], b = order[other];
  if (precedes(b, a) == ((id & stage.x) == 0)) { order[id] = b; order[other] = a; }
}
struct GaussianOut {
  float4 position [[position]];
  float2 offset;
  float3 conic;
  float3 color;
  float opacity, distance;
};
vertex GaussianOut gaussianVertex(device const Surfel *cells [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
  device const float4 *gaussians [[buffer(2)]], device const SortEntry *order [[buffer(3)]],
  device const SceneBounds *scene [[buffer(4)]], constant OrbitUniforms &orbit [[buffer(5)]],
  uint vertexID [[vertex_id]], uint instance [[instance_id]]) {
  uint index = order[instance].index;
  Surfel cell = cells[index]; float4 attribute = gaussians[index];
  float4x4 view = previewView(u, scene, orbit);
  float3 p = (view * float4(cell.positionRadius.xyz, 1)).xyz;
  float d = -p.z;
  GaussianOut out;
  out.position = float4(2, 2, 2, 1); out.offset = 0; out.conic = float3(1, 0, 1);
  out.color = cell.color.xyz; out.opacity = attribute.w; out.distance = d;
  if (!cell.state.x || !isfinite(order[instance].depth) || d <= 0.05f || d > 100) return out;
  if ((u.flags.w & 1u) && cell.state.z < 2) out.color = mix(out.color, float3(0.05f, 0.9f, 0.75f), 0.65f);
  float3 n = (view * float4(attribute.xyz, 0)).xyz;
  float sigma = cell.positionRadius.w * 0.5f, thin = max(0.001f, sigma * 0.2f);
  // Sigma = s_t^2 I + (s_n^2-s_t^2) n n^T. Project with the
  // pinhole Jacobian AND display crop/rotation, then add EWA's 0.3 pixel filter.
  float3 jx = float3(u.intrinsics.x / d, 0, u.intrinsics.x * p.x / (d * d));
  float3 jy = float3(0, -u.intrinsics.y / d, -u.intrinsics.y * p.y / (d * d));
  float3 ax = (u.displayX.x * jx + u.displayX.y * jy) * u.dimensions.x;
  float3 ay = (u.displayY.x * jx + u.displayY.y * jy) * u.dimensions.y;
  float ss = sigma * sigma, delta = thin * thin - ss;
  float xx = ss * dot(ax, ax) + delta * dot(ax, n) * dot(ax, n) + 0.3f;
  float xy = ss * dot(ax, ay) + delta * dot(ax, n) * dot(ay, n);
  float yy = ss * dot(ay, ay) + delta * dot(ay, n) * dot(ay, n) + 0.3f;
  float det = max(xx * yy - xy * xy, 1e-8f);
  out.conic = float3(yy, -xy, xx) / det;
  float2 extent = min(float2(96), 3 * sqrt(float2(xx, yy)));
  const float2 corners[6] = {float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1)};
  out.offset = corners[vertexID] * extent;
  out.position = previewPosition(p, u);
  out.position.xy += out.offset / u.dimensions.xy * float2(2, -2);
  return out;
}
fragment float4 gaussianFragment(GaussianOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
  texture2d<float> depth [[texture(0)]], texture2d<float> confidence [[texture(1)]]) {
  float power = in.conic.x * in.offset.x * in.offset.x + 2 * in.conic.y * in.offset.x * in.offset.y + in.conic.z * in.offset.y * in.offset.y;
  if (power > 9) discard_fragment();
  if (!(u.flags.w & 1u)) {
    float3 q = float3(in.position.xy / u.dimensions.xy, 1);
    float2 uv = float2(dot(u.inverseX.xyz, q), dot(u.inverseY.xyz, q));
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    float d = depth.sample(nearest, uv).r;
    if (validDepth(d, confidence.sample(nearest, uv).r) && in.distance > d + max(0.03f, d * 0.02f)) discard_fragment();
  }
  float alpha = min(0.95f, in.opacity * exp(-0.5f * power));
  if (alpha < 1.0f / 255) discard_fragment();
  float3 linear = select(in.color / 12.92f, pow((in.color + 0.055f) / 1.055f, float3(2.4f)), in.color > 0.04045f);
  return float4(linear * alpha, alpha);
}
struct FrustumOut { float4 position [[position]]; };
vertex FrustumOut phoneFrustumVertex(constant Uniforms &u [[buffer(1)]], device const SceneBounds *scene [[buffer(4)]],
  constant OrbitUniforms &orbit [[buffer(5)]], uint id [[vertex_id]]) {
  // Four rays and four perimeter edges. Phone intrinsics are retained separately
  // because the inset uses its own virtual pinhole camera.
  const uint ends[16] = {0,1,0,2,0,3,0,4,1,2,2,4,4,3,3,1};
  uint corner = ends[id];
  float3 local = 0;
  if (corner) {
    float2 uv = float2((corner - 1) % 2, (corner - 1) / 2);
    float2 xy = (uv - orbit.phoneIntrinsics.zw) * 0.4f / orbit.phoneIntrinsics.xy;
    local = float3(xy.x, -xy.y, -0.4f);
  }
  float3 p = (previewView(u, scene, orbit) * u.world * float4(local, 1)).xyz;
  FrustumOut out;
  out.position = -p.z > 0.05f ? previewPosition(p, u) : float4(2, 2, 2, 1);
  return out;
}
fragment float4 phoneFrustumFragment() { return float4(1, 0.55f, 0.08f, 1); }
kernel void compositeScanInset(texture2d<float, access::write> target [[texture(0)]],
  texture2d<float> inset [[texture(1)]], constant OrbitUniforms &orbit [[buffer(0)]], uint2 pixel [[thread_position_in_grid]]) {
  if (pixel.x >= target.get_width() || pixel.y >= target.get_height()) return;
  float2 uv = ((float2(pixel) + 0.5f) / float2(target.get_width(), target.get_height()) - orbit.rect.xy) / orbit.rect.zw;
  if (any(uv < 0) || any(uv > 1)) return;
  constexpr sampler linear(filter::linear, address::clamp_to_edge);
  float4 value = inset.sample(linear, uv);
  target.write(float4(value.rgb + float3(0.015f, 0.02f, 0.025f) * (1 - value.a), 1), pixel);
}
