#include <metal_stdlib>
using namespace metal;

// Kept separate from the 64-byte surfel ABI. Asserted by CoverageHairTests.
struct CoverageCell {
  short2 normal;
  uchar credit[14];
  ushort views, epoch, flags;
  half bestMmPerPx;
  ushort padding[3];
};
static_assert(sizeof(CoverageCell) == 32, "Coverage cell stride");
struct CoverageGeometry { int4 key; float4 position; };
struct CoverageHistory { short2 first[14]; uint mask; };
struct CoverageParameters { float4 shape, detail; uint4 counts; };
struct CoveragePreviewUniforms {
  float4x4 world, view;
  float4 intrinsics, displayX, displayY, inverseX, inverseY, dimensions;
  uint4 settings, flags;
};
constant uint coverageCapacity = 32768;
constant float coverageCellSize = 0.1f;
short2 coverageEncode(float3 n) {
  n /= abs(n.x) + abs(n.y) + abs(n.z);
  float2 p = n.xy;
  if (n.z < 0) p = (1 - abs(p.yx)) * select(float2(-1), float2(1), p >= 0);
  return short2(round(p * 32767));
}
float3 coverageDecode(short2 encoded) {
  float2 p = float2(encoded) / 32767;
  float3 n = float3(p, 1 - abs(p.x) - abs(p.y));
  float t = max(0.0f, -n.z);
  n.xy += select(float2(t), float2(-t), n.xy >= 0);
  return normalize(n);
}
float3 coverageHair(uint h, uint count, float3 n) {
  float3 tangent = normalize(cross(abs(n.z) < 0.999f ? float3(0,0,1) : float3(0,1,0), n));
  float z = (float(h) + 0.5f) / float(count), r = sqrt(max(0.0f, 1 - z*z)), phi = float(h) * 2.39996323f;
  return tangent * (r*cos(phi)) + cross(n, tangent) * (r*sin(phi)) + n*z;
}
// Matching CoverageCreditReference.credit and creditedByte, in view units before quantization.
float coverageCredit(float cosine, float mmPerPx, constant CoverageParameters &p) {
  if (!isfinite(cosine) || cosine <= p.shape.x || !isfinite(mmPerPx) || mmPerPx <= 0) return 0;
  float resolution = p.shape.w > 0 ? min(1.0f, p.shape.w / mmPerPx) : 1;
  return resolution * pow(min(1.0f, (cosine - p.shape.x) / (1 - p.shape.x)), p.shape.y);
}
float coverageRemaining(CoverageCell cell, uint h, constant CoverageParameters &p) {
  return max(1 - float(cell.credit[h]) / 255, p.detail.x > -1 && !(cell.flags & (1u << h)) ? p.detail.w : 0.0f);
}
float3 coverageWorld(float2 uv, float d, constant CoveragePreviewUniforms &u) {
  float2 xy = (uv - u.intrinsics.zw) * d / u.intrinsics.xy;
  return (u.world * float4(xy.x, -xy.y, -d, 1)).xyz;
}
bool coverageDepth(float d, float c) { return isfinite(d) && d >= 0.1f && d <= 8 && c > 1.5f / 255; }
float4 coverageProject(float3 world, constant CoveragePreviewUniforms &u) {
  float3 p = (u.view * float4(world, 1)).xyz;
  float d = -p.z;
  if (d < 0.05f || d > 100) return float4(2,2,2,1);
  float3 uv = float3(float2(p.x, -p.y) * u.intrinsics.xy / d + u.intrinsics.zw, 1);
  float2 screen = float2(dot(u.displayX.xyz, uv), dot(u.displayY.xyz, uv));
  return float4(screen.x*2-1, 1-screen.y*2, (d-0.05f)/99.95f, 1);
}
// Same rotating sample indices and confidence/depth admission as surfel fusion.
// One nonblocking bucket winner avoids read/write races, with retry on the next keyframe.
kernel void observeCoverageCells(device CoverageCell *cells [[buffer(0)]],
  device CoverageGeometry *geometry [[buffer(1)]], device atomic_uint *claims [[buffer(2)]],
  device CoverageHistory *history [[buffer(3)]], constant CoveragePreviewUniforms &u [[buffer(4)]],
  constant CoverageParameters &params [[buffer(5)]], texture2d<float, access::read> depth [[texture(0)]],
  texture2d<float, access::read> confidence [[texture(1)]], uint id [[thread_position_in_grid]]) {
  if (id >= u.settings.x) return;
  uint stride = u.settings.z, columns = (depth.get_width() + stride - 1) / stride;
  uint phase = u.settings.y % (stride * stride);
  uint2 pixel = uint2(id % columns, id / columns) * stride + uint2(phase % stride, phase / stride);
  if (pixel.x >= depth.get_width() || pixel.y >= depth.get_height()) return;
  float d = depth.read(pixel).r;
  if (!coverageDepth(d, confidence.read(pixel).r)) return;
  float3 point = coverageWorld(float2(pixel) / u.dimensions.zw, d, u);
  if (!all(isfinite(point)) || any(abs(point) >= 1000000)) return;
  int4 key = int4(int3(floor(point / coverageCellSize)), 0);
  uint b = ((uint(key.x)*73856093u) ^ (uint(key.y)*19349663u) ^ (uint(key.z)*83492791u)) % (coverageCapacity/4);
  uint expected = 0;
  if (!atomic_compare_exchange_weak_explicit(claims+b, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) return;
  uint start = b*4, slot = start;
  bool found = false;
  for (uint i = start; i < start+4; ++i) {
    if (!cells[i].views) { slot = i; break; }
    if (all(geometry[i].key.xyz == key.xyz)) { slot = i; found = true; break; }
    if (cells[i].views < cells[slot].views) slot = i;
  }
  CoverageCell cell = cells[slot];
  float3 view = normalize(u.world[3].xyz - point), normal = view;
  uint2 px = uint2(min(pixel.x+1, depth.get_width()-1), pixel.y);
  uint2 py = uint2(pixel.x, min(pixel.y+1, depth.get_height()-1));
  float dx = depth.read(px).r, dy = depth.read(py).r;
  if (px.x != pixel.x && py.y != pixel.y && coverageDepth(dx, confidence.read(px).r)
    && coverageDepth(dy, confidence.read(py).r) && abs(dx-d) < d*0.05f && abs(dy-d) < d*0.05f) {
    float3 n = cross(coverageWorld(float2(px)/u.dimensions.zw, dx, u)-point,
      coverageWorld(float2(py)/u.dimensions.zw, dy, u)-point);
    if (dot(n,n) > 1e-12f) normal = normalize(n);
  }
  if (found) {
    float3 old = coverageDecode(cell.normal);
    if (dot(normal, old) < 0) normal = -normal;
    normal = normalize(mix(old, normal, params.detail.y));
    geometry[slot].position.xyz = mix(geometry[slot].position.xyz, point, params.detail.y);
  } else {
    cell = {};
    history[slot].mask = 0;
    if (dot(normal, view) < 0) normal = -normal;
    geometry[slot].key = key;
    geometry[slot].position = float4(point, 1);
  }
  geometry[slot].key.w = int(params.counts.y);
  cell.normal = coverageEncode(normal);
  cell.views = ushort(min(65535u, uint(cell.views)+1));
  cell.epoch = ushort(u.settings.y);
  cells[slot] = cell;
}
// Separate encoder after observation: each seen cell receives exactly one view's credit.
kernel void creditCoverageCells(device CoverageCell *cells [[buffer(0)]],
  device const CoverageGeometry *geometry [[buffer(1)]], device CoverageHistory *history [[buffer(3)]],
  constant CoveragePreviewUniforms &u [[buffer(4)]], constant CoverageParameters &p [[buffer(5)]],
  uint id [[thread_position_in_grid]]) {
  if (id >= coverageCapacity) return;
  CoverageCell cell = cells[id];
  if (!cell.views || uint(geometry[id].key.w) != p.counts.y) return;
  float3 delta = u.world[3].xyz - geometry[id].position.xyz;
  float distance = length(delta);
  if (distance <= 0) return;
  float3 view = delta / distance, n = coverageDecode(cell.normal);
  float incidence = dot(n, view);
  if (incidence <= 0) return;
  float mm = distance * 1000 / (u.intrinsics.x * u.dimensions.z) / incidence;
  cell.bestMmPerPx = cell.bestMmPerPx > 0 ? min(cell.bestMmPerPx, half(mm)) : half(mm);
  for (uint h = 0; h < p.counts.x; ++h) {
    float credit = coverageCredit(dot(view, coverageHair(h, p.counts.x, n)), mm, p);
    if (credit <= 0) continue;
    uint bit = 1u << h;
    if (p.detail.x > -1) {
      if (!(history[id].mask & bit)) {
        history[id].first[h] = coverageEncode(view); history[id].mask |= bit;
      } else if (dot(view, coverageDecode(history[id].first[h])) <= p.detail.x) cell.flags |= ushort(bit);
    }
    cell.credit[h] = uchar(min(255.0f, floor(float(cell.credit[h]) + credit * 255 / p.shape.z + 0.5f)));
  }
  cells[id] = cell;
}
// Compact visible, unfinished hairs into a bounded indirect draw. Every quality level
// walks a fixed decimated subset; filled/offscreen instances consume no draw slots.
kernel void prepareCoverageHairs(device const CoverageCell *cells [[buffer(0)]],
  device const CoverageGeometry *geometry [[buffer(1)]], device uint2 *hairs [[buffer(2)]],
  device atomic_uint *args [[buffer(3)]], constant CoveragePreviewUniforms &u [[buffer(4)]],
  constant CoverageParameters &p [[buffer(5)]], uint id [[thread_position_in_grid]]) {
  uint slot = id * p.counts.w;
  if (slot >= coverageCapacity || !cells[slot].views) return;
  float4 origin = coverageProject(geometry[slot].position.xyz, u);
  // A small margin retains hairs that extend across the screen edge.
  if (origin.z < 0 || origin.z > 1 || any(abs(origin.xy) > 1.2f)) return;
  CoverageCell cell = cells[slot];
  float3 n = coverageDecode(cell.normal);
  for (uint h = 0; h < p.counts.x; ++h) {
    float remaining = coverageRemaining(cell, h, p);
    if (remaining <= 0) continue;
    float4 tip = coverageProject(geometry[slot].position.xyz + coverageHair(h, p.counts.x, n)*p.detail.z*remaining, u);
    if (any(abs(origin.xy) > 1) && any(abs(tip.xy) > 1)) continue;
    uint index = atomic_fetch_add_explicit(args+1, 1u, memory_order_relaxed);
    if (index < p.counts.z) hairs[index] = uint2(slot, h);
  }
}
kernel void finishCoverageHairs(device uint *args [[buffer(3)]], constant CoverageParameters &p [[buffer(5)]],
  uint id [[thread_position_in_grid]]) {
  if (id) return;
  args[0] = 2; args[1] = min(args[1], p.counts.z); args[2] = 0; args[3] = 0;
}
struct CoverageHairOut { float4 position [[position]]; float3 color; float distance; };
vertex CoverageHairOut coverageHairVertex(device const CoverageCell *cells [[buffer(0)]],
  device const CoverageGeometry *geometry [[buffer(1)]], device const uint2 *hairs [[buffer(2)]],
  constant CoveragePreviewUniforms &u [[buffer(4)]], constant CoverageParameters &p [[buffer(5)]],
  uint vertexID [[vertex_id]], uint instance [[instance_id]]) {
  uint2 hair = hairs[instance];
  CoverageCell cell = cells[hair.x];
  float remaining = coverageRemaining(cell, hair.y, p);
  float3 point = geometry[hair.x].position.xyz;
  point += coverageHair(hair.y, p.counts.x, coverageDecode(cell.normal)) * (float(vertexID)*p.detail.z*remaining + 0.002f);
  CoverageHairOut out;
  out.position = coverageProject(point, u);
  out.distance = -(u.view * float4(point,1)).z;
  out.color = mix(float3(0.05f,0.85f,0.15f), float3(1,0.5f,0.04f), remaining);
  return out;
}
fragment float4 coverageHairFragment(CoverageHairOut in [[stage_in]],
  constant CoveragePreviewUniforms &u [[buffer(4)]], texture2d<float> depth [[texture(0)]],
  texture2d<float> confidence [[texture(1)]]) {
  float3 screen = float3(in.position.xy / u.dimensions.xy, 1);
  float2 uv = float2(dot(u.inverseX.xyz, screen), dot(u.inverseY.xyz, screen));
  constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
  float d = depth.sample(nearest, uv).r;
  if (coverageDepth(d, confidence.sample(nearest,uv).r) && in.distance > d + max(0.03f,d*0.02f)) discard_fragment();
  return float4(in.color, 1);
}
