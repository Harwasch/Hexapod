#include <metal_stdlib>
using namespace metal;

struct SurfaceCamera {
    float4x4 worldToCamera;
    float4 intrinsics; // fx/W, fy/H, cx/W, cy/H
    float4 imageToView; // a,b,c,d
    float4 translation; // tx,ty,near,unused
};
struct SurfaceVertex { float4 position [[position]]; };
vertex SurfaceVertex worldSurfaceVertex(uint index [[vertex_id]],
                                       const device float4 *points [[buffer(0)]],
                                       constant SurfaceCamera &camera [[buffer(1)]]) {
    float3 p = (camera.worldToCamera * points[index]).xyz;
    float depth = -p.z;
    // Homogeneous projection avoids division by depth and lets Metal clip the near plane.
    float2 image = float2(camera.intrinsics.x * p.x, -camera.intrinsics.y * p.y)
                   + camera.intrinsics.zw * depth;
    float2 view = float2(camera.imageToView.x * image.x + camera.imageToView.z * image.y,
                        camera.imageToView.y * image.x + camera.imageToView.w * image.y)
                  + camera.translation.xy * depth;
    return {float4(2.0 * view.x - depth, depth - 2.0 * view.y,
                   depth - camera.translation.z, depth)};
}
fragment float4 worldSurfaceMask() { return float4(1.0); }
