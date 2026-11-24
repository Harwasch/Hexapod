import EntitySchema
import Perception
import Testing

@testable import ARAnswer

@Test func liftedPointsReprojectWithTheDisplayCameraWithoutAnotherSegmentation() throws {
  let original = CameraGeometry(
    fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4,
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
  let world = try #require(original.worldPoint(pixel: ImagePoint(x: 0.25, y: 0.75), depth: 2))
  var moved = original
  moved.transform[12] = 0.1
  let projection = SurfaceProjection(
    camera: moved, viewportToImage: [0, -0.6, 1, 0, 0, 0.8], capturedAt: 1)
  let view = try #require(projection.viewPoint(world))
  #expect(abs(view.x - (0.8 - 0.75) / 0.6) < 1e-12)
  #expect(abs(view.y - 0.2) < 1e-12)
  let packed = try #require(projection.uniforms)
  #expect(packed.count == 28)
  #expect(packed[12] == -0.1)
  #expect(packed[16] == 1 && packed[17] == 1)
  #expect(projection.isCurrent(at: 1 + PerceptionLimits.maximumCameraAge / 2))
  #expect(!projection.isCurrent(at: 1 + PerceptionLimits.maximumCameraAge * 2))
}

@Test func metalUniformLayoutProjectsAnAsymmetricRotatedCameraIntoPortrait() throws {
  let camera = CameraGeometry(
    fx: 800, fy: 810, cx: 960, cy: 720,
    imageWidth: 1920, imageHeight: 1440,
    transform: [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.3, 0.4, 0.5, 1])
  let world = try #require(camera.worldPoint(pixel: ImagePoint(x: 0.2, y: 0.8), depth: 3))
  let projection = SurfaceProjection(
    camera: camera, viewportToImage: [0, -0.6, 1, 0, 0, 0.8], capturedAt: 1)
  let u = try #require(projection.uniforms)
  let x = Float(world.x)
  let y = Float(world.y)
  let z = Float(world.z)
  let cx = u[0] * x + u[4] * y + u[8] * z + u[12]
  let cy = u[1] * x + u[5] * y + u[9] * z + u[13]
  let cz = u[2] * x + u[6] * y + u[10] * z + u[14]
  let depth = -cz
  let ix = u[16] * cx + u[18] * depth
  let iy = -u[17] * cy + u[19] * depth
  let vx = (u[20] * ix + u[22] * iy + u[24] * depth) / depth
  let vy = (u[21] * ix + u[23] * iy + u[25] * depth) / depth
  // Bound for the short Float32 shader operation chain, rather than a device-calibrated offset.
  let rounding = Float.ulpOfOne * 64
  #expect(abs(vx) < rounding)
  #expect(abs(vy - 0.2) < rounding)
}
