import EntitySchema
import Testing

@testable import Perception

private let surfaceCamera = CameraGeometry(
  fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4,
  transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

@Test func maskedDepthIsLiftedOnceIntoWorldSpaceTriangles() throws {
  let mesh = try DepthSurface.lift(
    mask: SubjectMask(width: 4, height: 4, pixels: Array(repeating: 255, count: 16)),
    depth: DepthImage(width: 4, height: 4, metres: Array(repeating: 2, count: 16)),
    camera: surfaceCamera)
  #expect(mesh.points.count == 16)
  #expect(mesh.triangles.count == 54)
  #expect(mesh.points[0] == Vector3(x: -0.75, y: 0.75, z: -2))
  for (index, point) in mesh.points.enumerated() {
    let image = try #require(surfaceCamera.project(point))
    #expect(image.x == (Double(index % 4) + 0.5) / 4)
    #expect(image.y == (Double(index / 4) + 0.5) / 4)
  }
}

@Test func missingDepthAndMaskHolesNeverBecomeBridgingTriangles() throws {
  var depth = Array(repeating: Float(2), count: 16)
  depth[5] = .nan
  depth[6] = 0
  var pixels = Array(repeating: UInt8(255), count: 16)
  pixels[9] = 0
  let mesh = try DepthSurface.lift(
    mask: SubjectMask(width: 4, height: 4, pixels: pixels),
    depth: DepthImage(width: 4, height: 4, metres: depth), camera: surfaceCamera)
  #expect(mesh.points.count == 13)
  #expect(mesh.triangles.count < 54)
  #expect(mesh.triangles.allSatisfy { Int($0) < mesh.points.count })
  #expect(mesh.points.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
}

@Test func lidarDiscontinuitiesDoNotStretchTheOutlineToTheBackground() throws {
  var depth = Array(repeating: Float(2), count: 16)
  depth[5] = 100
  let mesh = try DepthSurface.lift(
    mask: SubjectMask(width: 4, height: 4, pixels: Array(repeating: 255, count: 16)),
    depth: DepthImage(width: 4, height: 4, metres: depth), camera: surfaceCamera)
  let distant = try #require(mesh.points.firstIndex { $0.z == -100 })
  #expect(!mesh.triangles.contains(UInt32(distant)))
}
