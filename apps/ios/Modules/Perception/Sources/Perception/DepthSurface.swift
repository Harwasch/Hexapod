import EntitySchema
import Foundation

/// Captured LiDAR samples in the AR world frame. Indices join only neighbouring valid samples.
public struct DepthSurface: Sendable {
  public let points: [Vector3]
  public let triangles: [UInt32]
  public init(points: [Vector3], triangles: [UInt32]) {
    self.points = points
    self.triangles = triangles
  }
  public static func lift(mask: SubjectMask, depth: DepthImage, camera: CameraGeometry) throws
    -> Self
  {
    guard mask.isValid, camera.isValid, depth.width > 0, depth.height > 0,
      depth.metres.count / depth.width == depth.height,
      depth.metres.count % depth.width == 0
    else { throw PerceptionError.invalidGeometry }
    var points: [Vector3] = []
    points.reserveCapacity(depth.metres.count)
    var indices = Array(repeating: -1, count: depth.metres.count)
    for y in 0..<depth.height {
      for x in 0..<depth.width {
        let u = (Double(x) + 0.5) / Double(depth.width)
        let v = (Double(y) + 0.5) / Double(depth.height)
        let mx = min(mask.width - 1, Int(u * Double(mask.width)))
        let my = min(mask.height - 1, Int(v * Double(mask.height)))
        let d = Double(depth.metres[y * depth.width + x])
        guard mask.pixels[my * mask.width + mx] > 0, d.isFinite, d > 0,
          let world = camera.worldPoint(pixel: ImagePoint(x: u, y: v), depth: d)
        else { continue }
        indices[y * depth.width + x] = points.count
        points.append(world)
      }
    }
    var triangles: [UInt32] = []
    if depth.width > 1, depth.height > 1 {
      triangles.reserveCapacity((depth.width - 1) * (depth.height - 1) * 6)
    }
    let relativeJump = Float(PerceptionLimits.maximumRelativeDepthJump)
    func triangle(_ a: Int, _ b: Int, _ c: Int) {
      guard indices[a] >= 0, indices[b] >= 0, indices[c] >= 0 else { return }
      let da = depth.metres[a]
      let db = depth.metres[b]
      let dc = depth.metres[c]
      let minimum = min(da, min(db, dc))
      let maximum = max(da, max(db, dc))
      guard maximum - minimum <= minimum * relativeJump else { return }
      triangles.append(UInt32(indices[a]))
      triangles.append(UInt32(indices[b]))
      triangles.append(UInt32(indices[c]))
    }
    if depth.width > 1, depth.height > 1 {
      for y in 0..<(depth.height - 1) {
        for x in 0..<(depth.width - 1) {
          let a = y * depth.width + x
          let b = a + 1
          let c = a + depth.width
          let d = c + 1
          triangle(a, c, b)
          triangle(b, c, d)
        }
      }
    }
    return Self(points: points, triangles: triangles)
  }
}

/// Geometry stays paired with its entity and observation time, independently of the display camera.
public struct SubjectSurface: Sendable {
  public let entity: Entity
  public let mesh: DepthSurface
  public let capturedAt: TimeInterval
  public init(entity: Entity, mesh: DepthSurface, capturedAt: TimeInterval) {
    self.entity = entity
    self.mesh = mesh
    self.capturedAt = capturedAt
  }
}
