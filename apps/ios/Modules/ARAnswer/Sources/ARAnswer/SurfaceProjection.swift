import EntitySchema
import Foundation
import Perception

/// The display camera is independent of the camera that originally lifted the surface.
public struct SurfaceProjection: Sendable {
  public let camera: CameraGeometry
  public let viewportToImage: [Double]
  public let capturedAt: TimeInterval
  public init(camera: CameraGeometry, viewportToImage: [Double], capturedAt: TimeInterval) {
    self.camera = camera
    self.viewportToImage = viewportToImage
    self.capturedAt = capturedAt
  }
  /// Metal layout: worldToCamera (16), intrinsics (4), imageToView (4), translation/near (4).
  public var uniforms: [Float]? {
    guard camera.isValid else { return nil }
    let mapping = ViewportMapping(viewportToImage: viewportToImage)
    guard let origin = mapping.viewPoint(x: 0, y: 0),
      let x = mapping.viewPoint(x: 1, y: 0), let y = mapping.viewPoint(x: 0, y: 1)
    else { return nil }
    let t = camera.transform
    let inverse = [
      t[0], t[4], t[8], 0, t[1], t[5], t[9], 0, t[2], t[6], t[10], 0,
      -(t[0] * t[12] + t[1] * t[13] + t[2] * t[14]),
      -(t[4] * t[12] + t[5] * t[13] + t[6] * t[14]),
      -(t[8] * t[12] + t[9] * t[13] + t[10] * t[14]), 1,
    ]
    let intrinsics = [
      camera.fx / Double(camera.imageWidth), camera.fy / Double(camera.imageHeight),
      camera.cx / Double(camera.imageWidth), camera.cy / Double(camera.imageHeight),
    ]
    let affine: [Double] = [x.x - origin.x, x.y - origin.y, y.x - origin.x, y.y - origin.y]
    let translation: [Double] = [origin.x, origin.y, PerceptionLimits.minimumProjectionDepth, 0]
    var packed: [Double] = inverse
    packed.append(contentsOf: intrinsics)
    packed.append(contentsOf: affine)
    packed.append(contentsOf: translation)
    return packed.map { Float($0) }
  }
  public func viewPoint(_ world: Vector3) -> (x: Double, y: Double)? {
    guard let image = camera.project(world) else { return nil }
    return ViewportMapping(viewportToImage: viewportToImage).viewPoint(x: image.x, y: image.y)
  }
  public func isCurrent(at time: TimeInterval) -> Bool {
    PerceptionLimits.isCurrent(
      capturedAt: capturedAt, now: time, maximumAge: PerceptionLimits.maximumCameraAge)
  }
}
