import Foundation

extension SceneSelection {
  /// A mask is valid only while the selected surfaces move by at most two viewport pixels.
  /// Inference refreshes it after motion; never drag an old silhouette with a new display transform.
  public func isRegistered(
    camera current: CameraGeometry, viewportToImage currentMapping: [Double],
    width: Double, height: Double
  ) -> Bool {
    guard let camera, !entities.isEmpty, width > 0, height > 0 else { return false }
    func view(_ point: ImagePoint, _ t: [Double]) -> ImagePoint? {
      guard t.count == 6, t.allSatisfy(\.isFinite) else { return nil }
      let determinant = t[0] * t[3] - t[1] * t[2]
      guard abs(determinant) > 1e-12 else { return nil }
      return ImagePoint(
        x: (t[3] * (point.x - t[4]) - t[2] * (point.y - t[5])) / determinant,
        y: (-t[1] * (point.x - t[4]) + t[0] * (point.y - t[5])) / determinant)
    }
    for entity in entities {
      // Centre and all corners include depth-dependent parallax and camera rotation.
      for sx in [-1.0, 0, 1] {
        for sy in [-1.0, 0, 1] {
          for sz in [-1.0, 0, 1] {
            var world = entity.anchor.world
            world.x += sx * entity.extent.x / 2
            world.y += sy * entity.extent.y / 2
            world.z += sz * entity.extent.z / 2
            guard let old = camera.project(world), let new = current.project(world),
              let before = view(old, viewportToImage), let after = view(new, currentMapping)
            else { return false }
            if hypot((before.x - after.x) * width, (before.y - after.y) * height) > 2 {
              return false
            }
          }
        }
      }
    }
    return true
  }
}
