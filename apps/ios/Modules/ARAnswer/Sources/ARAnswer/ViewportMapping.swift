import Foundation

/// ARFrame's inverse displayTransform, in normalized, top-left image/view coordinates.
/// Both taps and mask rasterization use this mapping; no safe-area or pixel offset is added.
public struct ViewportMapping: Sendable {
  public let viewportToImage: [Double]
  public init(viewportToImage: [Double]) { self.viewportToImage = viewportToImage }
  public func imagePoint(x: Double, y: Double) -> (x: Double, y: Double)? {
    let t = viewportToImage
    guard t.count == 6, t.allSatisfy(\.isFinite), x.isFinite, y.isFinite else { return nil }
    return (t[0] * x + t[2] * y + t[4], t[1] * x + t[3] * y + t[5])
  }
  public func viewPoint(x: Double, y: Double) -> (x: Double, y: Double)? {
    let t = viewportToImage
    guard t.count == 6, t.allSatisfy(\.isFinite), x.isFinite, y.isFinite else { return nil }
    let determinant = t[0] * t[3] - t[1] * t[2]
    guard abs(determinant) > 1e-12 else { return nil }
    return (
      (t[3] * (x - t[4]) - t[2] * (y - t[5])) / determinant,
      (-t[1] * (x - t[4]) + t[0] * (y - t[5])) / determinant
    )
  }
}
