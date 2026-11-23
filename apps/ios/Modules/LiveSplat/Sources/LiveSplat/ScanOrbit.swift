import Foundation

/// Normalized top-trailing viewport shared by the postprocess and SwiftUI gesture
/// host. Geometry stays on the GPU; this value carries only camera controls.
public struct ScanOrbit: Sendable, Equatable {
  public var enlarged = false
  public private(set) var yaw: Float = 0
  public private(set) var pitch: Float = 0.35
  public init() {}
  public var rect: SIMD4<Float> {
    enlarged ? SIMD4(0.06, 0.16, 0.88, 0.58) : SIMD4(0.60, 0.17, 0.36, 0.25)
  }
  public mutating func rotate(horizontal: Float, vertical: Float) {
    guard horizontal.isFinite, vertical.isFinite else { return }
    yaw = (yaw + horizontal).truncatingRemainder(dividingBy: 2 * .pi)
    pitch = min(1.2, max(-0.8, pitch + vertical))
  }
  public func angle(at seconds: Double) -> Float {
    guard seconds.isFinite else { return yaw }
    return yaw + Float(seconds.truncatingRemainder(dividingBy: 120)) * (2 * .pi / 120)
  }
}
