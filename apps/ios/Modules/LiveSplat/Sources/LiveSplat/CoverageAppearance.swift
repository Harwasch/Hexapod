import Foundation

/// All coverage appearance thresholds live here and travel to MSL verbatim as uniforms.
/// Prototype targets for observed surfaces, not a reconstruction-quality guarantee.
public enum CoverageAppearance {
  // Eight observations, six consecutive high-confidence samples, and either three
  // 15° azimuth bins or two elevation bands develop a cell to its true colour.
  public static let observations: Float = 8
  public static let confidentRun: Float = 6
  public static let azimuthBins: Float = 3
  public static let elevationBins: Float = 2
  public static let transitionSeconds: Float = 0.7
  public static let pulseSeconds: Float = 1.2
  public static let shimmerPeriod: Float = 3.6
  public static let shimmerAmplitude: Float = 0.025
  public static let minimumOpacity: Float = 0.38
  public static let pulseStrength: Float = 0.16
  public static let coolTint = SIMD3<Float>(0.32, 0.43, 0.55)

  public static func ease(_ value: Float) -> Float {
    let t = min(1, max(0, value))
    return t * t * (3 - 2 * t)
  }
  public static func score(observations: Float, confidentRun: Float, azimuth: Int, elevation: Int)
    -> Float
  {
    let spread = min(1, max(Float(azimuth) / azimuthBins, Float(elevation) / elevationBins))
    return ease(min(observations / Self.observations, confidentRun / Self.confidentRun, spread))
  }
  public static func developed(previous: Float, target: Float, elapsed: Float) -> Float {
    previous + (target - previous) * ease(elapsed / transitionSeconds)
  }
  public struct Appearance: Equatable, Sendable {
    public let colour: SIMD3<Float>
    public let opacity: Float
  }
  public static func appearance(
    colour: SIMD3<Float>, progress: Float, time: Float = 0,
    pulseAge: Float = .infinity
  ) -> Appearance {
    let p = min(1, max(0, progress))
    let luminance = colour.x * 0.2126 + colour.y * 0.7152 + colour.z * 0.0722
    let cool = SIMD3<Float>(repeating: luminance) * 0.45 + coolTint * 0.55
    let shimmer = sin(time * 2 * .pi / shimmerPeriod) * shimmerAmplitude * (1 - p)
    let pulse =
      pulseAge >= 0 && pulseAge < pulseSeconds
      ? sin(.pi * pulseAge / pulseSeconds) * pulseStrength : 0
    let base = p == 1 ? colour : cool + (colour - cool) * p
    let rgb = base + SIMD3<Float>(repeating: shimmer + pulse)
    return Appearance(colour: rgb, opacity: minimumOpacity + (1 - minimumOpacity) * p)
  }
}

/// Portable ABI definitions: float4/uint4 groups match LiveSplat.metal exactly.
struct CoverageStyleUniforms {
  var thresholds = SIMD4<Float>(
    CoverageAppearance.observations, CoverageAppearance.confidentRun,
    CoverageAppearance.azimuthBins, CoverageAppearance.elevationBins)
  var timing = SIMD4<Float>(
    0, CoverageAppearance.transitionSeconds, CoverageAppearance.pulseSeconds,
    CoverageAppearance.shimmerPeriod)
  var effects = SIMD4<Float>(
    CoverageAppearance.minimumOpacity, CoverageAppearance.shimmerAmplitude,
    CoverageAppearance.pulseStrength, 1)
  var tint = SIMD4<Float>(CoverageAppearance.coolTint, 0)
}
struct CoverageGPUCell {
  var key = SIMD4<Int32>(repeating: 0)
  // Azimuth bitmask, elevation bitmask, observations, last-seen epoch.
  var views = SIMD4<UInt32>(repeating: 0)
}
struct CoverageSummary {
  var centerCount = SIMD4<Float>(repeating: 0)
  // Cells with insufficient angular spread, reserved.
  var counts = SIMD4<UInt32>(repeating: 0)
}
/// Per-surfel animation state in a separate fixed buffer; the surfel stays 64 bytes.
struct CoverageDevelopment {
  var animation = SIMD4<Float>(0, 0, 0, -1)  // previous, target, transition start, pulse start
}
