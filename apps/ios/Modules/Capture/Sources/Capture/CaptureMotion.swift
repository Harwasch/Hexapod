import Foundation

/// The motion blur the next frame will carry, predicted from consecutive ARKit poses.
///
/// During a shutter of `t` seconds, a camera turning at `ω` rad/s and moving at `v` m/s smears
/// a point at depth `d` across about
///
///     blur_px ≈ t · fx · (ω + v / d)
///
/// pixels (small angles; `fx` in recorded-image pixels; `v` taken whole, not only its part
/// across the view, so the estimate errs high). Holding one pixel at the capped 1/250 s shutter:
///
/// - 3840x2160 (fx ≈ 2830 px): ω ≤ 1 / (2830 · 0.004) = 0.088 rad/s, about 5°/s, or 0.13 m/s
///   sideways at 1.5 m;
/// - 1920x1440 (fx ≈ 1414 px): about 10°/s, or 0.27 m/s at 1.5 m.
///
/// So "steady" below one pixel, "slow down" to three, and "too fast" above; a 3 px smear at 4K
/// is still under 1.5 px once training downsamples to 1440p.
public struct MotionBlurMeter: Sendable {
  public enum Level: Equatable, Sendable { case steady, slowDown, tooFast }

  public static let steadyPixels = 1.0
  public static let tooFastPixels = 3.0
  /// When ARKit gives no exposure (simulator, older OS), assume the cap the app sets.
  public static let nominalShutterS = 1.0 / 250
  /// When the depth map has no confident centre sample.
  public static let nominalDepthM = 1.5
  /// Exponential smoothing time constant: about four frames at 15 fps, so a single jerk
  /// shows and a steady sweep does not flicker.
  public var smoothingS = 0.25

  private var previous: (time: Double, rotation: [Double], centre: [Double])?
  public private(set) var blurPixels = 0.0

  public init() {}

  public var level: Level {
    blurPixels < Self.steadyPixels
      ? .steady : blurPixels < Self.tooFastPixels ? .slowDown : .tooFast
  }

  /// `transform` is camera-to-world, row-major 4x4 (as CaptureFrame stores it).
  @discardableResult
  public mutating func update(
    time: Double, transform: [Double], focalPx: Double, shutterS: Double?, depthM: Double?
  ) -> Level {
    guard transform.count == 16, transform.allSatisfy(\.isFinite), focalPx > 0, time.isFinite
    else { return level }
    let rotation = [0, 1, 2, 4, 5, 6, 8, 9, 10].map { transform[$0] }
    let centre = [transform[3], transform[7], transform[11]]
    defer { previous = (time, rotation, centre) }
    guard let previous, time > previous.time else { return level }
    let dt = time - previous.time
    // Angle of R_prevᵀ R: trace = Σ_ij Rprev_ij R_ij.
    let trace = zip(previous.rotation, rotation).reduce(0) { $0 + $1.0 * $1.1 }
    let omega = acos(max(-1, min(1, (trace - 1) / 2))) / dt
    let speed =
      sqrt(zip(previous.centre, centre).reduce(0) { $0 + ($1.1 - $1.0) * ($1.1 - $1.0) }) / dt
    let shutter = shutterS.flatMap { $0 > 0 && $0.isFinite ? $0 : nil } ?? Self.nominalShutterS
    let depth = depthM.flatMap { $0 > 0.1 && $0.isFinite ? $0 : nil } ?? Self.nominalDepthM
    let instant = shutter * focalPx * (omega + speed / depth)
    let alpha = 1 - exp(-dt / max(smoothingS, 1e-3))
    blurPixels += alpha * (instant - blurPixels)
    return level
  }
}

/// What to do with the camera's exposure once it has metered the scene.
///
/// Auto-exposure lets the shutter run long indoors (1/30-1/60 s), which smears every frame
/// of a handheld sweep, and lets brightness and colour drift frame to frame, which the
/// splat then learns as view-dependent colour. After about a second of metering the app
/// holds what the camera chose: as-is when the shutter is already at most `maximumShutterS`,
/// else at that shutter with ISO raised by the same factor (more noise, far less blur). If the
/// sensor's ISO ceiling stops the compensation, the frames come out darker by `shortfallEV`.
public struct ExposurePlan: Equatable, Sendable {
  public static let maximumShutterS = 1.0 / 250
  public static let meteringS = 1.0

  public var mode: String
  public var durationS: Double
  public var iso: Double
  public var shortfallEV: Double

  public static func plan(
    meteredDurationS: Double, meteredISO: Double,
    minDurationS: Double, maxDurationS: Double, minISO: Double, maxISO: Double
  ) -> ExposurePlan {
    let cap = min(max(Self.maximumShutterS, minDurationS), maxDurationS)
    guard meteredDurationS > cap else {
      return ExposurePlan(
        mode: "locked", durationS: meteredDurationS, iso: meteredISO, shortfallEV: 0)
    }
    let wanted = meteredISO * meteredDurationS / cap
    let iso = min(max(wanted, minISO), maxISO)
    return ExposurePlan(
      mode: "custom", durationS: cap, iso: iso, shortfallEV: max(0, log2(wanted / iso)))
  }
}
