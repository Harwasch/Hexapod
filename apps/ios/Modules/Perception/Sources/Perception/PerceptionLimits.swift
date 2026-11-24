import Foundation

/// Targets and bounds, not an inference timer. The next available frame starts immediately.
public enum PerceptionLimits {
  public static let targetInferenceHz = 10.0
  public static let displayFramesPerSecond = 60
  public static let maximumInFlightRequests = 1
  public static let maximumRetainedFrames = 1
  public static let maximumRetainedResults = 1
  public static let maximumPendingTaps = 2
  public static let maximumTapAge = 1.0
  public static let maximumInputAge = 2 / targetInferenceHz
  public static let maximumCameraAge = 3.0 / Double(displayFramesPerSecond)
  public static let maximumGPUFramesInFlight = 3
  public static let maximumAssociationPoints = 128
  public static let metricsWindow = 30
  /// Avoid stitching foreground to a background return across a LiDAR discontinuity.
  public static let maximumRelativeDepthJump = 0.05
  public static let minimumProjectionDepth = 0.001
  public static func isCurrent(
    capturedAt: TimeInterval, now: TimeInterval, maximumAge: TimeInterval
  ) -> Bool {
    capturedAt.isFinite && now.isFinite && now >= capturedAt && now - capturedAt <= maximumAge
  }
}
