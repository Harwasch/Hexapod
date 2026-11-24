import Foundation

#if canImport(os)
  @preconcurrency import os
#endif

public struct InferenceMetrics: Sendable {
  private var durations: [TimeInterval] = []
  private var completions: [TimeInterval] = []
  public private(set) var requestCount = 0
  public init() {}
  public mutating func record(duration: TimeInterval, completedAt: TimeInterval) {
    guard duration.isFinite, duration >= 0, completedAt.isFinite else { return }
    requestCount += 1
    durations.append(duration)
    completions.append(completedAt)
    if durations.count > PerceptionLimits.metricsWindow {
      durations.removeFirst()
      completions.removeFirst()
    }
  }
  public var averageDurationMilliseconds: Double {
    durations.isEmpty ? 0 : durations.reduce(0, +) * 1000 / Double(durations.count)
  }
  public var achievedHz: Double {
    guard let first = completions.first, let last = completions.last, last > first else { return 0 }
    return Double(completions.count - 1) / (last - first)
  }
}

/// An Instruments interval per complete request, with its duration and achieved completion rate.
struct InferenceSignpost: @unchecked Sendable {
  private let name: StaticString
  #if canImport(os)
    private static let log = OSLog(subsystem: "com.hexapod.app", category: "Perception")
    private let id: OSSignpostID
  #endif
  init(name: StaticString = "ForegroundInference") {
    self.name = name
    #if canImport(os)
      id = OSSignpostID(log: Self.log)
      os_signpost(.begin, log: Self.log, name: name, signpostID: id)
    #endif
  }
  func end(duration: TimeInterval, hz: Double? = nil) {
    #if canImport(os)
      if let hz {
        os_signpost(
          .end, log: Self.log, name: name, signpostID: id,
          "request_ms=%{public}.2f achieved_hz=%{public}.2f target_hz=%{public}.1f",
          duration * 1000, hz, PerceptionLimits.targetInferenceHz)
      } else {
        os_signpost(
          .end, log: Self.log, name: name, signpostID: id,
          "request_ms=%{public}.2f", duration * 1000)
      }
    #endif
  }
}
