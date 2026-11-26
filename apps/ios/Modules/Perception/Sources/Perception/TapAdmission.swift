import Foundation

#if canImport(os)
  import os
#endif

public enum SelectionRefusal: Error, Equatable, Sendable, CustomStringConvertible {
  case trackingNotReady, noDepth, disabled, noViewport, unmappedTap
  case staleGeneration, queueFull, noDriver, noFrame, expired
  case frameCopyFailed(String)
  case inferenceFailed(String)

  public var description: String {
    switch self {
    case .trackingNotReady: "trackingNotReady"
    case .noDepth: "noDepth"
    case .disabled: "disabled"
    case .noViewport: "noViewport"
    case .unmappedTap: "unmappedTap"
    case .staleGeneration: "staleGeneration"
    case .queueFull: "queueFull"
    case .noDriver: "noDriver"
    case .noFrame: "noFrame"
    case .expired: "expired"
    case .frameCopyFailed(let error): "frameCopyFailed: \(error)"
    case .inferenceFailed(let error): "inferenceFailed: \(error)"
    }
  }
  var canRetry: Bool { self == .trackingNotReady || self == .noDepth || self == .noFrame }
}

public enum TapAdmissionResult: Equatable, Sendable {
  case accepted
  case waiting(SelectionRefusal)
  case refused(SelectionRefusal)
}

/// The caller owns synchronization. Waiting taps retain normalized coordinates, never ARFrames.
public struct TapAdmission<Frame: Sendable>: Sendable {
  private enum Entry: Sendable {
    case waiting(ImagePoint, deadline: TimeInterval, reason: SelectionRefusal)
    case ready(Frame, expires: TimeInterval)
  }
  private var entries: [Entry] = []
  public init() {}
  public var isEmpty: Bool { entries.isEmpty }
  /// Borrow a scheduling frame; consumption stays in the single inference worker.
  public var readyFrame: Frame? {
    guard let first = entries.first, case .ready(let frame, _) = first else { return nil }
    return frame
  }
  public static var retryDuration: TimeInterval { 1.5 }

  public mutating func submit(
    _ point: ImagePoint, now: TimeInterval,
    prepare: (ImagePoint) -> Result<Frame, SelectionRefusal>
  ) -> TapAdmissionResult {
    guard entries.count < PerceptionLimits.maximumPendingTaps else { return .refused(.queueFull) }
    switch prepare(point) {
    case .success(let frame):
      entries.append(.ready(frame, expires: now + PerceptionLimits.maximumTapAge))
      return .accepted
    case .failure(let reason) where reason.canRetry:
      entries.append(.waiting(point, deadline: now + Self.retryDuration, reason: reason))
      return .waiting(reason)
    case .failure(let reason): return .refused(reason)
    }
  }

  public mutating func retry(
    now: TimeInterval, prepare: (ImagePoint) -> Result<Frame, SelectionRefusal>
  ) -> [TapAdmissionResult] {
    var events = expire(now: now)
    entries = entries.compactMap { entry in
      guard case .waiting(let point, let deadline, _) = entry else { return entry }
      switch prepare(point) {
      case .success(let frame):
        events.append(.accepted)
        return .ready(frame, expires: now + PerceptionLimits.maximumTapAge)
      case .failure(let reason) where reason.canRetry:
        return .waiting(point, deadline: deadline, reason: reason)
      case .failure(let reason):
        events.append(.refused(reason))
        return nil
      }
    }
    return events
  }

  public mutating func expire(now: TimeInterval) -> [TapAdmissionResult] {
    var events: [TapAdmissionResult] = []
    entries.removeAll { entry in
      switch entry {
      case .waiting(_, let deadline, let reason) where now >= deadline:
        events.append(.refused(reason))
        return true
      case .ready(_, let deadline) where now >= deadline:
        events.append(.refused(.expired))
        return true
      default: return false
      }
    }
    return events
  }

  public mutating func take(now: TimeInterval) -> (frame: Frame?, events: [TapAdmissionResult]) {
    let events = expire(now: now)
    guard let first = entries.first, case .ready(let frame, _) = first else { return (nil, events) }
    entries.removeFirst()
    return (frame, events)
  }

  public mutating func cancel(reason: SelectionRefusal) -> [TapAdmissionResult] {
    let events = entries.map { _ in TapAdmissionResult.refused(reason) }
    entries.removeAll()
    return events
  }
}

/// Shared by the app and pipeline self-check; compile away Apple logging on Linux.
public enum CameraDiagnostics {
  public static func record(_ message: String) {
    #if canImport(os)
      let log = OSLog(subsystem: "com.harwasch.hex", category: "CameraSelection")
      os_log("%{public}@", log: log, type: .info, message)
      os_signpost(.event, log: log, name: "CameraSelection", "%{public}@", message)
    #endif
  }
}

/// Portable frame admission; the owned-buffer factory runs only after readiness checks.
public enum SelectionFrameAdmission {
  public static func capture<Frame>(
    enabled: Bool, hasViewport: Bool, trackingNormal: Bool, hasDepth: Bool,
    copy: () throws -> Frame
  ) -> Result<Frame, SelectionRefusal> {
    guard enabled else { return .failure(.disabled) }
    guard hasViewport else { return .failure(.noViewport) }
    guard trackingNormal else { return .failure(.trackingNotReady) }
    guard hasDepth else { return .failure(.noDepth) }
    do { return .success(try copy()) } catch {
      return .failure(.frameCopyFailed(String(describing: error)))
    }
  }
}
