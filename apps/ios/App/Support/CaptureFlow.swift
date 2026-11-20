import Foundation

/// The UI cannot return to idle until the accepted writes have drained and a manifest exists.
enum CapturePhase: Equatable, Sendable {
  case idle, starting
  case recording(Date)
  case stopping
  case saved(URL)
  case failed
  var isBusy: Bool {
    switch self {
    case .starting, .recording, .stopping: true
    default: false
    }
  }
  func elapsed(at date: Date) -> TimeInterval {
    guard case .recording(let start) = self else { return 0 }
    return max(0, date.timeIntervalSince(start))
  }
  static func time(_ seconds: TimeInterval) -> String {
    let value = max(0, Int(seconds))
    return String(format: "%02d:%02d", value / 60, value % 60)
  }
}
