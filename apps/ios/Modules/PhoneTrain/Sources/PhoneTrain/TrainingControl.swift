/// Pure step-boundary policy shared with Linux tests. Critical is latched for a
/// run: cooling cannot silently restart a measurement after an unsafe interval.
struct TrainingControl: Sendable {
  enum Decision: Equatable { case step, pause, throttle, critical, cancel }
  var active = true
  var manuallyPaused = false
  var cancelled = false
  var criticalObserved = false
  func decision(thermal: Int) -> Decision {
    if cancelled { return .cancel }
    if criticalObserved || thermal >= 3 { return .critical }
    if !active || manuallyPaused { return .pause }
    if thermal == 2 { return .throttle }
    return .step
  }
}
