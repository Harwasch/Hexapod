import Foundation

/// Single-flight background inference with one replacing latest-frame mailbox, never a timer.
/// Busy arrivals replace the mailbox; intermediate frames are dropped, not queued as requests.
public final class ForegroundInferenceLoop<Frame: Sendable, Result: Sendable>: @unchecked Sendable {
  private struct Input {
    let frame: Frame
    let capturedAt: TimeInterval
    let generation: Int
  }
  private let lock = NSLock()
  private var pending: Input?
  private var running = false
  private var enabled = true
  private var generation = 0
  private var statistics = InferenceMetrics()
  private var idleWaiters: [CheckedContinuation<Void, Never>] = []
  private let clock: @Sendable () -> TimeInterval
  private let process: @Sendable (Frame) async throws -> Result
  private let result: @Sendable (Result) -> Void
  private let failure: @Sendable () -> Void
  public init(
    clock: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
    process: @escaping @Sendable (Frame) async throws -> Result,
    result: @escaping @Sendable (Result) -> Void,
    failure: @escaping @Sendable () -> Void = {}
  ) {
    self.clock = clock
    self.process = process
    self.result = result
    self.failure = failure
  }
  public var metrics: InferenceMetrics { lock.withLock { statistics } }
  public func setEnabled(_ value: Bool) {
    lock.withLock {
      guard enabled != value else { return }
      enabled = value
      generation += 1
      pending = nil
      // An uncancellable Vision call still owns its slot until it actually returns.
    }
  }
  public func offer(_ frame: Frame, capturedAt: TimeInterval) {
    let start = lock.withLock {
      guard enabled else { return false }
      pending = Input(frame: frame, capturedAt: capturedAt, generation: generation)
      guard !running else { return false }
      running = true
      return true
    }
    if start { Task.detached(priority: .userInitiated) { await self.drain() } }
  }
  public func waitUntilIdle() async {
    await withCheckedContinuation { continuation in
      lock.withLock {
        if running { idleWaiters.append(continuation) } else { continuation.resume() }
      }
    }
  }
  private func take() -> Input? {
    lock.withLock {
      let next = enabled ? pending : nil
      pending = nil
      if next == nil {
        running = false
        idleWaiters.forEach { $0.resume() }
        idleWaiters.removeAll()
      }
      return next
    }
  }
  private func drain() async {
    while let input = take() {
      let start = clock()
      guard
        PerceptionLimits.isCurrent(
          capturedAt: input.capturedAt, now: start, maximumAge: PerceptionLimits.maximumInputAge)
      else { continue }
      let trace = InferenceSignpost()
      let output: Result?
      do { output = try await process(input.frame) } catch { output = nil }
      let end = clock()
      let state = lock.withLock {
        statistics.record(duration: end - start, completedAt: end)
        return (enabled && input.generation == generation, statistics.achievedHz)
      }
      trace.end(duration: end - start, hz: state.1)
      if state.0 {
        if let output { result(output) } else { failure() }
      }
    }
  }
}
