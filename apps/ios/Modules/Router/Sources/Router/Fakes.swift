import EntitySchema
import Foundation

public struct FakePlanner: Planning {
  public var plans: [String: AnswerPlan]
  public var fallback: AnswerPlan
  public init(
    plans: [String: AnswerPlan] = [:],
    fallback: AnswerPlan = .init(subject: "tree", tools: [.measure])
  ) {
    self.plans = plans
    self.fallback = fallback
  }
  public func plan(_ transcript: String) async throws -> AnswerPlan {
    try Task.checkCancellation()
    return plans[transcript] ?? fallback
  }
}

/// Public routing seam for isolated consumers; AnswerRouter is the production owner.
public protocol AnswerRouting: Sendable {
  func ask(_ request: AskRequest) async -> AsyncStream<AskEvent>
  func cancel() async
}
extension AnswerRouter: AnswerRouting {}

/// Manual delivery makes races and cancellation testable without sleeps.
public final class FakeDeepAnswer: DeepAnswering, Sendable {
  private let lock = NSLock()
  private nonisolated(unsafe) var outputs:
    [String: AsyncThrowingStream<AskEvent, Error>.Continuation] = [:]
  private nonisolated(unsafe) var recorded: [AskRequest] = []
  private nonisolated(unsafe) var ended: Set<String> = []
  private nonisolated(unsafe) var requestWaiters: [String: [CheckedContinuation<Void, Never>]] = [:]
  private nonisolated(unsafe) var terminationWaiters: [String: [CheckedContinuation<Void, Never>]] =
    [:]
  private let events: [AskEvent]?
  public init(events: [AskEvent]? = [.speech("Simulated recorded answer"), .done]) {
    self.events = events
  }
  public var requests: [AskRequest] { lock.withLock { recorded } }
  public var terminated: Set<String> { lock.withLock { ended } }
  public func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    let pair = AsyncThrowingStream<AskEvent, Error>.makeStream()
    let waiters = lock.withLock {
      recorded.append(request)
      outputs[request.id] = pair.continuation
      return requestWaiters.removeValue(forKey: request.id) ?? []
    }
    waiters.forEach { $0.resume() }
    pair.continuation.onTermination = { [weak self] _ in
      guard let self else { return }
      let waiters = self.lock.withLock {
        self.outputs[request.id] = nil
        self.ended.insert(request.id)
        return self.terminationWaiters.removeValue(forKey: request.id) ?? []
      }
      waiters.forEach { $0.resume() }
    }
    if let events {
      for event in events { pair.continuation.yield(event) }
      pair.continuation.finish()
    }
    return pair.stream
  }
  public func waitForRequest(_ id: String) async {
    await withCheckedContinuation { continuation in
      let ready = lock.withLock {
        if recorded.contains(where: { $0.id == id }) { return true }
        requestWaiters[id, default: []].append(continuation)
        return false
      }
      if ready { continuation.resume() }
    }
  }
  public func waitForTermination(_ id: String) async {
    await withCheckedContinuation { continuation in
      let ready = lock.withLock {
        if ended.contains(id) { return true }
        terminationWaiters[id, default: []].append(continuation)
        return false
      }
      if ready { continuation.resume() }
    }
  }
  public func send(_ event: AskEvent, to id: String) {
    let output = lock.withLock { outputs[id] }
    output?.yield(event)
    if event == .done { output?.finish() }
  }
}

public actor FakeRouter: AnswerRouting {
  private let events: [AskEvent]
  public private(set) var requests: [AskRequest] = []
  public init(events: [AskEvent] = [.speech("Simulated routed answer"), .done]) {
    self.events = events
  }
  public func ask(_ request: AskRequest) -> AsyncStream<AskEvent> {
    requests.append(request)
    return AsyncStream { c in
      events.forEach { c.yield($0) }
      c.finish()
    }
  }
  public func cancel() {}
}

#if canImport(SwiftUI)
  import SwiftUI
  public struct RouterDebugScreen: View {
    @State private var question = "How tall is this tree?"
    @State private var result = "Simulated planner"
    public init() {}
    public var body: some View {
      Form {
        TextField("Question", text: $question)
        Button("Plan with rules") {
          Task {
            do {
              let plan = try await RulePlanner().plan(question)
              result = "\(plan.subject): \(plan.tools.map(\.rawValue).joined(separator: ", "))"
            } catch { result = String(describing: error) }
          }
        }
        Text(result)
      }
    }
  }
  #Preview { RouterDebugScreen() }
#endif
