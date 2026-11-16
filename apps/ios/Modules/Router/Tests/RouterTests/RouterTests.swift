import EntitySchema
import Foundation
import Testing

@testable import Router

private func entity(_ x: Double = 0, label: String = "Tree", model: String = "device") -> Entity {
  Entity(
    label: label, anchor: .init(world: .init(x: x, y: 0, z: -2)), extent: .init(x: 1, y: 2, z: 1),
    masks: [], confidence: 0.8, attrs: [:], source: .init(module: "test", model: model))
}
@Test func rulePlanning() async throws {
  let planner = RulePlanner()
  #expect(try await planner.plan("How tall is this tree?").tools == [.measure])
  #expect(try await planner.plan("How many trees?").tools == [.count])
  #expect(try await planner.plan("What is this plant?").tools == [.identifyOnDevice])
  #expect(try await planner.plan("Show me the gate").subject == "gate")
  #expect(try await planner.plan("").tools == [.show])
  #expect(
    Set(try await planner.plan("Identify and measure oak").tools) == [.identifyOnDevice, .measure])
}
@Test func fastFirstDeepOverridesAndOnlyOneDone() {
  var merge = AnswerMerge()
  let fast = entity()
  let deep = entity(0.1, label: "tree", model: "deep")
  #expect(merge.receive(.entity(deep), from: .deep).isEmpty)
  #expect(merge.receive(.speech("Refined"), from: .deep).isEmpty)
  #expect(merge.receive(.done, from: .deep).isEmpty)
  #expect(merge.receive(.entity(fast), from: .fast) == [.entity(fast)])
  #expect(merge.receive(.entity(fast), from: .fast).isEmpty)
  #expect(merge.finish(.fast) == [.entity(deep), .speech("Refined"), .done])
  #expect(merge.entities == [deep])
  #expect(merge.finish(.deep).isEmpty)
  #expect(merge.receive(.speech("late"), from: .fast).isEmpty)
}
@Test func dedupeRequiresBothProximityAndLabel() {
  var merge = AnswerMerge()
  for e in [entity(), entity(2), entity(0.1, label: "rock")] {
    #expect(merge.receive(.entity(e), from: .fast) == [.entity(e)])
  }
  #expect(merge.entities.count == 3)
  #expect(merge.finish(.fast).isEmpty)
  #expect(merge.receive(.entity(entity(0.2, model: "cloud")), from: .deep).count == 1)
  #expect(merge.entities.count == 3)
  #expect(merge.finish(.deep) == [.done])
}

private struct Fast: FastAnswering {
  func answer(_ request: AskRequest, plan: AnswerPlan) async throws -> AsyncThrowingStream<
    AskEvent, Error
  > {
    AsyncThrowingStream { c in
      c.yield(.speech("fast " + request.question))
      c.yield(.done)
      c.finish()
    }
  }
}
private struct Deep: DeepAnswering {
  func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    AsyncThrowingStream { c in
      c.yield(.speech("deep " + request.question))
      c.finish()
    }
  }
}
@Test func routerMergesRealStreamsAndSynthesizesDoneOnEOF() async {
  let router = AnswerRouter(fast: Fast(), deep: Deep())
  let stream = await router.ask(.init(id: "1", question: "tree"))
  var events: [AskEvent] = []
  for await event in stream { events.append(event) }
  #expect(events == [.speech("fast tree"), .speech("deep tree"), .done])
}

/// Controlled continuations avoid timing sleeps in cancellation/concurrency tests.
private actor Gate {
  var waiting: CheckedContinuation<Void, Never>?
  var open = false
  func wait() async {
    if open { return }
    await withCheckedContinuation { waiting = $0 }
  }
  func release() {
    open = true
    waiting?.resume()
    waiting = nil
  }
}
private struct BlockedPlanner: Planning {
  let gate: Gate
  func plan(_ transcript: String) async throws -> AnswerPlan {
    if transcript == "old" { await gate.wait() }
    return AnswerPlan(subject: transcript, tools: [.show])
  }
}
private struct SignallingDeep: DeepAnswering {
  let gate: Gate
  func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    AsyncThrowingStream { c in
      Task {
        await gate.release()
        c.yield(.speech("deep"))
        c.finish()
      }
    }
  }
}
@Test func deepStartsWhileFastPlanningIsSuspendedAndNewQuestionCancelsOld() async {
  let planningGate = Gate()
  let started = Gate()
  let router = AnswerRouter(
    planner: BlockedPlanner(gate: planningGate), fast: Fast(),
    deep: SignallingDeep(gate: started))
  let old = await router.ask(.init(id: "old", question: "old"))
  await started.wait()  // Deep runs even while the planner is blocked.
  let new = await router.ask(.init(id: "new", question: "new"))
  await planningGate.release()
  var oldEvents: [AskEvent] = []
  var newEvents: [AskEvent] = []
  for await event in old { oldEvents.append(event) }
  for await event in new { newEvents.append(event) }
  #expect(oldEvents.isEmpty)
  #expect(newEvents == [.speech("fast new"), .speech("deep"), .done])
}
private struct BrokenFast: FastAnswering {
  struct Failure: Error {}
  func answer(_ request: AskRequest, plan: AnswerPlan) async throws -> AsyncThrowingStream<
    AskEvent, Error
  > {
    throw Failure()
  }
}
@Test func failureStillFinishesBothPathsOnce() async {
  let router = AnswerRouter(fast: BrokenFast(), deep: Deep())
  var events: [AskEvent] = []
  for await event in await router.ask(.init(id: "x", question: "x")) { events.append(event) }
  #expect(events == [.status("Fast answer unavailable"), .speech("deep x"), .done])
}

private struct Scene: SceneQuerying {
  var entities: [Entity]
  func inspect(_ plan: AnswerPlan) async throws -> [Entity] { entities }
}
@Test func deviceToolsMeasureCountIdentifyAndShowWithoutInventingLabels() async throws {
  var identified = entity()
  identified.attrs["identifiedOnDevice"] = .bool(true)
  let fast = DeviceFastAnswer(scene: Scene(entities: [identified]))
  let request = AskRequest(id: "1", question: "How tall is it?")
  var events: [AskEvent] = []
  for try await event in try await fast.answer(
    request, plan: .init(subject: "tree", tools: [.measure, .count, .identifyOnDevice]))
  {
    events.append(event)
  }
  #expect(events == [.entity(identified), .speech("I found 1. About 2.00 metres. Tree"), .done])
  let show = try await fast.answer(request, plan: .init(subject: "tree", tools: [.show]))
  var showEvents: [AskEvent] = []
  for try await event in show { showEvents.append(event) }
  #expect(showEvents == [.entity(identified), .done])
  let unknown = DeviceFastAnswer(scene: Scene(entities: [entity()]))
  var unknownEvents: [AskEvent] = []
  for try await event in try await unknown.answer(
    request, plan: .init(subject: "subject", tools: [.identifyOnDevice]))
  {
    unknownEvents.append(event)
  }
  #expect(unknownEvents.contains(.speech("I can't identify this yet.")))
}

private struct BrokenPlanner: Planning {
  struct Failure: Error {}
  func plan(_ transcript: String) async throws -> AnswerPlan { throw Failure() }
}
private actor PlanRecorder: FastAnswering {
  var received: AnswerPlan?
  func answer(_ request: AskRequest, plan: AnswerPlan) async throws -> AsyncThrowingStream<
    AskEvent, Error
  > {
    received = plan
    return AsyncThrowingStream { $0.finish() }
  }
}
@Test func plannerFailureUsesRulesAndPronounsSelectTheCurrentSubject() async {
  let fast = PlanRecorder()
  let router = AnswerRouter(planner: BrokenPlanner(), fast: fast)
  for await _ in await router.ask(.init(id: "1", question: "How tall is it?")) {}
  #expect(await fast.received == AnswerPlan(subject: "subject", tools: [.measure]))
}
private struct BrokenDeep: DeepAnswering {
  struct Failure: Error {}
  func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    AsyncThrowingStream { $0.finish(throwing: Failure()) }
  }
}
@Test func deepFailureKeepsTheFastAnswerAndCompletes() async {
  let router = AnswerRouter(fast: Fast(), deep: BrokenDeep())
  var events: [AskEvent] = []
  for await event in await router.ask(.init(id: "1", question: "tree")) { events.append(event) }
  #expect(events == [.speech("fast tree"), .status("Deep answer unavailable"), .done])
}
