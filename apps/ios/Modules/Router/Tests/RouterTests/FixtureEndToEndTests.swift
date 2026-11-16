import EntitySchema
import Foundation
import Perception
import Testing
import Voice

@testable import Router

private struct FixtureScene: SceneQuerying {
  let scene: FakePerception
  func inspect(_ plan: AnswerPlan) async throws -> [Entity] {
    try await scene.inspect(subject: plan.subject, count: plan.tools.contains(.count))
  }
}
private func replayScene() async throws -> FakePerception {
  let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/bundles/yard-small")
  let replay = try FixtureReplay(directory: root)
  let scene = FakePerception()
  await scene.update(
    try replay.snapshot(at: 2), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  return scene
}
@Test @MainActor func scriptedVoicePlansAndMergesFixtureFastPathWithRecordedDeepStream()
  async throws
{
  let voice = FakeVoice()
  let stream = try await voice.start()
  await voice.stop()
  var transcript = TranscriptAccumulator()
  for try await step in stream { transcript.receive(step) }
  #expect(transcript.text == "How tall is this tree")
  let plan = AnswerPlan(subject: "tree", tools: [.measure])
  let scene = try await replayScene()
  let fastEntity = try #require(try await scene.inspect(subject: "tree", count: false).first)
  var refined = fastEntity
  refined.confidence = 0.99
  refined.source = .init(module: "recorded-deep", model: "fixture")
  let deep = FakeDeepAnswer(events: [
    .entity(fastEntity), .entity(refined), .entity(refined), .speech("Recorded refinement"), .done,
    .done,
  ])
  let router: any AnswerRouting = AnswerRouter(
    planner: FakePlanner(plans: [transcript.text: plan]),
    fast: DeviceFastAnswer(scene: FixtureScene(scene: scene)), deep: deep)
  let request = AskRequest(
    id: "fixture", question: transcript.text, scope: .property, siteId: "synthetic-yard",
    deviceEntities: [fastEntity])
  var events: [AskEvent] = []
  for await event in await router.ask(request) { events.append(event) }
  let speech = String(
    format: "About %.2f metres.", locale: Locale(identifier: "en_US_POSIX"), fastEntity.extent.y)
  #expect(
    events == [
      .entity(fastEntity), .speech(speech), .entity(refined), .speech("Recorded refinement"), .done,
    ])
  #expect(deep.requests == [request])
  #expect(events.filter { $0 == .done }.count == 1)
}

private actor QuestionGate {
  var open = false
  var entered = false
  var entry: CheckedContinuation<Void, Never>?
  var pending: CheckedContinuation<Void, Never>?
  func hold() async {
    entered = true
    entry?.resume()
    entry = nil
    if !open { await withCheckedContinuation { pending = $0 } }
  }
  func waitForEntry() async { if !entered { await withCheckedContinuation { entry = $0 } } }
  func release() {
    open = true
    pending?.resume()
    pending = nil
  }
}
private struct GatedPlanner: Planning {
  let gate: QuestionGate
  func plan(_ transcript: String) async throws -> AnswerPlan {
    if transcript == "old" { await gate.hold() }
    try Task.checkCancellation()
    return .init(subject: "tree", tools: [.measure])
  }
}
@Test func newQuestionCancelsBothPathsAndDiscardsPendingRecordedRefinement() async throws {
  let scene = try await replayScene()
  let gate = QuestionGate()
  let deep = FakeDeepAnswer(events: nil)
  let router = AnswerRouter(
    planner: GatedPlanner(gate: gate), fast: DeviceFastAnswer(scene: FixtureScene(scene: scene)),
    deep: deep)
  let old = await router.ask(.init(id: "old", question: "old"))
  await gate.waitForEntry()
  await deep.waitForRequest("old")
  deep.send(.speech("stale refinement"), to: "old")
  let new = await router.ask(.init(id: "new", question: "How tall is this tree"))
  await gate.release()
  await deep.waitForTermination("old")
  await deep.waitForRequest("new")
  deep.send(.speech("current refinement"), to: "new")
  deep.send(.done, to: "new")
  var oldEvents: [AskEvent] = []
  for await event in old { oldEvents.append(event) }
  var newEvents: [AskEvent] = []
  for await event in new { newEvents.append(event) }
  #expect(oldEvents.isEmpty)
  #expect(deep.terminated.contains("old"))
  #expect(newEvents.filter { $0 == .done }.count == 1)
  #expect(!newEvents.contains(.speech("stale refinement")))
  #expect(newEvents.suffix(2) == [.speech("current refinement"), .done])
}
