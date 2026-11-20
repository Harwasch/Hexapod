import EntitySchema
import Foundation
import Perception
import Router
import Testing
import Voice

@testable import ARAnswer

private struct Segmentation: InstanceSegmenting {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    SubjectMask(width: 4, height: 4, pixels: [UInt8](repeating: 255, count: 16))
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] {
    [try await segment(image, prompt: .point(.init(x: 0.5, y: 0.5)))]
  }
}
private struct Identification: ImageIdentifying {
  func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel? {
    DeviceLabel(label: "panel", confidence: 0.9)
  }
}
private struct SceneAdapter: SceneQuerying {
  let scene: PerceptionScene
  func inspect(_ plan: AnswerPlan) async throws -> [Entity] {
    try await scene.inspect(subject: plan.subject, count: plan.tools.contains(.count))
  }
}
private struct Refinement: DeepAnswering {
  let entity: Entity
  func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    AsyncThrowingStream { c in
      c.yield(.entity(entity))
      c.yield(.speech("Refined measurement"))
      c.yield(.done)
      c.finish()
    }
  }
}
@Test func transcriptToPerceptionToMergedPresentation() async throws {
  var transcript = TranscriptAccumulator()
  transcript.receive(.init(text: "How", isFinal: false))
  transcript.receive(.init(text: "How tall", isFinal: true))
  transcript.receive(.init(text: "is this panel", isFinal: true))
  let scene = PerceptionScene(
    segmenter: Segmentation(), instances: Segmentation(), identifier: Identification())
  let camera = CameraGeometry(
    fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4,
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
  await scene.update(
    FrameSnapshot(
      image: .init(width: 4, height: 4, rgba: [UInt8](repeating: 255, count: 64)),
      depth: .init(width: 4, height: 4, metres: [Float](repeating: 2, count: 16)), camera: camera),
    viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  let refined = Entity(
    label: "panel", anchor: .init(world: .init(x: 0.1, y: 0, z: -2)),
    extent: .init(x: 1.5, y: 1.6, z: 0), masks: [], confidence: 0.95, attrs: [:],
    source: .init(module: "deep", model: "fixture"))
  let router = AnswerRouter(
    fast: DeviceFastAnswer(scene: SceneAdapter(scene: scene)), deep: Refinement(entity: refined))
  var presentation = AnswerPresentation()
  var events: [AskEvent] = []
  for await event in await router.ask(.init(id: "test", question: transcript.text)) {
    presentation.receive(event)
    events.append(event)
  }
  #expect(events.filter { $0 == .done }.count == 1)
  let speech = events.compactMap { event -> String? in
    if case .speech(let text) = event { return text }
    return nil
  }
  #expect(speech == ["About 1.50 metres.", "Refined measurement"])
  #expect(presentation.entities == [refined])
  #expect(presentation.isComplete)
}
