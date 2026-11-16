import EntitySchema
import Testing

@testable import ARAnswer

@Test @MainActor func overlaySinkRecordsPresentationCommandsWithoutRenderingOrAudio() {
  let sink: FakeOverlaySink = FakeOverlaySink()
  let entity = Entity(
    label: "Simulated tree", anchor: .init(world: .init(x: 1, y: 1, z: -2)),
    extent: .init(x: 1, y: 2, z: 1), masks: [], confidence: 1, attrs: [:],
    source: .init(module: "test", model: "fixture"))
  let overlay: any AnswerOverlaySink = sink
  overlay.update(entities: [entity])
  overlay.setMask(.init(width: 2, height: 1, pixels: [255, 0]))
  overlay.speak("Simulated tree")
  overlay.stopSpeaking()
  overlay.remove()
  #expect(sink.draws == [[entity]])
  #expect(sink.masks.first??.pixels == [255, 0])
  #expect(sink.utterances == ["Simulated tree"])
  #expect(sink.stopCount == 1 && sink.removeCount == 1)
}
