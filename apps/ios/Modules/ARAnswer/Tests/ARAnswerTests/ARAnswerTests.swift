import EntitySchema
import Testing

@testable import ARAnswer

private func entity(_ x: Double, confidence: Double = 0.5) -> Entity {
  Entity(
    label: "tree", anchor: .init(world: .init(x: x, y: 0, z: -2)), extent: .init(x: 1, y: 2, z: 1),
    masks: [], confidence: confidence, attrs: [:], source: .init(module: "test", model: "depth"))
}
@Test func refinedEntitiesReplacePinsAndResetClearsQuestion() {
  var state = AnswerPresentation()
  state.receive(.entity(entity(0)))
  state.receive(.entity(entity(0.1, confidence: 0.9)))
  state.receive(.entity(entity(2)))
  state.receive(.speech("Two trees"))
  state.receive(.done)
  #expect(state.entities.count == 2)
  #expect(state.entities[0].confidence == 0.9)
  #expect(state.speech == "Two trees" && state.isComplete)
  state.reset()
  #expect(state.entities.isEmpty && state.speech.isEmpty && !state.isComplete)
}
@Test func viewportMaskAppliesCameraTransformAndRejectsInvalidData() throws {
  let mask = try #require(
    HighlightMask.project(
      width: 2, height: 2, pixels: [255, 0, 0, 0],
      viewportToImage: [1, 0, 0, 1, 0, 0], outputSize: 2))
  #expect(mask.pixels == [255, 0, 0, 0])
  let flipped = try #require(
    HighlightMask.project(
      width: 2, height: 2, pixels: mask.pixels,
      viewportToImage: [-1, 0, 0, 1, 1, 0], outputSize: 2))
  #expect(flipped.pixels == [0, 255, 0, 0])
  #expect(
    HighlightMask.project(width: 0, height: 2, pixels: [], viewportToImage: [1, 0, 0, 1, 0, 0])
      == nil)
  #expect(!HighlightMask(width: 2, height: 2, pixels: [255]).isValid)
}

@Test func progressStatusesDoNotReportAnAnswerFailure() {
  var presentation = AnswerPresentation()
  presentation.receive(.status("Looking"))
  presentation.receive(.status("Analyzing"))
  #expect(!presentation.isUnavailable)
  presentation.receive(.status("Deep answer unavailable"))
  #expect(presentation.isUnavailable)
  presentation.reset()
  #expect(!presentation.isUnavailable)
  presentation.receive(.status("Answer interrupted"))
  #expect(presentation.isUnavailable)
}

@Test func entityCardDisplaysOnlyKnownInvasiveAndTaxonFacts() {
  var subject = entity(0)
  #expect(!EntityCardFacts(subject).invasive && EntityCardFacts(subject).latinName == nil)
  subject.attrs["invasive"] = .object(["listed": .null, "source": .string("survey")])
  #expect(!EntityCardFacts(subject).invasive)
  subject.attrs["invasive"] = .object(["listed": .bool(true)])
  subject.attrs["scientificName"] = .string("Cortaderia jubata")
  #expect(EntityCardFacts(subject).invasive)
  #expect(EntityCardFacts(subject).latinName == "Cortaderia jubata")
}
