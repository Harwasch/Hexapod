import Testing

@testable import Voice

@Test func volatileResultsReplaceAndFinalSegmentsAccumulate() {
  var accumulator = TranscriptAccumulator()
  accumulator.receive(.init(text: "how", isFinal: false))
  accumulator.receive(.init(text: "how tall", isFinal: false))
  #expect(accumulator.text == "how tall")
  accumulator.receive(.init(text: "how tall", isFinal: true))
  accumulator.receive(.init(text: "is this", isFinal: false))
  #expect(accumulator.text == "how tall is this")
  accumulator.receive(.init(text: "is this tree", isFinal: true))
  accumulator.receive(.init(text: "  ", isFinal: true))
  #expect(accumulator.text == "how tall is this tree")
}

@Test @MainActor func injectableVoiceDrainsFinalResultOnRelease() async throws {
  let voice: any VoiceInput = FakeVoice(script: [
    .init(text: "tree", isFinal: false), .init(text: "tree", isFinal: true),
  ])
  let stream = try await voice.start()
  await voice.stop()
  var results: [Transcript] = []
  for try await result in stream { results.append(result) }
  #expect(results == [.init(text: "tree", isFinal: false), .init(text: "tree", isFinal: true)])
}

@Test @MainActor func cancellingFakeDiscardsFinalAndRestartIsIndependent() async throws {
  let voice = FakeVoice()
  let old = try await voice.start()
  voice.cancel()
  let current = try await voice.start()
  await voice.stop()
  var oldSteps: [Transcript] = []
  for try await step in old { oldSteps.append(step) }
  var steps: [Transcript] = []
  for try await step in current { steps.append(step) }
  #expect(oldSteps.allSatisfy { !$0.isFinal })
  #expect(steps == FakeVoice.treeScript)
}
