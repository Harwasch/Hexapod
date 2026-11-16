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

@MainActor private final class FakeVoice: VoiceInput {
  private var continuation: AsyncThrowingStream<Transcript, Error>.Continuation?
  func start() async throws -> AsyncThrowingStream<Transcript, Error> {
    let pair = AsyncThrowingStream<Transcript, Error>.makeStream()
    continuation = pair.continuation
    continuation?.yield(.init(text: "tree", isFinal: false))
    return pair.stream
  }
  func stop() async {
    continuation?.yield(.init(text: "tree", isFinal: true))
    continuation?.finish()
  }
  func cancel() { continuation?.finish() }
}

@Test @MainActor func injectableVoiceDrainsFinalResultOnRelease() async throws {
  let voice: any VoiceInput = FakeVoice()
  let stream = try await voice.start()
  await voice.stop()
  var results: [Transcript] = []
  for try await result in stream { results.append(result) }
  #expect(results == [.init(text: "tree", isFinal: false), .init(text: "tree", isFinal: true)])
}
