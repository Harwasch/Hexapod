import Foundation

/// Deterministic push-to-talk replay. Volatile steps arrive on start; release drains
/// the remaining script, including final steps. No microphone, timer or permission.
@MainActor public final class FakeVoice: VoiceInput {
  public static let treeScript: [Transcript] = [
    .init(text: "How", isFinal: false),
    .init(text: "How tall is this", isFinal: false),
    .init(text: "How tall is this tree", isFinal: true),
  ]
  private let script: [Transcript]
  private var output: AsyncThrowingStream<Transcript, Error>.Continuation?
  private var remaining: [Transcript] = []
  public init(script: [Transcript] = FakeVoice.treeScript) { self.script = script }
  public func start() async throws -> AsyncThrowingStream<Transcript, Error> {
    cancel()
    let pair = AsyncThrowingStream<Transcript, Error>.makeStream()
    output = pair.continuation
    let split = script.firstIndex(where: \.isFinal) ?? script.endIndex
    for step in script[..<split] { output?.yield(step) }
    remaining = Array(script[split...])
    return pair.stream
  }
  public func stop() async {
    for step in remaining { output?.yield(step) }
    remaining = []
    output?.finish()
    output = nil
  }
  public func cancel() {
    remaining = []
    output?.finish()
    output = nil
  }
}

#if canImport(SwiftUI)
  import SwiftUI
  public struct VoiceDebugScreen: View {
    @State private var text = "Simulated voice: tap Replay"
    public init() {}
    public var body: some View {
      VStack {
        Text(text)
        Button("Replay volatile → final") {
          Task { @MainActor in
            let voice = FakeVoice()
            var accumulator = TranscriptAccumulator()
            do {
              let stream = try await voice.start()
              await voice.stop()
              var lines: [String] = []
              for try await step in stream {
                accumulator.receive(step)
                lines.append("\(step.isFinal ? "Final" : "Volatile"): \(accumulator.text)")
              }
              text = "Simulated\n" + lines.joined(separator: "\n")
            } catch { text = String(describing: error) }
          }
        }
      }.padding()
    }
  }
  #Preview { VoiceDebugScreen() }
#endif
