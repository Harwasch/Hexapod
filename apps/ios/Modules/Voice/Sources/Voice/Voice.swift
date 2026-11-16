import Foundation

public struct Transcript: Equatable, Sendable {
  public var text: String
  public var isFinal: Bool
  public init(text: String, isFinal: Bool) {
    self.text = text
    self.isFinal = isFinal
  }
}

public enum VoiceError: Error, Sendable { case permissionDenied, unavailable, interrupted }

/// Main-actor ownership includes microphone/session lifecycle and injected UI fakes.
@MainActor public protocol VoiceInput: AnyObject {
  func start() async throws -> AsyncThrowingStream<Transcript, Error>
  /// Ends input and drains final recognition results.
  func stop() async
  /// Discards the current utterance immediately.
  func cancel()
}

/// Speech results replace their volatile segment; final segments accumulate once.
public struct TranscriptAccumulator: Sendable {
  private var committed: [String] = []
  public private(set) var text = ""
  public init() {}
  public mutating func receive(_ result: Transcript) {
    let value = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
    if result.isFinal && !value.isEmpty { committed.append(value) }
    text = (committed + (result.isFinal || value.isEmpty ? [] : [value])).joined(separator: " ")
  }
}
