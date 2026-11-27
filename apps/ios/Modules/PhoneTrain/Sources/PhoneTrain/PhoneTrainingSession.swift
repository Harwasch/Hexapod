import Foundation

/// UI-facing seam. The native PhoneTrainer keeps its existing API and ownership rules.
public protocol PhoneTrainingSession: Sendable {
  func progress() async -> AsyncStream<TrainProgress>
  func setActive(_ active: Bool) async
  func pause() async
  func resume() async
  func cancel() async
  func run() async throws -> PhoneTrainingResult
  func previewImage() async throws -> Data?
  func closeAndCleanup() async throws
}

public struct PhoneTrainingResult: Equatable, Sendable {
  public let ply: URL
  public let simulated: Bool
  public init(ply: URL, simulated: Bool) {
    self.ply = ply
    self.simulated = simulated
  }
}

#if canImport(Msplat) && os(iOS)
  /// Delegates lifecycle and training to the real actor; no native behavior is overridden.
  public struct DevicePhoneTrainingSession: PhoneTrainingSession {
    public let trainer: PhoneTrainer
    public init(trainer: PhoneTrainer) { self.trainer = trainer }
    public func progress() async -> AsyncStream<TrainProgress> { await trainer.progress() }
    public func setActive(_ active: Bool) async { await trainer.setActive(active) }
    public func pause() async { await trainer.pause() }
    public func resume() async { await trainer.resume() }
    public func cancel() async { await trainer.cancel() }
    public func run() async throws -> PhoneTrainingResult {
      let outcome = try await trainer.run()
      switch outcome {
      case .finished:
        let ply = try await trainer.exportPLY()
        return PhoneTrainingResult(ply: ply, simulated: false)
      case .cancelled: throw CancellationError()
      case .critical: throw PrepError.invalid("Training paused: critical temperature")
      }
    }
    // Native preview consumers continue to await trainer.submitPreview(...).waitUntilReady().
    public func previewImage() async throws -> Data? { nil }
    public func closeAndCleanup() async throws { try await trainer.closeAndCleanup() }
  }
#endif
