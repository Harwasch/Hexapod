#if DEBUG
  import Foundation

  /// Selected by the app's UI-test launch factory only, never by the provider menu.
  /// Three finite steps, a supplied PNG and a tiny valid PLY; no Metal, polling or sleeps.
  public actor FixturePhoneTrainingSession: PhoneTrainingSession {
    private let output: URL
    private let preview: Data
    private var active = true
    private var paused = false
    private var cancelled = false
    private var running = false
    private var closed = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var observers: [AsyncStream<TrainProgress>.Continuation] = []

    public init(outputDirectory: URL, previewPNG: Data) {
      output = outputDirectory
      preview = previewPNG
    }
    public func progress() -> AsyncStream<TrainProgress> {
      let (stream, continuation) = AsyncStream<TrainProgress>.makeStream()
      if closed { continuation.finish() } else { observers.append(continuation) }
      return stream
    }
    public func setActive(_ value: Bool) {
      active = value
      wake()
    }
    public func pause() { paused = true }
    public func resume() {
      paused = false
      wake()
    }
    public func cancel() {
      cancelled = true
      wake()
    }
    private func wake() {
      guard cancelled || closed || (active && !paused) else { return }
      let pending = waiters
      waiters.removeAll()
      for waiter in pending { waiter.resume() }
    }
    private func awaitActive() async throws {
      while !cancelled && !closed && (!active || paused) {
        await withCheckedContinuation { waiters.append($0) }
      }
      try Task.checkCancellation()
      guard !cancelled, !closed else { throw CancellationError() }
    }
    public func run() async throws -> PhoneTrainingResult {
      guard !running, !closed else { throw PrepError.invalid("Trainer running or closed") }
      running = true
      defer { running = false }
      return try await withTaskCancellationHandler {
        for step in 1...3 {
          try await awaitActive()
          let elapsed: Double = Double(step)
          let remaining: Double = Double(3 - step)
          let sample = TrainSample(
            step: step, wallSeconds: 1, elapsedSeconds: elapsed, completedMilliseconds: 1,
            gpuMilliseconds: nil, loss: nil, psnr: nil, splatCount: 1, etaSeconds: remaining,
            physicalFootprint: nil, availableMemory: 0, thermalState: 0)
          let progress = TrainProgress(sample: sample, message: "Simulated training", psnr: nil)
          for observer in observers { observer.yield(progress) }
          await Task.yield()
        }
        try await awaitActive()
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let ply = output.appendingPathComponent("trained.ply")
        let contents = """
          ply
          format ascii 1.0
          element vertex 1
          property float x
          property float y
          property float z
          property float nx
          property float ny
          property float nz
          property float f_dc_0
          property float f_dc_1
          property float f_dc_2
          property float opacity
          property float scale_0
          property float scale_1
          property float scale_2
          property float rot_0
          property float rot_1
          property float rot_2
          property float rot_3
          end_header
          0 0 -1 0 0 0 0 0 0 4 -3 -3 -3 1 0 0 0

          """
        try Data(contents.utf8).write(to: ply, options: .atomic)
        for observer in observers { observer.finish() }
        observers.removeAll()
        return PhoneTrainingResult(ply: ply, simulated: true)
      } onCancel: {
        Task { await self.cancel() }
      }
    }
    public func previewImage() -> Data? { preview }
    public func closeAndCleanup() throws {
      guard !running else { throw PrepError.invalid("Cancel and await run before close") }
      closed = true
      wake()
      for observer in observers { observer.finish() }
      observers.removeAll()
      // Result files belong to the capture, just as the native trainer's export does.
    }
  }
#endif
