#if canImport(Msplat) && os(iOS)
  import Darwin
  import os
  import Foundation
  import Msplat
  import PhoneTrainPrep

  public enum TrainOutcome: String, Codable, Sendable {
    case finished, cancelled, critical
  }

  /// Sole owner of a session and step loop. msplat's global runtime actor executes
  /// all native calls, off MainActor. One PhoneTrainer may be running process-wide.
  /// The product screen consumes progress(), setActive(), pause()/resume() and
  /// exportPLY(); it must forward scene inactivity immediately. No background work.
  public actor PhoneTrainer {
    private let dataset: PreparedDataset
    private let capture: URL
    private let output: URL
    private let prep: PrepConfig
    private let config: PhoneTrainingConfig
    private var session: MsplatSession?
    private var running = false
    private var control = TrainingControl()
    private var auxiliaryBusy = false
    private var thermalObserver: (any NSObjectProtocol)?
    private var closed = false
    private var samples: [TrainSample] = []
    private var continuations: [UUID: AsyncStream<TrainProgress>.Continuation] = [:]
    private var lastPSNR: Float?
    private var previewPose: CameraPose?
    private var previewObservers: [UUID: AsyncStream<MetalPreviewSurface>.Continuation] = [:]

    public func setPreviewPose(_ pose: CameraPose) { previewPose = pose }
    public func previews() -> AsyncStream<MetalPreviewSurface> {
      let id = UUID()
      let (stream, continuation) = AsyncStream<MetalPreviewSurface>.makeStream(bufferingPolicy: .bufferingNewest(1))
      previewObservers[id] = continuation
      continuation.onTermination = { [weak self] _ in Task { await self?.removePreview(id) } }
      return stream
    }
    private func removePreview(_ id: UUID) { previewObservers.removeValue(forKey: id) }

    public init(
      dataset: PreparedDataset, capture: URL, outputDirectory: URL,
      prep: PrepConfig, config: PhoneTrainingConfig
    ) {
      self.dataset = dataset
      self.capture = capture
      output = outputDirectory
      self.prep = prep
      self.config = config
    }

    /// Recreate the same deterministic dataset and restore native optimizer state.
    /// Call run(loadCheckpoint: true) on the returned actor, then closeAndCleanup().
    public static func restoring(from directory: URL) async throws -> PhoneTrainer {
      let metadata = try JSONDecoder().decode(
        TrainingCheckpoint.self,
        from: Data(contentsOf: directory.appendingPathComponent("checkpoint.json")))
      let bundle = URL(fileURLWithPath: metadata.capturePath, isDirectory: true)
      let dataset = try await CaptureDatasetBuilder().prepare(bundle: bundle, config: metadata.prep)
      return PhoneTrainer(
        dataset: dataset, capture: bundle, outputDirectory: directory,
        prep: metadata.prep, config: metadata.training)
    }

    public func closeAndCleanup() async throws {
      try await close()
      try dataset.cleanup()
    }

    public func progress() -> AsyncStream<TrainProgress> {
      let id = UUID()
      let (stream, continuation) = AsyncStream<TrainProgress>.makeStream(
        bufferingPolicy: .bufferingNewest(2))
      continuations[id] = continuation
      continuation.onTermination = { [weak self] _ in Task { await self?.removeObserver(id) } }
      return stream
    }
    private func removeObserver(_ id: UUID) { continuations.removeValue(forKey: id) }
    private func emit(_ message: String, sample: TrainSample? = nil) {
      for continuation in continuations.values {
        continuation.yield(TrainProgress(sample: sample, message: message, psnr: lastPSNR))
      }
    }
    public func setActive(_ value: Bool) async {
      control.active = value
      // A final-step evaluation/export can leave the loop idle. Preserve that
      // completed optimizer state on inactivity as well as at loop boundaries.
      if !value, !running, !auxiliaryBusy, !closed, let session {
        auxiliaryBusy = true
        try? await checkpoint(session)
        auxiliaryBusy = false
      }
    }
    public func needsCooling() -> Bool {
      control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical
    }
    public func pause() { control.manuallyPaused = true }
    public func resume() {
      control.manuallyPaused = false
      if ProcessInfo.processInfo.thermalState != .critical { control.criticalObserved = false }
    }
    public func cancel() { control.cancelled = true }
    public func criticalThermalObserved() { control.criticalObserved = true }
    private func observeThermal() {
      guard thermalObserver == nil else { return }
      thermalObserver = NotificationCenter.default.addObserver(
        forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
      ) { [weak self] _ in
        if ProcessInfo.processInfo.thermalState == .critical {
          Task { await self?.criticalThermalObserved() }
        }
      }
    }
    public func recordedSamples() -> [TrainSample] { samples }

    /// Uses the code-derived TrainingPlan estimate and current process allowance,
    /// including recommended headroom. Rejects rather than silently changing a run.
    public static func estimatedPeak(dataset: PreparedDataset, config: PhoneTrainingConfig) throws
      -> Int64
    {
      try config.validate()
      let stages = try config.resolutionStages().map { stage in
        try TrainingResolutionStage(
          iterations: stage.iterations, downscaleFactor: stage.downscaleFactor)
      }
      return try TrainingPlan(
        inputDimensions: TrainingImageDimensions(width: dataset.width, height: dataset.height),
        inputDecodeScale: 1, iterationBudget: config.iterations, stages: stages,
        targetSHDegree: config.shDegree, maximumGaussianCount: config.maxGaussians
      ).estimatedPeakMemory
    }

    public static func recommendedPreset(dataset: PreparedDataset) throws -> PhoneTrainingConfig {
      let available = Double(os_proc_available_memory())
      for preset in [PhoneTrainingConfig.best, .quick] {
        if Double(try estimatedPeak(dataset: dataset, config: preset)) < available
          * preset.memoryBudgetFraction
        {
          return preset
        }
      }
      throw PrepError.invalid("Not enough available memory for Quick")
    }

    /// Reuse the same prep/config recorded in checkpoint.json after relaunch.
    /// The native checkpoint restores optimization state; JPEGs can be rebuilt.
    public func run(loadCheckpoint: Bool = false) async throws -> TrainOutcome {
      guard !running, !auxiliaryBusy, !closed else {
        throw PrepError.invalid("Trainer already running or closed")
      }
      running = true
      observeThermal()
      defer { running = false }
      let start = ProcessInfo.processInfo.systemUptime
      do {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        while session == nil && (!control.active || control.manuallyPaused) {
          try Task.checkCancellation()
          if control.cancelled { return .cancelled }
          if control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical {
            return .critical
          }
          try await Task.sleep(for: .milliseconds(100))
        }
        if control.cancelled { return .cancelled }
        if session == nil
          && (control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical)
        {
          emit("Critical temperature before training; no session started.")
          return .critical
        }
        if session == nil {
          let estimate = try Self.estimatedPeak(dataset: dataset, config: config)
          guard Double(estimate) < Double(os_proc_available_memory()) * config.memoryBudgetFraction
          else {
            throw PrepError.invalid(
              "TrainingPlan estimate \(estimate) exceeds available memory budget")
          }
          var native = TrainingConfig()
          native.iterations = config.iterations
          native.shDegree = config.shDegree
          native.shDegreeInterval = max(1, config.iterations / (config.shDegree + 1))
          native.numDownscales = config.numDownscales
          native.resolutionSchedule = config.resolutionSchedule
          native.ssimWeight = config.ssimWeight
          native.keepCrs = true
          // Densification must occur within the short benchmark budget.
          native.warmupLength = config.warmupLength
          let count = min(dataset.descriptor.points.count, config.maxGaussians / 2)
          var xyz: [Float] = []
          var rgb: [UInt8] = []
          for index in 0..<count {
            let offset = (index * dataset.descriptor.points.count / count) * 3
            xyz.append(contentsOf: dataset.descriptor.points.xyz[offset..<offset + 3])
            rgb.append(contentsOf: dataset.descriptor.points.rgb[offset..<offset + 3])
          }
          let initialPoints = try DatasetSparsePointSet(xyz: xyz, rgb: rgb)
          let descriptor = try DatasetDescriptor(
            frames: dataset.descriptor.frames, points: initialPoints)
          session = try await MsplatSession(
            dataset: descriptor,
            options: DatasetOptions(evalMode: true, testEvery: Int32(prep.holdoutEvery)),
            config: native, maximumGaussianCount: config.maxGaussians)
          if loadCheckpoint {
            let metadata = try JSONDecoder().decode(
              TrainingCheckpoint.self,
              from: Data(contentsOf: output.appendingPathComponent("checkpoint.json")))
            guard metadata.capturePath == capture.path, metadata.training == config,
              metadata.prep == prep
            else {
              throw PrepError.invalid("Checkpoint capture/config mismatch")
            }
            _ = try await session?.loadCheckpoint(
              from: output.appendingPathComponent("checkpoint.bin"))
          }
        }
        guard let session else { throw PrepError.invalid("No session") }
        var etaEstimator = TrainingETA()
        var checkpointedPause = false
        while try await session.iteration < Int(config.iterations) {
          try Task.checkCancellation()
          if control.cancelled {
            emit("Cancelled")
            return .cancelled
          }
          let thermal = ProcessInfo.processInfo.thermalState
          if control.decision(thermal: thermal.rawValue) == .critical {
            try await checkpoint(session)
            emit(
              "Paused: critical temperature. Checkpoint saved; let the phone cool before resuming.")
            return .critical
          }
          if !control.active || control.manuallyPaused {
            if !checkpointedPause {
              try await checkpoint(session)
              emit("Paused. Checkpoint saved.")
              checkpointedPause = true
            }
            try await Task.sleep(for: .milliseconds(100))
            continue
          }
          checkpointedPause = false
          let activeStepStart = ProcessInfo.processInfo.systemUptime
          if thermal == .serious {
            try await Task.sleep(for: .milliseconds(config.seriousSleepMilliseconds))
            // Sleep is a reentrancy point: check lifecycle before submitting again.
            if !control.active || control.manuallyPaused || control.cancelled { continue }
            if control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical {
              continue
            }
          }
          let stepStart = ProcessInfo.processInfo.systemUptime
          let submitted = try await session.step()
          // Metrics are query-only. Wait for this logical GPU step before another
          // submission, as the reference benchmark collects completed iterations.
          var metrics = try await session.trainingMetrics()
          while metrics.completed?.iteration != submitted.iteration {
            try Task.checkCancellation()
            if ProcessInfo.processInfo.systemUptime - stepStart > config.gpuStepTimeoutSeconds {
              throw PrepError.invalid("GPU step completion timed out")
            }
            if metrics.lastFailedIteration != nil { throw PrepError.invalid("GPU step failed") }
            try await Task.sleep(for: .milliseconds(1))
            metrics = try await session.trainingMetrics()
          }
          guard let completed = metrics.completed else {
            throw PrepError.invalid("No completed step")
          }
          let memory = try await session.memoryMetrics()
          // Each argument is computed with an explicit type first: a single initializer
          // full of mixed-type arithmetic is what times out Xcode's type checker.
          let now: Double = ProcessInfo.processInfo.systemUptime
          let elapsed: Double = now - start
          let stepsLeft: Int = Int(config.iterations) - submitted.iteration
          let eta: Double = etaEstimator.record(stepSeconds: now - activeStepStart, remaining: stepsLeft)
          var loss: Float? = nil
          if let value = completed.loss, value.isFinite { loss = value }
          var footprint: UInt64? = ProcessMemory.physicalFootprint()
          if footprint == nil { footprint = memory.processPhysicalFootprintBytes }
          let availableNow: UInt64 = UInt64(os_proc_available_memory())
          let thermalNow: Int = ProcessInfo.processInfo.thermalState.rawValue
          let sample = TrainSample(
            step: submitted.iteration, wallSeconds: now - stepStart, elapsedSeconds: elapsed,
            completedMilliseconds: completed.endToEndMs, gpuMilliseconds: completed.gpuExecutionMs,
            loss: loss, psnr: lastPSNR, splatCount: completed.splatCount, etaSeconds: eta,
            physicalFootprint: footprint, availableMemory: availableNow, thermalState: thermalNow)
          samples.append(sample)
          emit("Training", sample: sample)
          // Product consumers request poses, never native work. The completed-step
          // boundary serializes preview/evaluation with optimization and checkpoints.
          let previewRequested = previewPose != nil
          let thermalBoundary = control.criticalObserved ? 3 : ProcessInfo.processInfo.thermalState.rawValue
          if TrainingSnapshotPolicy.preview(step: submitted.iteration, requested: previewRequested,
            active: control.active, paused: control.manuallyPaused,
            cancelled: control.cancelled, thermal: thermalBoundary), let pose = previewPose {
            let submission = try await session.submitPreview(pose: pose)
            let surface = try await submission.waitUntilReady()
            for observer in previewObservers.values { observer.yield(surface) }
          }
          // Preview completion is a reentrancy point: check lifecycle again.
          let evaluationThermal = control.criticalObserved ? 3 : ProcessInfo.processInfo.thermalState.rawValue
          if TrainingSnapshotPolicy.evaluate(step: submitted.iteration, requested: previewRequested,
            active: control.active, paused: control.manuallyPaused,
            cancelled: control.cancelled, thermal: evaluationThermal) {
            let evaluation = try await session.evaluate()
            if evaluation.psnr.isFinite { lastPSNR = evaluation.psnr }
            emit("Training", sample: sample)
          }
        }
        // Final-step thermal/lifecycle checks precede evaluation/export too.
        if control.decision(thermal: ProcessInfo.processInfo.thermalState.rawValue) == .critical {
          try await checkpoint(session)
          emit("Paused: critical temperature. Checkpoint saved.")
          return .critical
        }
        if !control.active || control.manuallyPaused {
          try await checkpoint(session)
          emit("Paused. Checkpoint saved.")
        }
        while !control.active || control.manuallyPaused {
          try Task.checkCancellation()
          if control.cancelled { return .cancelled }
          try await Task.sleep(for: .milliseconds(100))
        }
        if control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical {
          try await checkpoint(session)
          emit("Paused: critical temperature. Checkpoint saved.")
          return .critical
        }
        return control.cancelled ? .cancelled : .finished
      } catch {
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        thermalObserver = nil
        if let session { try? await session.close() }
        session = nil
        throw error
      }
    }

    private func checkpoint(_ session: MsplatSession) async throws {
      let temporary = output.appendingPathComponent(".checkpoint-\(UUID().uuidString).bin")
      defer { try? FileManager.default.removeItem(at: temporary) }
      try await session.saveCheckpoint(to: temporary)
      let destination = output.appendingPathComponent("checkpoint.bin")
      let status = temporary.path.withCString { source in
        destination.path.withCString { target in Darwin.rename(source, target) }
      }
      guard status == 0 else {
        throw PrepError.invalid("Checkpoint atomic rename failed: \(errno)")
      }
      let metadata = TrainingCheckpoint(
        capturePath: capture.path, prep: prep, training: config,
        step: try await session.iteration)
      try JSONEncoder().encode(metadata).write(
        to: output.appendingPathComponent("checkpoint.json"), options: .atomic)
    }

    public func evaluate() async throws -> EvalMetrics {
      let session = try await acquireAuxiliary()
      defer { auxiliaryBusy = false }
      try await waitForAuxiliaryBoundary(session)
      let metrics = try await session.evaluate()
      if control.criticalObserved || ProcessInfo.processInfo.thermalState == .critical {
        try await checkpoint(session)
        throw PrepError.invalid("Critical temperature during evaluation; checkpoint saved")
      }
      if !control.active { try await checkpoint(session) }
      lastPSNR = metrics.psnr
      emit("Evaluation complete: PSNR \(metrics.psnr), SSIM \(metrics.ssim)")
      return metrics
    }

    /// PLY remains in ARKit world metres (keepCrs=true). MsplatSession v3.0.0
    /// has no public SPZ export; do not mix in deprecated GaussianTrainer handles.
    public func exportPLY() async throws -> URL {
      let session = try await acquireAuxiliary()
      defer { auxiliaryBusy = false }
      try await waitForAuxiliaryBoundary(session)
      let url = output.appendingPathComponent("trained.ply")
      try await session.exportPLY(to: url)
      if !control.active { try await checkpoint(session) }
      return url
    }

    private func acquireAuxiliary() async throws -> MsplatSession {
      while auxiliaryBusy {
        try Task.checkCancellation()
        try await Task.sleep(for: .milliseconds(10))
      }
      guard !running, !closed, let session else { throw PrepError.invalid("Session unavailable") }
      auxiliaryBusy = true
      return session
    }

    /// Uses the same portable control decisions as the training loop. It keeps
    /// auxiliary ownership while waiting, so no second evaluator can enter.
    private func waitForAuxiliaryBoundary(_ session: MsplatSession) async throws {
      var checkpointed = false
      while true {
        let decision = control.decision(thermal: ProcessInfo.processInfo.thermalState.rawValue)
        switch decision {
        case .cancel: throw CancellationError()
        case .critical:
          try await checkpoint(session)
          throw PrepError.invalid("Cooling required; checkpoint saved")
        case .pause:
          if !checkpointed { try await checkpoint(session); checkpointed = true }
          try await Task.sleep(for: .milliseconds(100))
        case .step, .throttle: return
        }
      }
    }

    /// Later live-forming preview seam: no texture or session escapes ownership.
    /// Consumers await submission.waitUntilReady() before displaying the surface.
    public func submitPreview(pose: CameraPose, referenceCamera: Int = 0) async throws
      -> MetalPreviewSubmission
    {
      guard !running, !auxiliaryBusy, !closed, let session else {
        throw PrepError.invalid("Preview unavailable")
      }
      auxiliaryBusy = true
      defer { auxiliaryBusy = false }
      return try await session.submitPreview(pose: pose, referenceCamera: referenceCamera)
    }

    public func close() async throws {
      while auxiliaryBusy { try await Task.sleep(for: .milliseconds(10)) }
      guard !running else {
        throw PrepError.invalid("Cancel and await operations before close")
      }
      if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
      thermalObserver = nil
      closed = true
      defer {
        for c in continuations.values { c.finish() }
        continuations.removeAll()
        for observer in previewObservers.values { observer.finish() }
        previewObservers.removeAll()
      }
      let previous = session
      session = nil
      if let previous { try await previous.close() }
    }
  }
#endif
