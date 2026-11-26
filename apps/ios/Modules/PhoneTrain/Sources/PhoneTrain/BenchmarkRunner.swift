#if canImport(Msplat) && os(iOS)
  import Darwin
  import os
  import Foundation
  import PhoneTrainPrep

  /// Hidden measurement harness. Forwards lifecycle to the one current trainer;
  /// callers must await cancellation before discarding this actor. No network.
  public actor PhoneTrainBenchmark {
    private var preparation: Task<PreparedDataset, Error>?
    private var trainer: PhoneTrainer?
    private var active = true
    private var cancelled = false
    private var running = false
    private var criticalObserved = false
    private var thermalObserver: (any NSObjectProtocol)?
    private var monitor: Task<Void, Never>?
    private var currentReport: BenchmarkReport?
    public init() {}
    public func report() -> BenchmarkReport? { currentReport }
    public func setActive(_ value: Bool) async {
      active = value
      await trainer?.setActive(value)
    }
    public func cancel() async {
      cancelled = true
      preparation?.cancel()
      await trainer?.cancel()
    }

    private func observeThermal(_ state: Int) async {
      guard let report = currentReport else { return }
      currentReport?.thermalTimeline.append(
        ThermalSample(
          elapsedSeconds: Date().timeIntervalSince(report.startedAt), state: state,
          physicalFootprint: ProcessMemory.physicalFootprint(),
          availableMemory: UInt64(os_proc_available_memory())))
      if state >= 3 {
        criticalObserved = true
        preparation?.cancel()
        currentReport?.stoppedForCriticalThermal = true
        currentReport?.stopReason = "critical temperature; sweep stopped"
        await trainer?.criticalThermalObserved()
      }
    }

    public func run(capture: URL, documents: URL) async throws -> URL {
      guard !running else { throw PrepError.invalid("Benchmark already running") }
      running = true
      defer {
        running = false
        monitor?.cancel()
        monitor = nil
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        thermalObserver = nil
      }
      criticalObserved = false
      cancelled = false
      var prep = PrepConfig()
      // Identical initialization across all caps, below the smallest cap's reserve.
      prep.maximumSeedPoints = 100_000
      let reportURL = documents.appendingPathComponent("phone-train-\(UUID().uuidString).json")
      try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
      currentReport = BenchmarkReport(
        capturePath: capture.path, startedAt: Date(),
        device: Self.machine(),
        operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
        physicalMemory: ProcessInfo.processInfo.physicalMemory, prep: prep)
      thermalObserver = NotificationCenter.default.addObserver(
        forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
      ) { [weak self] _ in
        let state = ProcessInfo.processInfo.thermalState.rawValue
        Task { await self?.observeThermal(state) }
      }
      monitor = Task { [weak self] in
        while !Task.isCancelled {
          await self?.observeThermal(ProcessInfo.processInfo.thermalState.rawValue)
          do { try await Task.sleep(for: .milliseconds(250)) } catch { return }
        }
      }
      try currentReport?.write(to: reportURL)
      for item in BenchmarkCase.sweep {
        if cancelled || Task.isCancelled {
          currentReport?.stopReason = "cancelled"
          break
        }
        if criticalObserved || ProcessInfo.processInfo.thermalState == .critical {
          currentReport?.stoppedForCriticalThermal = true
          currentReport?.stopReason = "critical temperature before next run"
          break
        }
        while !active && !cancelled && !criticalObserved && !Task.isCancelled {
          try? await Task.sleep(for: .milliseconds(100))
        }
        if cancelled || Task.isCancelled {
          currentReport?.stopReason = "cancelled"
          break
        }
        if criticalObserved { break }
        var row = BenchmarkRun(configuration: item)
        row.availableAtStart = UInt64(os_proc_available_memory())
        row.thermalAtStart = ProcessInfo.processInfo.thermalState.rawValue
        row.peakPhysicalFootprint = ProcessMemory.physicalFootprint()
        let started = ProcessInfo.processInfo.systemUptime
        let timelineStart = currentReport?.thermalTimeline.count ?? 0
        var dataset: PreparedDataset?
        do {
          prep.trainingMaxSide = item.maxSide
          let prepConfig = prep
          let preparing = Task {
            try await CaptureDatasetBuilder().prepare(bundle: capture, config: prepConfig)
          }
          preparation = preparing
          let prepared = try await withTaskCancellationHandler {
            try await preparing.value
          } onCancel: {
            preparing.cancel()
          }
          preparation = nil
          dataset = prepared
          if criticalObserved { throw PrepError.invalid("Critical temperature during preparation") }
          row.preparationSeconds = ProcessInfo.processInfo.systemUptime - started
          row.seedCount = prepared.descriptor.points.count
          row.seedTargetReached = prepared.descriptor.points.count >= prep.minimumSeedPoints
          row.keyframeCount = prepared.frames.count
          row.voxelSize = prepared.voxelSize
          let config = PhoneTrainingConfig.benchmark(maxGaussians: item.maxGaussians)
          row.estimatedPeakBytes = try PhoneTrainer.estimatedPeak(dataset: prepared, config: config)
          // Companion storage leaves the immutable upload bundle untouched.
          let output = capture.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(
              "PhoneTrain/\(capture.lastPathComponent)/\(item.maxGaussians)-\(item.maxSide)")
          let worker = PhoneTrainer(
            dataset: prepared, capture: capture,
            outputDirectory: output, prep: prep, config: config)
          trainer = worker
          if criticalObserved { await worker.criticalThermalObserved() }
          await worker.setActive(active)
          if cancelled { await worker.cancel() }
          while !active && !cancelled && !criticalObserved {
            try await Task.sleep(for: .milliseconds(100))
          }
          if criticalObserved { await worker.criticalThermalObserved() }
          let trainStart = ProcessInfo.processInfo.systemUptime
          let outcome = try await worker.run()
          row.trainingSeconds = ProcessInfo.processInfo.systemUptime - trainStart
          row.samples = await worker.recordedSamples()
          row.status = outcome.rawValue
          if outcome == .critical {
            currentReport?.stoppedForCriticalThermal = true
            currentReport?.stopReason = "critical temperature; checkpoint saved"
          } else if outcome == .finished {
            // Recheck after each actor hop; an inactive app never starts evaluation.
            while !active && !cancelled && !criticalObserved {
              try await Task.sleep(for: .milliseconds(100))
            }
            if !cancelled && !criticalObserved && ProcessInfo.processInfo.thermalState != .critical
            {
              let evalStart = ProcessInfo.processInfo.systemUptime
              let metrics = try await worker.evaluate()
              row.evaluationSeconds = ProcessInfo.processInfo.systemUptime - evalStart
              row.psnr = metrics.psnr.isFinite ? metrics.psnr : nil
              row.ssim = metrics.ssim.isFinite ? metrics.ssim : nil
              row.evaluatedViews = metrics.numTest
              // Benchmark leaves a real output for later viewer verification.
              while !active && !cancelled && !criticalObserved {
                try await Task.sleep(for: .milliseconds(100))
              }
              if !cancelled && !criticalObserved
                && ProcessInfo.processInfo.thermalState != .critical
              {
                row.outputPath = try await worker.exportPLY().path
              }
            }
          }
          try await worker.close()
        } catch {
          row.status =
            criticalObserved ? "critical" : (cancelled || Task.isCancelled ? "cancelled" : "failed")
          row.error = String(describing: error)
          if let trainer {
            row.samples = await trainer.recordedSamples()
            try? await trainer.close()
          }
        }
        trainer = nil
        preparation = nil
        try? dataset?.cleanup()
        row.totalSeconds = ProcessInfo.processInfo.systemUptime - started
        row.thermalAtEnd = ProcessInfo.processInfo.thermalState.rawValue
        var footprints: [UInt64] = row.samples.compactMap(\.physicalFootprint)
        if let timeline = currentReport?.thermalTimeline {
          footprints.append(contentsOf: timeline.dropFirst(timelineStart).compactMap(\.physicalFootprint))
        }
        if let peak = row.peakPhysicalFootprint { footprints.append(peak) }
        if let now = ProcessMemory.physicalFootprint() { footprints.append(now) }
        row.peakPhysicalFootprint = footprints.max()
        currentReport?.runs.append(row)
        if criticalObserved || ProcessInfo.processInfo.thermalState == .critical
          || row.status == "critical"
        {
          currentReport?.stoppedForCriticalThermal = true
          currentReport?.stopReason = "critical temperature; sweep stopped"
        }
        try currentReport?.write(to: reportURL)
        if currentReport?.stoppedForCriticalThermal == true { break }
      }
      if cancelled || Task.isCancelled { currentReport?.stopReason = "cancelled" }
      try currentReport?.write(to: reportURL)
      return reportURL
    }

    private static func machine() -> String {
      var system = utsname()
      uname(&system)
      return withUnsafeBytes(of: &system.machine) { bytes in
        String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
      }
    }
  }
#endif
