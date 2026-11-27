#if canImport(Msplat) && os(iOS)
  import Foundation
  import LiveSplat
  import Msplat
  import Observation
  import PhoneTrain
  import PhoneTrainPrep
  import UIKit
  import simd

  @MainActor @Observable final class TrainingModel {
    let capture: LocalCapture
    let simulated: Bool
    var experience = TrainingExperience()
    var best = false
    var progress = 0.0
    var eta = "—"
    var sharpness: Int?
    var thermal = false
    var surface: MetalPreviewSurface?
    var scene: SavedSplat?
    var renderer: SavedSplatRenderer?
    var orbit = SplatOrbit()
    var shareURL: URL?
    var active = true
    var inPlace = false
    var cameraOnly = false
    var interacting = false
    private var trainer: PhoneTrainer?
    private var operation: Task<Void, Never>?
    private var observer: Task<Void, Never>?
    private var previewObserver: Task<Void, Never>?
    private var stopRequested = false
    private var critical = false
    private var lastTick: Double?
    private var cameraOpacity: Float = 0

    init(capture: LocalCapture) { self.capture = capture; simulated = capture.isSimulated }

    func loadExisting() async {
      let capture = capture
      guard let record = await Task.detached(operation: { capture.phoneSplat }).value else { return }
      do {
        try await loadResult(record.url(for: capture))
        sharpness = record.sharpness
        experience.showResult()
      } catch { experience.fail() }
    }
    func start() {
      guard operation == nil else { return }
      stopRequested = false
      experience.begin()
      progress = 0
      eta = "—"
      sharpness = nil
      let config = best ? PhoneTrainingConfig.best : PhoneTrainingConfig.quick
      operation = Task { [self] in
        var prepared: PreparedDataset?
        do {
          let prep = PrepConfig()
          let dataset = try await CaptureDatasetBuilder().prepare(bundle: capture.directory, config: prep)
          prepared = dataset
          guard !stopRequested else { try dataset.cleanup(); operation = nil; return }
          // Seed centre from training geometry, not camera-origin assumptions.
          let xyz = dataset.descriptor.points.xyz
          var center = SIMD3<Float>(repeating: 0)
          let count = dataset.descriptor.points.count
          for i in 0..<count { center += SIMD3(xyz[i*3], xyz[i*3+1], xyz[i*3+2]) }
          if count > 0 { center /= Float(count) }
          orbit.center = center
          let pose = dataset.descriptor.frames[0].cameraToWorld.elements
          let camera = SIMD3(pose[3], pose[7], pose[11])
          let delta = camera - center
          orbit.distance = max(0.3, simd_length(delta))
          orbit.yaw = atan2(delta.x, delta.z)
          orbit.pitch = asin(max(-0.95, min(0.95, delta.y / orbit.distance)))
          let root = PhoneSplatRecord.directory(for: capture).appendingPathComponent("Training")
          let trainer = PhoneTrainer(dataset: dataset, capture: capture.directory,
            outputDirectory: root, prep: prep, config: config)
          self.trainer = trainer
          prepared = nil
          await trainer.setActive(active)
          await trainer.setPreviewPose(try CameraPose(elements: orbit.pose))
          let events = await trainer.progress()
          observer = Task { [weak self] in
            for await event in events {
              guard let self else { return }
              if let sample = event.sample {
                self.progress = min(1, Double(sample.step) / Double(config.iterations))
                self.eta = TrainingPresentation.eta(sample.etaSeconds)
                self.thermal = sample.thermalState >= 2
              }
              self.sharpness = TrainingPresentation.sharpness(psnr: event.psnr)
            }
          }
          let previews = await trainer.previews()
          previewObserver = Task { [weak self] in
            for await surface in previews { self?.surface = surface }
          }
          experience.prepared()
          try await finishRun(trainer)
        } catch {
          if stopRequested { experience.cancel() } else { experience.fail() }
        }
        if !critical {
          if let trainer { try? await trainer.closeAndCleanup() }
          else if let prepared { try? prepared.cleanup() }
          trainer = nil
          observer?.cancel(); previewObserver?.cancel()
        }
        operation = nil
      }
    }

    private func finishRun(_ trainer: PhoneTrainer) async throws {
      let outcome = try await trainer.run()
      if outcome == .critical {
        critical = true
        thermal = true
        experience.cool()
        return
      }
      guard outcome == .finished, !stopRequested else { experience.cancel(); return }
      let ply: URL
      do {
        try await waitForForeground()
        if stopRequested { experience.cancel(); return }
        let metrics = try await trainer.evaluate()
        sharpness = TrainingPresentation.sharpness(psnr: metrics.psnr)
        try await waitForForeground()
        if stopRequested { experience.cancel(); return }
        ply = try await trainer.exportPLY()
      } catch {
        if await trainer.needsCooling(), !stopRequested {
          critical = true
          thermal = true
          experience.cool()
          return
        }
        throw error
      }
      let capture = capture
      let score = sharpness
      let record = try await Task.detached(priority: .utility) {
        try PhoneSplatRecord.save(ply: ply, sharpness: score, checkpoint: nil, for: capture)
      }.value
      try await trainer.closeAndCleanup()
      self.trainer = nil
      observer?.cancel(); previewObserver?.cancel()
      progress = 1
      experience.finish()
      try await loadResult(record.url(for: capture))
      guard !stopRequested else { experience.cancel(); return }
      UINotificationFeedbackGenerator().notificationOccurred(.success)
      try await Task.sleep(for: .milliseconds(600))
      experience.showResult()
    }
    private func waitForForeground() async throws {
      while !active && !stopRequested { try await Task.sleep(for: .milliseconds(100)) }
    }
    private func loadResult(_ url: URL) async throws {
      let result = try await Task.detached(priority: .userInitiated) {
        try SavedSplat(data: Data(contentsOf: url, options: .mappedIfSafe))
      }.value
      scene = result
      renderer = try SavedSplatRenderer(scene: result)
      shareURL = url
      orbit.center = result.center
      orbit.distance = max(0.3, result.radius * 2.4)
      surface = nil
    }
    func pause() async {
      experience.pause()
      await trainer?.pause()
    }
    func resume() async {
      guard let trainer else { return }
      experience.resume()
      await trainer.resume()
      if critical, operation == nil {
        critical = false
        operation = Task { [self] in
          do { try await finishRun(trainer) } catch { experience.fail() }
          if !critical {
            if self.trainer != nil { try? await trainer.closeAndCleanup() }
            self.trainer = nil
            observer?.cancel(); previewObserver?.cancel()
          }
          operation = nil
        }
      }
    }
    func setActive(_ value: Bool) async {
      active = value
      lastTick = nil
      await trainer?.setActive(value)
    }
    func stop() async {
      stopRequested = true
      await trainer?.cancel()
      await operation?.value
      if let trainer { try? await trainer.closeAndCleanup() }
      trainer = nil
      observer?.cancel(); previewObserver?.cancel()
    }
    func focus() {
      guard let scene else { return }
      orbit.center = scene.center
      orbit.distance = max(0.3, scene.radius * 2.4)
      orbit.pitch = 0.15
    }
    func tick(size: CGSize, camera: CameraModel?) async {
      guard active else { return }
      let now = ProcessInfo.processInfo.systemUptime
      let frameSeconds = Float(min(0.2, max(0, now - (lastTick ?? now))))
      let targetOpacity: Float = cameraOnly ? 0 : 1
      cameraOpacity += (targetOpacity - cameraOpacity) * min(1, frameSeconds * 6)
      if let lastTick, !interacting, !inPlace {
        let elapsed = Float(min(0.2, max(0, now-lastTick)))
        orbit.yaw += elapsed * 0.08
      }
      lastTick = now
      if let trainer, let pose = try? CameraPose(elements: orbit.pose) {
        await trainer.setPreviewPose(pose)
      }
      guard let scene, let renderer else { return }
      let view: simd_float4x4
      let projection: simd_float4x4
      if inPlace {
        guard let camera, let matrices = camera.trainedSplatCamera(size: size, capture: capture.directory)
        else { inPlace = false; return }
        camera.showTrainedSplat(renderer)
        view = matrices.0; projection = matrices.1
      } else {
        view = Self.matrix(orbit.pose).inverse
        let aspect = Float(max(1,size.width) / max(1,size.height))
        let focal: Float = 1.9
        let near: Float = 0.02
        let far: Float = 100
        let depthScale: Float = far / (near - far)
        let depthOffset: Float = near * depthScale
        projection = simd_float4x4(columns: (
          SIMD4(focal/aspect,0,0,0), SIMD4(0,focal,0,0),
          SIMD4(0,0,depthScale,-1), SIMD4(0,0,depthOffset,0)))
      }
      let rows = Self.rows(view)
      let indices = await Task.detached(priority: .userInitiated) { scene.sortedIndices(viewRows: rows) }.value
      guard active else { return }
      renderer.update(view: view, projection: projection,
        size: SIMD2(Float(size.width),Float(size.height)), opacity: inPlace ? cameraOpacity : 1, indices: indices)
    }
    static func matrix(_ rows: [Float]) -> simd_float4x4 {
      simd_float4x4(columns: (SIMD4(rows[0],rows[4],rows[8],rows[12]),
        SIMD4(rows[1],rows[5],rows[9],rows[13]), SIMD4(rows[2],rows[6],rows[10],rows[14]),
        SIMD4(rows[3],rows[7],rows[11],rows[15])))
    }
    static func rows(_ matrix: simd_float4x4) -> [Float] {
      var rows: [Float] = []
      for row in 0..<4 { for column in 0..<4 { rows.append(matrix[column][row]) } }
      return rows
    }
  }
#endif
