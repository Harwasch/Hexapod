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
    private(set) var simulated: Bool
    var experience = TrainingExperience()
    var best = false
    var progress = 0.0
    var eta = "—"
    var sharpness: Int?
    var thermal = false
    var surface: MetalPreviewSurface?
    var previewImage: UIImage?
    private var previewPNG: Data?
    var scene: SavedSplat?
    var renderer: SavedSplatRenderer?
    var orbit = SplatOrbit()
    var shareURL: URL?
    var active = true
    var inPlace = false
    var cameraOnly = false
    var interacting = false
    private var session: (any PhoneTrainingSession)?
    private var nativePreview: (any PhoneTrainingNativePreview)?
    private var operation: Task<Void, Never>?
    private var observer: Task<Void, Never>?
    private var previewObserver: Task<Void, Never>?
    private var stopRequested = false
    private var critical = false
    private var lastTick: Double?
    private var cameraOpacity: Float = 0

    init(capture: LocalCapture) {
      self.capture = capture
      simulated = capture.isSimulated || LaunchMode.current.isUITest
    }

    func loadExisting() async {
      guard experience.stage == .choosing else { return }
      let capture = capture
      guard let record = await Task.detached(operation: { capture.phoneSplat }).value, experience.stage == .choosing else { return }
      do {
        simulated = simulated || record.simulated == true
        if let url = record.previewURL(for: capture) {
          let data = try await Task.detached { try Data(contentsOf: url) }.value
          guard experience.stage == .choosing else { return }
          previewPNG = data
          previewImage = UIImage(data: data)
        }
        try await loadResult(record.url(for: capture), bitmapOnly: previewImage != nil || LaunchMode.current.isUITest, onlyWhileChoosing: true)
        guard experience.stage == .choosing else { return }
        sharpness = record.sharpness
        experience.showResult()
      } catch { if experience.stage == .choosing { experience.fail() } }
    }
    func start() {
      guard operation == nil else { return }
      stopRequested = false
      experience.begin()
      progress = 0
      eta = "—"
      sharpness = nil
      previewPNG = nil
      previewImage = nil
      surface = nil
      scene = nil
      renderer = nil
      let config = best ? PhoneTrainingConfig.best : PhoneTrainingConfig.quick
      operation = Task { [self] in
        do {
          let output = PhoneSplatRecord.directory(for: capture).appendingPathComponent("Training")
          let selected = try await PhoneTrainingFactory.make(outputDirectory: output) {
            let prep = PrepConfig()
            let dataset = try await CaptureDatasetBuilder().prepare(bundle: capture.directory, config: prep)
            do {
              if stopRequested { throw CancellationError() }
              // Only production construction prepares data or acquires a native owner.
              let xyz = dataset.descriptor.points.xyz
              var center = SIMD3<Float>(repeating: 0)
              let count = dataset.descriptor.points.count
              for i in 0..<count { center += SIMD3(xyz[i*3], xyz[i*3+1], xyz[i*3+2]) }
              if count > 0 { center /= Float(count) }
              orbit.center = center
              let pose = dataset.descriptor.frames[0].cameraToWorld.elements
              let cameraPosition = SIMD3(pose[3], pose[7], pose[11])
              let delta = cameraPosition - center
              orbit.distance = max(0.3, simd_length(delta))
              orbit.yaw = atan2(delta.x, delta.z)
              orbit.pitch = asin(max(-0.95, min(0.95, delta.y / orbit.distance)))
              let owner = PhoneTrainer(dataset: dataset, capture: capture.directory,
                outputDirectory: output, prep: prep, config: config)
              return DevicePhoneTrainingSession(trainer: owner)
            } catch {
              try? dataset.cleanup()
              throw error
            }
          }
          session = selected
          nativePreview = selected as? any PhoneTrainingNativePreview
          await selected.setActive(active)
          if stopRequested { await selected.cancel() }
          if let nativePreview {
            await nativePreview.setPreviewPose(try CameraPose(elements: orbit.pose))
            let previews = await nativePreview.previews()
            previewObserver = Task { [weak self] in
              for await surface in previews { self?.surface = surface }
            }
          } else {
            previewPNG = try await selected.previewImage()
            if let previewPNG { previewImage = UIImage(data: previewPNG) }
          }
          let events = await selected.progress()
          let budget: Int? = nativePreview == nil ? nil : Int(config.iterations)
          observer = Task { [weak self] in
            for await event in events {
              guard let self else { return }
              guard self.experience.stage.busy, self.experience.stage != .settling else { continue }
              if let sample = event.sample {
                self.progress = TrainingPresentation.fraction(step: sample.step, budget: budget,
                  elapsed: sample.elapsedSeconds, remaining: sample.etaSeconds)
                self.eta = TrainingPresentation.eta(sample.etaSeconds)
                self.thermal = sample.thermalState >= 2
              }
              self.sharpness = TrainingPresentation.sharpness(psnr: event.psnr)
            }
          }
          experience.prepared()
          try await finishRun(selected)
        } catch {
          if stopRequested { experience.cancel() } else { experience.fail() }
        }
        if !critical {
          try? await releaseSession()
        }
        operation = nil
      }
    }

    private func finishRun(_ selected: any PhoneTrainingSession) async throws {
      let result: PhoneTrainingResult
      do {
        result = try await selected.run()
        if stopRequested { experience.cancel(); return }
        if let nativePreview {
          try await waitForForeground()
          if stopRequested { experience.cancel(); return }
          let measured = try await nativePreview.evaluatePSNR()
          sharpness = TrainingPresentation.sharpness(psnr: measured)
        }
      } catch {
        if let nativePreview, await nativePreview.needsCooling(), !stopRequested {
          critical = true
          thermal = true
          experience.cool()
          return
        }
        throw error
      }
      if stopRequested { experience.cancel(); return }
      simulated = simulated || result.simulated
      let ply = result.ply
      let capture = capture
      let score = sharpness
      let png = previewPNG
      let isSimulated = simulated
      let record = try await Task.detached(priority: .utility) {
        try PhoneSplatRecord.save(ply: ply, sharpness: score, checkpoint: nil, for: capture,
          previewPNG: png, simulated: isSimulated)
      }.value
      try await releaseSession(resultSaved: true)
      progress = 1
      experience.finish()
      try await loadResult(record.url(for: capture), bitmapOnly: previewImage != nil || LaunchMode.current.isUITest)
      guard !stopRequested else { experience.cancel(); return }
      if !LaunchMode.current.isUITest {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        try await Task.sleep(for: .milliseconds(600))
      }
      experience.showResult()
    }
    private func releaseSession(resultSaved: Bool = false) async throws {
      observer?.cancel(); previewObserver?.cancel()
      nativePreview = nil
      surface = nil
      if let session { try await session.closeAndCleanup() }
      session = nil
      if let output = TrainingOutputPolicy.removalURL(for: capture, stage: experience.stage,
        sessionClosed: true, resultSaved: resultSaved) {
        await Task.detached(priority: .utility) {
          try? FileManager.default.removeItem(at: output)
        }.value
      }
    }
    private func waitForForeground() async throws {
      while !active && !stopRequested { try await Task.sleep(for: .milliseconds(100)) }
    }
    private func loadResult(_ url: URL, bitmapOnly: Bool, onlyWhileChoosing: Bool = false) async throws {
      if onlyWhileChoosing && experience.stage != .choosing { return }
      if bitmapOnly {
        guard previewImage != nil else { throw PrepError.invalid("Preview missing") }
        shareURL = url
        surface = nil
        scene = nil
        renderer = nil
        return
      }
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
      await session?.pause()
    }
    func resume() async {
      guard let selected = session else { return }
      experience.resume()
      await selected.resume()
      if critical, operation == nil {
        critical = false
        operation = Task { [self] in
          do { try await finishRun(selected) } catch { experience.fail() }
          if !critical {
            try? await releaseSession()
          }
          operation = nil
        }
      }
    }
    func setActive(_ value: Bool) async {
      active = value
      lastTick = nil
      await session?.setActive(value)
    }
    func stop() async {
      stopRequested = true
      await session?.cancel()
      await operation?.value
      if experience.stage.busy { experience.cancel() }
      critical = false
      try? await releaseSession()
    }
    func focus() {
      guard let scene else { return }
      orbit.center = scene.center
      orbit.distance = max(0.3, scene.radius * 2.4)
      orbit.pitch = 0.15
    }
    func tick(size: CGSize, camera: CameraModel?) async {
      guard active, !LaunchMode.current.isUITest else { return }
      let now = ProcessInfo.processInfo.systemUptime
      let frameSeconds = Float(min(0.2, max(0, now - (lastTick ?? now))))
      let targetOpacity: Float = cameraOnly ? 0 : 1
      cameraOpacity += (targetOpacity - cameraOpacity) * min(1, frameSeconds * 6)
      if let lastTick, !interacting, !inPlace {
        let elapsed = Float(min(0.2, max(0, now-lastTick)))
        orbit.yaw += elapsed * 0.08
      }
      lastTick = now
      if let nativePreview, let pose = try? CameraPose(elements: orbit.pose) {
        await nativePreview.setPreviewPose(pose)
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
