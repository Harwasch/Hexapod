import ARAnswer
import ARKit
import EntitySchema
import Foundation
import LiveSplat
import Observation
import Perception
import RealityKit
import Router
import UIKit
import Voice

@MainActor protocol RecordingControl: AnyObject {
  var isRecording: Bool { get }
  func toggleRecording()
}

@MainActor protocol PushToTalkControl: AnyObject {
  func beginTalking()
  func endTalking()
}

/// The camera shares providers and one world coordinate frame.
@MainActor @Observable final class CameraModel: PushToTalkControl {
  private(set) var cameraReady = false
  private(set) var isTalking = false
  private(set) var recording = false
  private(set) var failed = false
  var card: EntitySchema.Entity?
  private var presentation = AnswerPresentation()
  private let voice: any VoiceInput
  let perception: ScenePerception
  private let router: any AnswerRouting
  private let recorder = SharedSessionRecorder()
  private var overlay: (any AnswerOverlaySink)?
  private let captureSource: any CameraCaptureSource
  private let fakeOverlay: Bool
  let simulated: Bool
  let liveSplat: (any LiveSplatRenderer)?
  private var replayTask: Task<Void, Never>?
  var usesLiveCapture: Bool { captureSource.usesLiveSession }
  private weak var arView: ARView?
  private var voiceTask: Task<Void, Never>?
  private var answerTask: Task<Void, Never>?
  private var perceptionTask: Task<Void, Never>?
  private var questionGeneration = UUID()
  private var voiceGeneration = UUID()
  private var lastPerceptionTime: TimeInterval = -.infinity
  private var frameGeneration = UUID()
  private var lastMaskUpdate: TimeInterval = 0
  private var hasSnapshot = false
  private var viewportToImage: [Double] = [1, 0, 0, 1, 0, 0]

  init(providers: CameraProviders) {
    voice = providers.voice
    perception = ScenePerception(scene: providers.perception)
    captureSource = providers.capture
    fakeOverlay = providers.fakeOverlay
    simulated = providers.simulated
    liveSplat = providers.liveSplat
    router = AnswerRouter(
      planner: providers.planner, fast: DeviceFastAnswer(scene: perception), deep: providers.deep)
    recorder.exportBundle = { [weak self] directory in
      guard let renderer = self?.liveSplat else { return }
      renderer.stop()
      _ = try renderer.exportPLY(to: directory)
      renderer.start()
    }
    recorder.onChange = { [weak self] active, failure in
      self?.recording = active
      if failure { self?.failed = true }
    }
  }

  func reportUnavailable() {
    failed = true
    cameraReady = false
  }

  func attach(_ view: ARView) {
    arView = view
    if usesLiveCapture { liveSplat?.start() }
    cameraReady = true
    do {
      if fakeOverlay {
        overlay = FakeOverlaySink()
      } else {
        overlay = try RealityAnswerOverlay(view: view)
      }
      failed = false
    } catch { failed = true }
  }

  func beginReplay() {
    guard !usesLiveCapture, replayTask == nil else { return }
    replayTask = Task { [weak self] in
      guard let self else { return }
      do {
        for try await snapshot in captureSource.frames() {
          try Task.checkCancellation()
          let time = ProcessInfo.processInfo.systemUptime
          await perception.update(snapshot, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: time)
          hasSnapshot = true
          cameraReady = true
          let entities = try await perception.inspect(.init(subject: "subject", tools: [.count]))
          overlay?.update(entities: entities)
          highlight(await perception.selection())
        }
      } catch is CancellationError {} catch { failed = true }
    }
  }

  func frame(_ frame: ARFrame) {
    recorder.append(frame)
    if let view = arView, let renderer = liveSplat {
      let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
      do {
        try renderer.ingest(frame, orientation: orientation, viewportSize: view.bounds.size)
      } catch {
        renderer.trackingUnavailable()
        failed = true
      }
    }
    (overlay as? RealityAnswerOverlay)?.updateCamera()
    if ProcessInfo.processInfo.systemUptime - lastMaskUpdate > 0.75 { overlay?.setMask(nil) }
    guard perceptionTask == nil, frame.timestamp - lastPerceptionTime >= 0.35,
      let view = arView, view.bounds.width > 0, view.bounds.height > 0,
      case .normal = frame.camera.trackingState
    else { return }
    lastPerceptionTime = frame.timestamp
    let inverse = frame.displayTransform(for: .portrait, viewportSize: view.bounds.size).inverted()
    viewportToImage = [inverse.a, inverse.b, inverse.c, inverse.d, inverse.tx, inverse.ty].map {
      Double($0)
    }
    // FrameSnapshot copies immutable ARFrame buffers off the UI thread, one frame at a time.
    let transfer = FrameTransfer(frame)
    let transform = viewportToImage
    let capturedAt = ProcessInfo.processInfo.systemUptime
    let scene = perception
    let frameID = UUID()
    frameGeneration = frameID
    perceptionTask = Task { [weak self] in
      defer { if self?.frameGeneration == frameID { self?.perceptionTask = nil } }
      do {
        let snapshot = try await Task.detached(priority: .userInitiated) {
          try Task.checkCancellation()
          return try FrameSnapshot(frame: transfer.frame)
        }.value
        try Task.checkCancellation()
        guard self?.frameGeneration == frameID else { return }
        await scene.update(snapshot, viewportToImage: transform, capturedAt: capturedAt)
        self?.hasSnapshot = true
        let selection = try await scene.refreshSelection()
        try Task.checkCancellation()
        self?.highlight(selection)
      } catch is CancellationError {} catch { self?.overlay?.setMask(nil) }
    }
  }

  func tap(_ point: CGPoint) {
    if let entity = (overlay as? RealityAnswerOverlay)?.pin(at: point) {
      card = entity
      return
    }
    guard let view = arView, view.bounds.width > 0, view.bounds.height > 0, hasSnapshot else {
      return
    }
    let t = viewportToImage
    let u = Double(point.x / view.bounds.width)
    let v = Double(point.y / view.bounds.height)
    let prompt = ImagePoint(x: t[0] * u + t[2] * v + t[4], y: t[1] * u + t[3] * v + t[5])
    answerTask?.cancel()
    let generation = UUID()
    questionGeneration = generation
    card = nil
    overlay?.stopSpeaking()
    Task {
      await router.cancel()
      guard generation == questionGeneration else { return }
      do {
        let selected = try await perception.select(prompt)
        guard generation == questionGeneration else { return }
        presentation.reset()
        selected.entities.forEach { presentation.receive(.entity($0)) }
        overlay?.update(entities: presentation.entities)
        highlight(selected)
        failed = false
      } catch is CancellationError {} catch {
        if generation == questionGeneration {
          failed = true
          overlay?.setMask(nil)
        }
      }
    }
  }

  func toggleRecording() { if usesLiveCapture { recorder.toggleRecording() } }

  func beginTalking() {
    guard !isTalking else { return }
    isTalking = true
    failed = false
    card = nil
    voiceTask?.cancel()
    voice.cancel()
    answerTask?.cancel()
    questionGeneration = UUID()
    let router = router
    Task { await router.cancel() }
    overlay?.stopSpeaking()
    let id = UUID()
    voiceGeneration = id
    voiceTask = Task {
      var accumulator = TranscriptAccumulator()
      do {
        let stream = try await voice.start()
        for try await result in stream {
          try Task.checkCancellation()
          guard id == voiceGeneration else { return }
          accumulator.receive(result)
        }
        guard id == voiceGeneration, !Task.isCancelled else { return }
        isTalking = false
        if !accumulator.text.isEmpty { ask(accumulator.text) }
      } catch is CancellationError {} catch {
        if id == voiceGeneration {
          failed = true
          isTalking = false
          voice.cancel()
        }
      }
    }
  }

  func endTalking() {
    guard isTalking else { return }
    isTalking = false
    let id = voiceGeneration
    Task {
      guard id == voiceGeneration else { return }
      await voice.stop()
    }
  }

  private func ask(_ text: String) {
    answerTask?.cancel()
    let id = UUID()
    questionGeneration = id
    presentation.reset()
    overlay?.update(entities: [])
    overlay?.stopSpeaking()
    answerTask = Task {
      guard id == questionGeneration, !Task.isCancelled else { return }
      let request = AskRequest(id: id.uuidString, question: text)
      let stream = await router.ask(request)
      guard id == questionGeneration, !Task.isCancelled else { return }
      for await event in stream {
        guard id == questionGeneration, !Task.isCancelled else { return }
        presentation.receive(event)
        switch event {
        case .entity: overlay?.update(entities: presentation.entities)
        case .speech(let text): overlay?.speak(text)
        case .status: failed = presentation.isUnavailable
        case .done:
          let selection = await perception.selection()
          guard id == questionGeneration else { return }
          highlight(selection)
        }
      }
    }
  }

  private func highlight(_ selection: SceneSelection) {
    guard ProcessInfo.processInfo.systemUptime - selection.capturedAt <= 0.75 else {
      overlay?.setMask(nil)
      return
    }
    let masks = selection.masks
    let transform = selection.viewportToImage
    lastMaskUpdate = selection.capturedAt
    var projected: HighlightMask?
    for mask in masks {
      guard
        let next = HighlightMask.project(
          width: mask.width, height: mask.height,
          pixels: mask.pixels, viewportToImage: transform)
      else { continue }
      if projected == nil {
        projected = next
      } else {
        for i in next.pixels.indices {
          projected!.pixels[i] = max(projected!.pixels[i], next.pixels[i])
        }
      }
    }
    overlay?.setMask(projected)
  }

  func suspend() {
    replayTask?.cancel()
    replayTask = nil
    cameraReady = false
    isTalking = false
    voiceGeneration = UUID()
    questionGeneration = UUID()
    voiceTask?.cancel()
    voice.cancel()
    answerTask?.cancel()
    perceptionTask?.cancel()
    frameGeneration = UUID()
    perceptionTask = nil
    hasSnapshot = false
    lastPerceptionTime = -.infinity
    let router = router
    let perception = perception
    Task {
      await router.cancel()
      await perception.clear()
    }
    recorder.stop()
    liveSplat?.stop()
    overlay?.remove()
    overlay = nil
    arView = nil
    presentation.reset()
    card = nil
  }
}

/// ARFrame and its buffers are immutable snapshots retained until the copy finishes.
/// No ARSession or mutable renderer crosses the task boundary.
private final class FrameTransfer: @unchecked Sendable {
  let frame: ARFrame
  init(_ frame: ARFrame) { self.frame = frame }
}
