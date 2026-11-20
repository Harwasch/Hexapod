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
  private(set) var failureText = "Unavailable"
  private(set) var capturePhase = CapturePhase.idle
  private(set) var preparingVoice = false
  private(set) var transcript = ""
  private(set) var question = ""
  private(set) var answerText = ""
  private(set) var answering = false
  var latestCapture: URL?
  var savedCapture: URL?
  var captureDuration: TimeInterval = 0
  var siteName: String?
  var siteID: UUID?
  private var recordingStarted: Date?
  var splatEnabled: Bool {
    (liveSplat as? DefaultLiveSplatRenderer)?.preference.trainingEnabled ?? (liveSplat != nil)
  }
  var coverage: Double { liveSplat?.coverageFraction ?? 0 }
  var answerEntities: [EntitySchema.Entity] { presentation.entities }
  var splatFallback: Bool {
    guard splatEnabled, let liveSplat else { return false }
    if case .fallback = liveSplat.status { return true }
    return liveSplat.status == .pointPreview
  }
  var gapPosition: CGPoint? {
    guard splatEnabled, let renderer = liveSplat as? DefaultLiveSplatRenderer,
      renderer.latestFrame != nil,
      let point = renderer.coverage.gapHint(displayTransform: renderer.displayTransform)
    else { return nil }
    return CGPoint(x: point.x, y: point.y)
  }

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
  private var previewTask: Task<Void, Never>?
  var usesLiveCapture: Bool { captureSource.usesLiveSession }
  var canRecord: Bool { usesLiveCapture || (simulated && captureSource.fixtureDirectory != nil) }
  struct ScreenPin: Identifiable {
    let id: Int
    let x: CGFloat
    let y: CGFloat
  }
  private(set) var previewGeneration = 0
  var previewSize: CGSize = .zero
  var screenPins: [ScreenPin] {
    if LaunchMode.current.isUITest && simulated {
      return presentation.entities.indices.map {
        ScreenPin(id: $0, x: previewSize.width / 2, y: previewSize.height * 0.55)
      }
    }
    guard let view = arView else { return [] }
    return presentation.entities.enumerated().compactMap { index, entity in
      let p = entity.anchor.world
      if let point = view.project(SIMD3(Float(p.x), Float(p.y), Float(p.z))),
        view.bounds.contains(point)
      {
        return ScreenPin(id: index, x: point.x, y: point.y)
      }
      if simulated {
        return ScreenPin(id: index, x: view.bounds.midX, y: view.bounds.height * 0.55)
      }
      return nil
    }
  }
  func openPin(_ index: Int) {
    guard presentation.entities.indices.contains(index) else { return }
    card = presentation.entities[index]
  }
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
      let splats = renderer.splats
      guard !splats.isEmpty else { return }
      try await Task.detached(priority: .utility) {
        try SplatPLY.encode(splats).write(
          to: directory.appendingPathComponent("livesplat.ply"), options: .atomic)
      }.value
    }
    siteName = UserDefaults.standard.string(forKey: "hex.currentSite.name")
    siteID = UserDefaults.standard.string(forKey: "hex.currentSite.id").flatMap(
      UUID.init(uuidString:))
    recorder.captureDetails = { [weak self] in
      CaptureDetails(
        siteName: self?.siteName, siteID: self?.siteID, duration: self?.captureDuration ?? 0)
    }
    recorder.onPhase = { [weak self] phase in
      guard let self else { return }
      capturePhase = phase
      switch phase {
      case .recording(let start):
        recordingStarted = start
        liveSplat?.start()
      case .stopping:
        captureDuration = LaunchMode.current.date().timeIntervalSince(
          recordingStarted ?? LaunchMode.current.date())
        liveSplat?.stop()
      case .saved(let directory):
        savedCapture = directory
        latestCapture = directory
      case .failed:
        failureText = "Scan failed"
        liveSplat?.stop()
      default: break
      }
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

  func beginUITestPreview(size: CGSize) {
    previewSize = size
    guard !usesLiveCapture else { return }
    overlay = FakeOverlaySink()
    beginReplay()
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
          if LaunchMode.current.isUITest { break }
        }
      } catch is CancellationError {} catch { failed = true }
    }
  }

  func frame(_ frame: ARFrame) {
    recorder.append(frame)
    if recording, splatEnabled, let view = arView, let renderer = liveSplat {
      let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
      do {
        try renderer.ingest(frame, orientation: orientation, viewportSize: view.bounds.size)
      } catch {
        renderer.trackingUnavailable()
        failed = true
        failureText = "Preview unavailable"
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
          failureText = "Selection unavailable"
          overlay?.setMask(nil)
        }
      }
    }
  }

  func toggleRecording() {
    guard canRecord, cameraReady else { return }
    if !usesLiveCapture {
      toggleFixtureRecording()
      return
    }
    if case .stopping = capturePhase { return }
    if case .starting = capturePhase { return }
    failed = false
    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    if !recording { clearAnswer() }
    recorder.toggleRecording()
  }

  private func toggleFixtureRecording() {
    guard let source = captureSource.fixtureDirectory else { return }
    if recording {
      recording = false
      capturePhase = .stopping
      captureDuration = LaunchMode.current.date().timeIntervalSince(
        recordingStarted ?? LaunchMode.current.date())
      previewTask?.cancel()
      liveSplat?.stop()
      let destination = FileManager.default.urls(
        for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Captures/" + UUID().uuidString)
      let details = CaptureDetails(siteName: siteName, siteID: siteID, duration: captureDuration)
      Task {
        do {
          let directory = try await FixtureCaptureRecording.save(
            source: source.appendingPathComponent("bundle"),
            destination: destination, details: details)
          capturePhase = .saved(directory)
          latestCapture = directory
          savedCapture = directory
        } catch {
          capturePhase = .failed
          failed = true
        }
      }
    } else if !capturePhase.isBusy {
      clearAnswer()
      let start = LaunchMode.current.date()
      recordingStarted = start
      capturePhase = .recording(start)
      recording = true
      previewGeneration = 0
      liveSplat?.start()
      if LaunchMode.current.isUITest { seedStaticPreview() }
    }
  }

  private func seedStaticPreview() {
    guard let directory = LaunchMode.current.previewDirectory, let renderer = liveSplat else { return }
    previewTask?.cancel()
    previewTask = Task {
      do {
        try await FixtureSplatPreview.seed(renderer: renderer, directory: directory)
        try Task.checkCancellation()
        previewGeneration += 1
      } catch is CancellationError {} catch {
        failed = true
        failureText = "Fixture preview unavailable"
      }
    }
  }

  func chooseSite(id: UUID?, name: String?) {
    siteID = id
    siteName = name
    UserDefaults.standard.set(id?.uuidString, forKey: "hex.currentSite.id")
    UserDefaults.standard.set(name, forKey: "hex.currentSite.name")
  }
  func dismissCapture() {
    savedCapture = nil
    capturePhase = .idle
  }
  func clearAnswer() {
    answerTask?.cancel()
    questionGeneration = UUID()
    question = ""
    answerText = ""
    answering = false
    overlay?.stopSpeaking()
    overlay?.update(entities: [])
    overlay?.setMask(nil)
    presentation.reset()
    let router = router
    Task { await router.cancel() }
  }
  func replayAnswer() { if !answerText.isEmpty { overlay?.speak(answerText) } }
  func pinCard() {
    guard let card else { return }
    overlay?.update(entities: [card])
    self.card = nil
  }

  func beginTalking() {
    guard !isTalking else { return }
    isTalking = true
    preparingVoice = true
    transcript = ""
    question = ""
    answerText = ""
    answering = false
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
        guard id == voiceGeneration, !Task.isCancelled else { return }
        preparingVoice = false
        for try await result in stream {
          try Task.checkCancellation()
          guard id == voiceGeneration else { return }
          accumulator.receive(result)
          transcript = accumulator.text
        }
        guard id == voiceGeneration, !Task.isCancelled else { return }
        isTalking = false
        if !accumulator.text.isEmpty {
          ask(accumulator.text)
        } else {
          question = ""
          answering = false
        }
      } catch is CancellationError {} catch {
        if id == voiceGeneration {
          failed = true
          preparingVoice = false
          failureText = "Voice unavailable"
          question = ""
          answering = false
          isTalking = false
          voice.cancel()
        }
      }
    }
  }

  func endTalking() {
    guard isTalking else { return }
    isTalking = false
    if preparingVoice {
      voiceGeneration = UUID()
      voiceTask?.cancel()
      voice.cancel()
      preparingVoice = false
      return
    }
    question = transcript.isEmpty ? "Listening" : transcript
    answering = true
    let id = voiceGeneration
    Task {
      guard id == voiceGeneration else { return }
      await voice.stop()
    }
  }

  func ask(_ text: String) {
    question = text
    answerText = ""
    answering = true
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
        case .speech(let text):
          answerText = text
          overlay?.speak(text)
        case .status(let status):
          failed = presentation.isUnavailable
          if failed {
            failureText = status.hasPrefix("Deep") ? "Cloud unavailable" : "Answer unavailable"
          }
        case .done:
          answering = false
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
    previewTask?.cancel()
    previewTask = nil
    replayTask?.cancel()
    replayTask = nil
    cameraReady = false
    isTalking = false
    preparingVoice = false
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
    if !usesLiveCapture && recording { toggleFixtureRecording() }
    recorder.stop()
    liveSplat?.stop()
    overlay?.remove()
    overlay = nil
    arView = nil
    presentation.reset()
    card = nil
    answering = false
  }
}

/// ARFrame and its buffers are immutable snapshots retained until the copy finishes.
/// No ARSession or mutable renderer crosses the task boundary.
private final class FrameTransfer: @unchecked Sendable {
  let frame: ARFrame
  init(_ frame: ARFrame) { self.frame = frame }
}
