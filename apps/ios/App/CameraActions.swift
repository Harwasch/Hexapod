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
    (liveSplat as? DefaultLiveSplatRenderer)?.preference.previewEnabled ?? (liveSplat != nil)
  }
  var coverage: Double { usesLiveCapture ? scanCoverage : (liveSplat?.coverageFraction ?? 0) }
  var answerEntities: [EntitySchema.Entity] { presentation.entities }
  var answerConnecting: Bool { presentation.isConnecting }

  var card: EntitySchema.Entity?
  private var presentation = AnswerPresentation()
  private let voice: any VoiceInput
  let perception: ScenePerception
  private let router: any AnswerRouting
  private let recorder = SharedSessionRecorder()
  let frameFeed: CameraFrameFeed
  private var overlay: (any AnswerOverlaySink)?
  private let captureSource: any CameraCaptureSource
  private let fakeOverlay: Bool
  let simulated: Bool
  let liveSplat: (any LiveSplatRenderer)?
  private var replayTask: Task<Void, Never>?
  var usesLiveCapture: Bool { captureSource.usesLiveSession }
  var canRecord: Bool { usesLiveCapture || (simulated && captureSource.fixtureDirectory != nil) }
  struct ScreenPin: Identifiable {
    let id: Int
    let x: CGFloat
    let y: CGFloat
  }
  private(set) var screenPins: [ScreenPin] = []
  private func updatePins() {
    guard let view = arView else {
      screenPins = []
      return
    }
    screenPins = presentation.entities.enumerated().compactMap { index, entity in
      let p = entity.anchor.world
      let top = SIMD3(Float(p.x), Float(p.y + entity.extent.y / 2), Float(p.z))
      if let point = view.project(top), view.bounds.contains(point) {
        return ScreenPin(id: index, x: point.x, y: max(22, point.y - 18))
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
  private var hasSnapshot = false
  private var latestCamera: CameraGeometry?
  private var selecting = false
  private var tapSelectionActive = false
  private var maskSelection: SceneSelection?
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
    frameFeed = CameraFrameFeed(ingress: recorder.ingress)
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
        (overlay as? RealityAnswerOverlay)?.setScanning(true)
        liveSplat?.start()
        installPreview()
      case .stopping:
        captureDuration = Date().timeIntervalSince(recordingStarted ?? Date())
        liveSplat?.stop()
        arView?.renderCallbacks.postProcess = nil
        (overlay as? RealityAnswerOverlay)?.setScanning(false)
      case .saved(let directory):
        savedCapture = directory
        latestCapture = directory
      case .failed:
        failureText = "Scan failed"
        liveSplat?.stop()
        arView?.renderCallbacks.postProcess = nil
        (overlay as? RealityAnswerOverlay)?.setScanning(false)
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

  private func installPreview() {
    guard let renderer = liveSplat as? DefaultLiveSplatRenderer, let view = arView else { return }
    do {
      let processor = try renderer.preparePreview()
      processor.setActive(true, enabled: splatEnabled)
      processor.setViewport(
        view.bounds.size, orientation: view.window?.windowScene?.interfaceOrientation ?? .portrait)
      frameFeed.setPreview(processor)
      view.renderCallbacks.postProcess = { @Sendable context in processor.render(context) }
    } catch {
      failed = true
      failureText = "Preview unavailable"
    }
  }

  /// Called by a 10 Hz UI timer, never from the ARSession delegate.
  func tick() {
    recorder.pollFailure()
    if recording {
      if let renderer = liveSplat as? DefaultLiveSplatRenderer, let view = arView {
        renderer.processor?.setViewport(
          view.bounds.size, orientation: view.window?.windowScene?.interfaceOrientation ?? .portrait
        )
        renderer.pollSummary()
        scanCoverage = renderer.coverageFraction
        if renderer.processor?.unavailable == true {
          failed = true
          failureText = "Preview unavailable"
        } else if failureText == "Preview unavailable" {
          failed = false
        }
      }
      return
    }
    if let frame = frameFeed.takeFrame() { frameForPerception(frame.frame) }
  }
  private var scanCoverage = 0.0
  private func frameForPerception(_ frame: ARFrame) {
    updatePins()
    guard let view = arView, view.bounds.width > 0, view.bounds.height > 0 else { return }
    let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
    let inverse = frame.displayTransform(for: orientation, viewportSize: view.bounds.size)
      .inverted()
    viewportToImage = [inverse.a, inverse.b, inverse.c, inverse.d, inverse.tx, inverse.ty].map {
      Double($0)
    }
    latestCamera = CameraGeometry(camera: frame.camera)
    if let maskSelection, !maskIsCurrent(maskSelection) { overlay?.setMask(nil) }
    guard case .normal = frame.camera.trackingState else {
      overlay?.setMask(nil)
      return
    }
    guard !selecting, perceptionTask == nil, frame.timestamp - lastPerceptionTime >= 0.35 else {
      return
    }
    lastPerceptionTime = frame.timestamp
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
        try Task.checkCancellation()
        guard self?.frameGeneration == frameID else { return }
        self?.hasSnapshot = true
        let selection = try await scene.refreshSelection()
        try Task.checkCancellation()
        if let self, tapSelectionActive {
          presentation.replaceSelection(selection.entities)
          updatePins()
        }
        self?.highlight(selection)
      } catch is CancellationError {} catch { self?.overlay?.setMask(nil) }
    }
  }

  func tap(_ point: CGPoint) {
    guard !capturePhase.isBusy else { return }
    guard let view = arView, view.bounds.width > 0, view.bounds.height > 0, hasSnapshot else {
      return
    }
    let transfer = usesLiveCapture ? view.session.currentFrame.map(FrameTransfer.init) : nil
    let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
    let tapTransform = transfer?.frame.displayTransform(
      for: orientation, viewportSize: view.bounds.size
    ).inverted()
    let t =
      tapTransform.map { [$0.a, $0.b, $0.c, $0.d, $0.tx, $0.ty].map(Double.init) }
      ?? viewportToImage
    let u = Double(point.x / view.bounds.width)
    let v = Double(point.y / view.bounds.height)
    guard let mapped = ViewportMapping(viewportToImage: t).imagePoint(x: u, y: v) else { return }
    let prompt = ImagePoint(x: mapped.x, y: mapped.y)
    answerTask?.cancel()
    let generation = UUID()
    questionGeneration = generation
    card = nil
    overlay?.stopSpeaking()
    selecting = true
    tapSelectionActive = true
    question = ""
    answerText = ""
    answering = false
    perceptionTask?.cancel()
    perceptionTask = nil
    frameGeneration = UUID()
    answerTask = Task {
      defer { if generation == questionGeneration { selecting = false } }
      await router.cancel()
      guard generation == questionGeneration else { return }
      do {
        if let transfer {
          let capturedAt = ProcessInfo.processInfo.systemUptime
          let snapshot = try await Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            return try FrameSnapshot(frame: transfer.frame)
          }.value
          try Task.checkCancellation()
          guard generation == questionGeneration else { return }
          await perception.update(snapshot, viewportToImage: t, capturedAt: capturedAt)
        }
        let previousSelectionCount = presentation.entities.count
        let selected = try await perception.select(prompt)
        guard generation == questionGeneration else { return }
        presentation.replaceSelection(selected.entities)
        overlay?.update(entities: presentation.entities)
        updatePins()
        card = selected.entities.count > previousSelectionCount ? selected.entities.last : nil
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
    if !recording {
      clearAnswer()
      perceptionTask?.cancel()
      frameGeneration = UUID()
      perceptionTask = nil
    }
    recorder.toggleRecording()
  }

  private func toggleFixtureRecording() {
    guard let source = captureSource.fixtureDirectory else { return }
    if recording {
      recording = false
      capturePhase = .stopping
      captureDuration = Date().timeIntervalSince(recordingStarted ?? Date())
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
      let start = Date()
      recordingStarted = start
      capturePhase = .recording(start)
      recording = true
      liveSplat?.start()
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
    selecting = false
    tapSelectionActive = false
    question = ""
    answerText = ""
    answering = false
    overlay?.stopSpeaking()
    overlay?.update(entities: [])
    overlay?.setMask(nil)
    presentation.reset()
    card = nil
    screenPins = []
    maskSelection = nil
    hasSnapshot = false
    perceptionTask?.cancel()
    perceptionTask = nil
    frameGeneration = UUID()
    let router = router
    let perception = perception
    Task {
      await router.cancel()
      await perception.clear()
    }
  }
  func replayAnswer() { if !answerText.isEmpty { overlay?.speak(answerText) } }
  func pinCard() {
    card = nil
  }

  func beginTalking() {
    guard !isTalking else { return }
    isTalking = true
    selecting = false
    tapSelectionActive = false
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
    selecting = false
    tapSelectionActive = false
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
        case .entity:
          overlay?.update(entities: presentation.entities)
          updatePins()
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

  private func maskIsCurrent(_ selection: SceneSelection) -> Bool {
    guard ProcessInfo.processInfo.systemUptime - selection.capturedAt <= 0.75,
      let view = arView, let latestCamera
    else { return simulated }
    return selection.isRegistered(
      camera: latestCamera, viewportToImage: viewportToImage,
      width: Double(view.bounds.width * view.contentScaleFactor),
      height: Double(view.bounds.height * view.contentScaleFactor))
  }

  private func highlight(_ selection: SceneSelection) {
    guard !capturePhase.isBusy else { return }
    maskSelection = selection
    guard maskIsCurrent(selection), let view = arView else {
      overlay?.setMask(nil)
      return
    }
    // Match the full camera viewport, including aspect-fill crop and safe areas.
    let scale = min(
      1,
      1536
        / max(
          view.bounds.width * view.contentScaleFactor,
          view.bounds.height * view.contentScaleFactor))
    let outputWidth = max(1, Int((view.bounds.width * view.contentScaleFactor * scale).rounded()))
    let outputHeight = max(1, Int((view.bounds.height * view.contentScaleFactor * scale).rounded()))
    let masks = selection.masks
    let transform = selection.viewportToImage
    var projected: HighlightMask?
    for mask in masks {
      guard
        let next = HighlightMask.project(
          width: mask.width, height: mask.height,
          pixels: mask.pixels, viewportToImage: transform,
          outputWidth: outputWidth, outputHeight: outputHeight)
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
    selecting = false
    tapSelectionActive = false
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
    frameFeed.setPreview(nil)
    arView?.renderCallbacks.postProcess = nil
    overlay?.remove()
    overlay = nil
    arView = nil
    presentation.reset()
    screenPins = []
    maskSelection = nil
    latestCamera = nil
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
