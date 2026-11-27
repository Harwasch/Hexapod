import ARAnswer
import ARKit
import Capture
import EntitySchema
import Foundation
import LiveSplat
import Observation
import Perception
import RealityKit
import Router
import UIKit
import Voice
import simd

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
  private(set) var recording = false {
    didSet { if recording != oldValue { scanGuidance = ScanGuidance() } }
  }
  private(set) var scanGuidance = ScanGuidance()
  /// The slow-down cue while recording: predicted motion blur (CaptureCamera, MotionBlurMeter).
  private(set) var motionLevel = MotionBlurMeter.Level.steady
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
  private var previewTask: Task<Void, Never>?
  var usesLiveCapture: Bool { captureSource.usesLiveSession }
  var canRecord: Bool { usesLiveCapture || (simulated && captureSource.fixtureDirectory != nil) }
  struct ScreenPin: Identifiable {
    let id: Int
    let x: CGFloat
    let y: CGFloat
  }
  private(set) var screenPins: [ScreenPin] = []
  private(set) var previewGeneration = 0
  var previewSize: CGSize = .zero
  private func updatePins() {
    if LaunchMode.current.isUITest && simulated {
      screenPins = presentation.entities.indices.map {
        ScreenPin(id: $0, x: previewSize.width / 2, y: previewSize.height * 0.55)
      }
      return
    }
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
  private var splatWorld = SplatWorldOrigin()
  private(set) var trainingBusy = false
  private var trainingNeedsInPlace = false
  var keepsCameraSession: Bool {
    TrainingCameraPolicy.keepSession(trainingBusy: trainingBusy, needsInPlace: trainingNeedsInPlace)
  }
  private var perceptionEnabled: Bool {
    TrainingCameraPolicy.perceptionEnabled(trainingBusy: trainingBusy,
      captureBusy: capturePhase.isBusy, talking: isTalking, answering: answering)
  }
  func setTrainingBusy(_ busy: Bool, capture: URL) {
    if busy && !trainingBusy { trainingNeedsInPlace = canPlaceSplat(capture: capture) }
    trainingBusy = busy
    frameFeed.setTrainingBusy(busy)
    perceptionDriver?.setEnabled(perceptionEnabled)
    if !busy { trainingNeedsInPlace = false }
  }
  func invalidateSplatOrigin() {
    splatWorld.invalidate()
    trainingNeedsInPlace = false
  }
  func canPlaceSplat(capture: URL) -> Bool {
    guard !LaunchMode.current.isUITest, usesLiveCapture, cameraReady, let view = arView, view.session.currentFrame != nil else { return false }
    return splatWorld.permits(capture: capture, session: ObjectIdentifier(view.session))
  }
  func trainedSplatCamera(capture: URL) -> (view: simd_float4x4, projection: simd_float4x4, size: CGSize)? {
    guard canPlaceSplat(capture: capture), let view = arView,
      let frame = view.session.currentFrame else { return nil }
    let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
    return (frame.camera.viewMatrix(for: orientation),
      frame.camera.projectionMatrix(for: orientation, viewportSize: view.bounds.size, zNear: 0.02, zFar: 100), view.bounds.size)
  }
  func showTrainedSplat(_ renderer: SavedSplatRenderer?) {
    guard let view = arView else { frameFeed.setTrainedPreview(nil); return }
    renderer?.setViewport(view.bounds.size)
    frameFeed.setTrainedPreview(renderer)
    if let renderer {
      view.renderCallbacks.postProcess = { @Sendable context in renderer.composite(context) }
    } else { view.renderCallbacks.postProcess = nil }
  }
  private var voiceTask: Task<Void, Never>?
  private var answerTask: Task<Void, Never>?
  private var questionGeneration = UUID()
  private var voiceGeneration = UUID()
  private var perceptionDriver: CameraPerceptionDriver?
  private var tapSelectionActive = false
  private(set) var perceptionMetrics = InferenceMetrics()
  private var selectionFeedback = CameraSelectionFeedback()
  private(set) var pipelineDiagnostics = "Metal self-check not run"
  var selectionDiagnostics: String {
    "Last refusal: \(selectionFeedback.lastRefusal?.description ?? "none")"
  }
  private func receiveTap(_ result: TapAdmissionResult, log: Bool = true) {
    if log, case .refused(let reason) = result {
      CameraDiagnostics.record("Selection refused: \(reason)")
    }
    let hadBanner = selectionFeedback.banner != nil
    selectionFeedback.receive(result)
    if let banner = selectionFeedback.banner {
      failureText = banner
      failed = true
    } else if hadBanner || failureText == "Selection unavailable" {
      failed = false
    }
  }
  var perceptionPerformance: String {
    if !perceptionEnabled { return "Vision paused" }
    return String(
      format: "Vision %.1f Hz · %.0f ms · target ≥%.0f Hz",
      perceptionMetrics.achievedHz, perceptionMetrics.averageDurationMilliseconds,
      PerceptionLimits.targetInferenceHz)
  }

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
        perceptionDriver?.setEnabled(false)
        recordingStarted = start
        (overlay as? RealityAnswerOverlay)?.setScanning(true)
        liveSplat?.start()
        installPreview()
      case .stopping:
        captureDuration = LaunchMode.current.date().timeIntervalSince(
          recordingStarted ?? LaunchMode.current.date())
        liveSplat?.stop()
        arView?.renderCallbacks.postProcess = nil
        (overlay as? RealityAnswerOverlay)?.setScanning(false)
      case .saved(let directory):
        savedCapture = directory
        if usesLiveCapture, cameraReady, let view = arView {
          splatWorld.record(capture: directory, session: ObjectIdentifier(view.session))
        } else { splatWorld.invalidate() }
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
    guard !LaunchMode.current.isUITest else { return }
    arView = view
    cameraReady = true
    overlay?.remove()
    let attachment = CameraAttachment.attach(
      makeOverlay: { () throws -> any AnswerOverlaySink in
        if fakeOverlay { return FakeOverlaySink() }
        return try RealityAnswerOverlay(view: view)
      },
      makeDriver: { overlay -> CameraPerceptionDriver? in
        guard usesLiveCapture else { return nil }
        let renderer = (overlay as? RealityAnswerOverlay)?.worldSurfaceRenderer
        let driver =
          perceptionDriver ?? CameraPerceptionDriver(scene: perception, renderer: renderer)
        driver.setRenderer(renderer)
        driver.setEnabled(perceptionEnabled)
        return driver
      })
    overlay = attachment.overlay
    perceptionDriver = attachment.driver
    if let reality = overlay as? RealityAnswerOverlay {
      pipelineDiagnostics = reality.pipelineChecks.map(\.description).joined(separator: "\n")
    } else if let error = attachment.overlayError {
      pipelineDiagnostics = "Overlay failed: \(error)"
      CameraDiagnostics.record(pipelineDiagnostics)
    } else {
      pipelineDiagnostics = "Metal self-check skipped: simulated overlay"
    }
    frameFeed.setPerceptionDriver(perceptionDriver)
    (overlay as? RealityAnswerOverlay)?.enableWorldHighlight()
    displayTick()
    failed = false
  }

  func beginUITestPreview(size: CGSize) {
    previewSize = size
    guard LaunchMode.current.isUITest, !usesLiveCapture else { return }
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
          cameraReady = true
          let entities = try await perception.inspect(.init(subject: "subject", tools: [.count]))
          overlay?.update(entities: entities)
          highlight(await perception.selection())
          if LaunchMode.current.isUITest { break }
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
    guard !LaunchMode.current.isUITest else { return }
    recorder.pollFailure()
    let level = recording && usesLiveCapture ? frameFeed.camera.motionLevel : .steady
    if level != motionLevel { motionLevel = level }
    if let driver = perceptionDriver {
      let enabled = perceptionEnabled
      driver.setEnabled(enabled)
      if enabled { (overlay as? RealityAnswerOverlay)?.enableWorldHighlight() }
      perceptionMetrics = driver.metrics
      for event in driver.takeTapEvents() { receiveTap(event, log: false) }
      if let result = driver.takeResult(), driver.isCurrent(result.generation), perceptionEnabled
      {
        if tapSelectionActive {
          presentation.replaceSelection(result.selection.entities)
          overlay?.update(entities: presentation.entities)
          updatePins()
          highlight(result.selection)
        }
        if result.tapped {
          card = result.card
        }
      }
    }
    updatePins()
    if recording {
      if let renderer = liveSplat as? DefaultLiveSplatRenderer, let view = arView {
        renderer.processor?.setViewport(
          view.bounds.size, orientation: view.window?.windowScene?.interfaceOrientation ?? .portrait
        )
        renderer.pollSummary()
        scanCoverage = renderer.coverageFraction
        let blur: Double = motionLevel == .steady ? 0 : motionLevel == .slowDown ? 1 : 2
        let completed = scanGuidance.update(
          renderer.processor?.guidanceSample, blur: blur,
          now: ProcessInfo.processInfo.systemUptime)
        if completed != 0 {
          UIImpactFeedbackGenerator(style: .soft).impactOccurred(intensity: 0.45)
        }
        if renderer.processor?.unavailable == true {
          failed = true
          failureText = "Preview unavailable"
        } else if failureText == "Preview unavailable" {
          failed = false
        }
      }
      return
    }
  }
  private var scanCoverage = 0.0

  /// Lightweight camera uniforms only. Surface rasterization happens on the GPU each display frame.
  func displayTick() {
    guard !LaunchMode.current.isUITest, !trainingBusy else { return }
    guard let view = arView, let driver = perceptionDriver else { return }
    driver.setViewport(
      size: view.bounds.size,
      orientation: view.window?.windowScene?.interfaceOrientation ?? .portrait)
    if let frame = view.session.currentFrame { driver.display(frame) }
  }

  func tap(_ point: CGPoint) {
    guard !capturePhase.isBusy, !trainingBusy else {
      receiveTap(.refused(.disabled))
      return
    }
    guard let view = arView else {
      receiveTap(.refused(.noDriver))
      return
    }
    guard view.bounds.width > 0, view.bounds.height > 0 else {
      receiveTap(.refused(.noViewport))
      return
    }
    card = nil
    answerTask?.cancel()
    questionGeneration = UUID()
    question = ""
    answerText = ""
    answering = false
    tapSelectionActive = true
    overlay?.stopSpeaking()
    let router = router
    Task { await router.cancel() }
    if let driver = perceptionDriver {
      driver.setEnabled(perceptionEnabled)
      let outcome = driver.select(
        ImagePoint(
          x: Double(point.x / view.bounds.width),
          y: Double(point.y / view.bounds.height)), frame: view.session.currentFrame)
      receiveTap(outcome)
      return
    }
    if usesLiveCapture {
      receiveTap(.refused(.noDriver))
      return
    }
    // Explicit simulated replay has no ARFrame or live camera projection.
    let generation = questionGeneration
    answerTask = Task {
      do {
        let previousCount = presentation.entities.count
        let selection = try await perception.select(
          ImagePoint(
            x: Double(point.x / view.bounds.width),
            y: Double(point.y / view.bounds.height)))
        guard generation == questionGeneration, !Task.isCancelled else { return }
        presentation.replaceSelection(selection.entities)
        updatePins()
        card = selection.entities.count > previousCount ? selection.entities.last : nil
        highlight(selection)
        receiveTap(.accepted)
      } catch is CancellationError {} catch {
        receiveTap(.refused(.inferenceFailed(String(describing: error))))
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
    if !LaunchMode.current.isUITest {
      UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }
    if !recording {
      clearAnswer()
    }
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
      let destination = LaunchMode.current.captureRoot.appendingPathComponent(UUID().uuidString)
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
        if let reference = renderer as? DefaultLiveSplatRenderer {
          scanGuidance = FixtureSplatPreview.guidance(renderer: reference)
        }
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
    let router = router
    let perception = perception
    let driver = perceptionDriver
    driver?.resetSelection(enabled: perceptionEnabled)
    Task {
      await router.cancel()
      if driver == nil { await perception.clear() }
    }
  }
  func replayAnswer() { if !answerText.isEmpty { overlay?.speak(answerText) } }
  func pinCard() {
    card = nil
  }

  func beginTalking() {
    guard !isTalking else { return }
    isTalking = true
    perceptionDriver?.setEnabled(false)
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
    perceptionDriver?.setEnabled(false)
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
      await perceptionDriver?.waitUntilIdle()
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

  private func highlight(_ selection: SceneSelection) {
    // Use flat projected masks when the world outline renderer could not be built.
    guard (overlay as? RealityAnswerOverlay)?.worldSurfaceRenderer == nil,
      !capturePhase.isBusy, let view = arView
    else { return }
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
    previewTask?.cancel()
    previewTask = nil
    frameFeed.setTrainedPreview(nil)
    perceptionDriver?.resetSelection(enabled: false)
    frameFeed.setPerceptionDriver(nil)
    let stoppedDriver = perceptionDriver
    replayTask?.cancel()
    replayTask = nil
    tapSelectionActive = false
    cameraReady = false
    isTalking = false
    preparingVoice = false
    voiceGeneration = UUID()
    questionGeneration = UUID()
    voiceTask?.cancel()
    voice.cancel()
    answerTask?.cancel()
    let router = router
    let perception = perception
    Task {
      await router.cancel()
      if stoppedDriver == nil { await perception.clear() }
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
    card = nil
    answering = false
  }
}
