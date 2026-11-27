#if canImport(ARKit) && canImport(RealityKit) && canImport(UIKit)
  @preconcurrency import ARKit
  import AVFoundation
  import Darwin
  import RealityKit
  import Perception
  import LiveSplat
  import SwiftUI
  import UIKit

  @MainActor struct ARCameraView: UIViewRepresentable {
    var isActive: Bool
    var model: CameraModel

    static var supportsDevice: Bool {
      var system = utsname()
      uname(&system)
      let identifier = withUnsafeBytes(of: &system.machine) { bytes in
        String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
      }
      guard identifier.hasPrefix("iPhone"),
        let generation = Int(identifier.dropFirst(6).split(separator: ",")[0]),
        generation >= 16
      else { return false }
      return ARWorldTrackingConfiguration.isSupported
        && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    func makeUIView(context: Context) -> ARView {
      let view = ARView(
        frame: .zero, cameraMode: model.usesLiveCapture ? .ar : .nonAR,
        automaticallyConfigureSession: false)
      view.debugOptions = []
      let coordinator = context.coordinator
      coordinator.view = view
      view.session.delegate = model.frameFeed
      view.session.delegateQueue = model.frameFeed.queue
      view.addGestureRecognizer(
        UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.tap(_:))))
      return view
    }

    func updateUIView(_ view: ARView, context: Context) {
      let coordinator = context.coordinator
      if !model.usesLiveCapture {
        if isActive {
          if !coordinator.running {
            model.attach(view)
            model.beginReplay()
            coordinator.running = true
          }
        } else {
          coordinator.stop()
        }
        return
      }
      coordinator.active = isActive && Self.supportsDevice
      guard coordinator.active else {
        coordinator.stop()
        if isActive { model.reportUnavailable() }
        return
      }
      coordinator.startIfNeeded()
    }

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }
    static func dismantleUIView(_ view: ARView, coordinator: Coordinator) {
      coordinator.active = false
      coordinator.model.invalidateSplatOrigin()
      coordinator.stop()
      view.session.delegate = nil
    }

    @MainActor final class Coordinator: NSObject {
      weak var view: ARView?
      let model: CameraModel
      var active = false
      var running = false
      var interrupted = false
      private var startGeneration = UUID()
      var starting: Task<Void, Never>?
      private var summaryTimer: Timer?
      private var displayLink: CADisplayLink?
      init(model: CameraModel) {
        self.model = model
        super.init()
        model.frameFeed.camera.onFourKRejected = { [weak self] in
          Task { @MainActor [weak self] in self?.restartWithoutFourK() }
        }
        model.frameFeed.onEvent = { [weak self] event in
          Task { @MainActor [weak self] in
            guard let self else { return }
            switch event {
            case .interrupted:
              self.model.invalidateSplatOrigin()
              self.interrupted = true
              self.stop()
            case .resumed:
              self.interrupted = false
              self.startIfNeeded()
            case .failed:
              self.model.invalidateSplatOrigin()
              self.active = false
              self.stop()
              self.model.reportUnavailable()
            }
          }
        }
      }
      func startIfNeeded() {
        guard active, !interrupted, !running, starting == nil, let view else { return }
        let id = UUID()
        startGeneration = id
        starting = Task { @MainActor [weak self] in
          guard let self else { return }
          defer { if self.startGeneration == id { self.starting = nil } }
          let allowed = await AVCaptureDevice.requestAccess(for: .video)
          guard self.startGeneration == id, !Task.isCancelled else { return }
          guard allowed else {
            self.model.reportUnavailable()
            return
          }
          guard self.active, !self.interrupted else { return }
          let configuration = self.configuration(fourK: true)
          self.model.attach(view)
          view.session.run(configuration)
          self.running = true
          let displayLink = CADisplayLink(target: self, selector: #selector(self.displayFrame))
          displayLink.preferredFrameRateRange = CAFrameRateRange(
            minimum: Float(PerceptionLimits.displayFramesPerSecond),
            maximum: Float(PerceptionLimits.displayFramesPerSecond),
            preferred: Float(PerceptionLimits.displayFramesPerSecond))
          displayLink.add(to: .main, forMode: .common)
          self.displayLink = displayLink
          self.summaryTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) {
            [weak self] _ in
            Task { @MainActor [weak self] in self?.model.tick() }
          }
        }
      }
      /// The scan session: gravity-aligned world tracking with raw and smoothed LiDAR depth,
      /// in ARKit's 4K format unless `fourK` is false or this device ruled it out.
      private func configuration(fourK: Bool) -> ARWorldTrackingConfiguration {
        Self.configuration(fourK: fourK, camera: model.frameFeed.camera)
      }
      private static func configuration(fourK: Bool, camera: CaptureCamera)
        -> ARWorldTrackingConfiguration
      {
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        configuration.worldAlignment = .gravity
        var kind = "default"
        if fourK {
          let (format, chosen) = CaptureCamera.videoFormat()
          if let format { configuration.videoFormat = format }
          kind = chosen
        }
        camera.configure(configuration, kind: kind)
        return configuration
      }
      /// The first frames showed 4K without matching depth: run the default format instead.
      func restartWithoutFourK() {
        model.invalidateSplatOrigin()
        guard running, let view else { return }
        view.session.run(configuration(fourK: false), options: [.resetTracking])
      }
      @objc private func displayFrame() { model.displayTick() }
      @objc func tap(_ gesture: UITapGestureRecognizer) {
        guard let view else { return }
        model.tap(gesture.location(in: view))
      }
      func stop() {
        displayLink?.invalidate()
        displayLink = nil
        summaryTimer?.invalidate()
        summaryTimer = nil
        startGeneration = UUID()
        starting?.cancel()
        starting = nil
        view?.session.pause()
        if running { model.suspend() }
        running = false
      }
    }
  }

  /// Serial delegate ingress: fixed-cost admission and two bounded retained-frame mailboxes.
  final class CameraFrameFeed: NSObject, ARSessionDelegate, @unchecked Sendable {
    enum Event: Sendable { case interrupted, resumed, failed }
    let queue = DispatchQueue(label: "hex.scan.admission", qos: .userInitiated)
    private let ingress: ScanIngress
    private let lock = NSLock()
    private var preview: ScanPreviewProcessor?
    private var trainedPreview: SavedSplatRenderer?
    private var perceptionDriver: CameraPerceptionDriver?
    private var eventHandler: (@Sendable (Event) -> Void)?
    var onEvent: (@Sendable (Event) -> Void)? {
      get { lock.withLock { eventHandler } }
      set { lock.withLock { eventHandler = newValue } }
    }
    init(ingress: ScanIngress) { self.ingress = ingress }
    func setPreview(_ processor: ScanPreviewProcessor?) {
      lock.withLock {
        preview = processor
      }
    }
    func setTrainedPreview(_ processor: SavedSplatRenderer?) {
      let previous = lock.withLock {
        let previous = trainedPreview
        trainedPreview = processor
        return previous
      }
      if previous !== processor {
        previous?.setLiveCamera(false)
        processor?.setLiveCamera(true)
      }
    }
    func setPerceptionDriver(_ driver: CameraPerceptionDriver?) {
      lock.withLock { perceptionDriver = driver }
    }
    var camera: CaptureCamera { ingress.camera }
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
      ingress.camera.observe(frame)
      ingress.offer(frame)
      let consumers = lock.withLock { (preview, perceptionDriver, trainedPreview) }
      consumers.0?.offer(frame)
      consumers.1?.offer(frame)
      consumers.2?.offer(frame)
    }
    func sessionWasInterrupted(_ session: ARSession) {
      ingress.stop()
      onEvent?(.interrupted)
    }
    func sessionInterruptionEnded(_ session: ARSession) { onEvent?(.resumed) }
    func session(_ session: ARSession, didFailWithError error: Error) {
      ingress.stop()
      onEvent?(.failed)
    }
  }
#endif
