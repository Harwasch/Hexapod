#if canImport(ARKit) && canImport(RealityKit) && canImport(UIKit)
  @preconcurrency import ARKit
  import AVFoundation
  import Darwin
  import RealityKit
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
      let coordinator = context.coordinator
      coordinator.view = view
      view.session.delegate = coordinator
      view.session.delegateQueue = .main
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
        return
      }
      coordinator.startIfNeeded()
    }

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }
    static func dismantleUIView(_ view: ARView, coordinator: Coordinator) {
      coordinator.active = false
      coordinator.stop()
      view.session.delegate = nil
    }

    // The session above delivers every delegate callback on the main queue.
    @MainActor final class Coordinator: NSObject, @preconcurrency ARSessionDelegate {
      weak var view: ARView?
      let model: CameraModel
      var active = false
      var running = false
      var interrupted = false
      private var startGeneration = UUID()
      var starting: Task<Void, Never>?
      init(model: CameraModel) { self.model = model }
      func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard running, active, !interrupted else { return }
        model.frame(frame)
      }
      func sessionWasInterrupted(_ session: ARSession) {
        interrupted = true
        stop()
      }
      func sessionInterruptionEnded(_ session: ARSession) {
        interrupted = false
        startIfNeeded()
      }
      func session(_ session: ARSession, didFailWithError error: Error) {
        active = false
        stop()
        model.reportUnavailable()
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
          let configuration = ARWorldTrackingConfiguration()
          configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
          configuration.worldAlignment = .gravity
          self.model.attach(view)
          view.session.run(configuration)
          self.running = true
        }
      }
      @objc func tap(_ gesture: UITapGestureRecognizer) {
        guard let view else { return }
        model.tap(gesture.location(in: view))
      }
      func stop() {
        startGeneration = UUID()
        starting?.cancel()
        starting = nil
        view?.session.pause()
        if running { model.suspend() }
        running = false
      }
    }
  }
#endif
