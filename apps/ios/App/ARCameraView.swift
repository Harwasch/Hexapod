#if canImport(ARKit) && canImport(RealityKit) && canImport(UIKit)
import ARKit
import Darwin
import RealityKit
import SwiftUI

@MainActor
struct ARCameraView: UIViewRepresentable {
    var isActive: Bool

    // ARKit alone does not exclude older phones. Pair LiDAR support with the
    // iPhone 15 Pro hardware generation (iPhone16,1) or newer.
    static var supportsDevice: Bool {
        var system = utsname()
        uname(&system)
        let identifier = withUnsafeBytes(of: &system.machine) { bytes in
            String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        guard identifier.hasPrefix("iPhone"),
              let generation = Int(identifier.dropFirst(6).split(separator: ",")[0]),
              generation >= 16 else { return false }
        return ARWorldTrackingConfiguration.isSupported
            && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    func makeUIView(context: Context) -> ARView {
        ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    }

    func updateUIView(_ view: ARView, context: Context) {
        guard isActive && Self.supportsDevice else {
            view.session.pause()
            context.coordinator.isRunning = false
            return
        }
        guard !context.coordinator.isRunning else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics.insert(.sceneDepth)
        view.session.run(configuration)
        context.coordinator.isRunning = true
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    static func dismantleUIView(_ view: ARView, coordinator: Coordinator) {
        view.session.pause()
        coordinator.isRunning = false
    }

    final class Coordinator {
        var isRunning = false
    }
}
#endif
