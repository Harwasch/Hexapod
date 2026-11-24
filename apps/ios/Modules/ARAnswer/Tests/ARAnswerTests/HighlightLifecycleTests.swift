#if canImport(RealityKit) && os(iOS)
  import Metal
  import RealityKit
  import Testing
  @testable import ARAnswer

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
  @MainActor func inactiveHighlightAndScanningUninstallThePostprocess() throws {
    let view = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
    let overlay = try RealityAnswerOverlay(view: view)
    #expect(view.renderCallbacks.postProcess == nil)
    let mask = HighlightMask(width: 2, height: 2, pixels: [255, 0, 0, 255])
    overlay.setMask(mask)
    #expect(view.renderCallbacks.postProcess != nil)
    overlay.setMask(nil)
    #expect(view.renderCallbacks.postProcess == nil)
    overlay.setMask(mask)
    overlay.setScanning(true)
    #expect(view.renderCallbacks.postProcess == nil)
    overlay.setMask(mask)
    #expect(view.renderCallbacks.postProcess == nil)
    overlay.setScanning(false)
    overlay.setMask(mask)
    #expect(view.renderCallbacks.postProcess != nil)
    overlay.remove()
    #expect(view.renderCallbacks.postProcess == nil)
  }
#endif

#if canImport(RealityKit) && os(iOS)
  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
  @MainActor func worldSurfaceRenderingYieldsThePostprocessToScanning() throws {
    let view = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
    let overlay = try RealityAnswerOverlay(view: view)
    overlay.enableWorldHighlight()
    #expect(view.renderCallbacks.postProcess != nil)
    overlay.setScanning(true)
    #expect(view.renderCallbacks.postProcess == nil)
    overlay.enableWorldHighlight()
    #expect(view.renderCallbacks.postProcess == nil)
    overlay.setScanning(false)
    overlay.enableWorldHighlight()
    #expect(view.renderCallbacks.postProcess != nil)
    overlay.remove()
    #expect(view.renderCallbacks.postProcess == nil)
  }
#endif
