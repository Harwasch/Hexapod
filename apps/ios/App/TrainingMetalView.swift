#if canImport(Msplat) && os(iOS)
  import MetalKit
  import Msplat
  import LiveSplat
  import SwiftUI

  @MainActor struct TrainingMetalView: UIViewRepresentable {
    let surface: MetalPreviewSurface?
    let renderer: SavedSplatRenderer?
    var paused = false
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> MTKView {
      let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
      view.colorPixelFormat = .bgra8Unorm
      view.framebufferOnly = false
      view.preferredFramesPerSecond = 30
      view.clearColor = MTLClearColor(red: 0.018, green: 0.03, blue: 0.035, alpha: 1)
      context.coordinator.configure(view)
      view.delegate = context.coordinator
      return view
    }
    func updateUIView(_ view: MTKView, context: Context) {
      view.isPaused = paused
      context.coordinator.surface = surface
      context.coordinator.renderer = renderer
      if let surface {
        view.autoResizeDrawable = false
        view.drawableSize = CGSize(width: CGFloat(surface.width), height: CGFloat(surface.height))
      } else { view.autoResizeDrawable = true }
    }
    static func dismantleUIView(_ view: MTKView, coordinator: Coordinator) {
      view.isPaused = true
      view.delegate = nil
      coordinator.surface = nil
      coordinator.renderer = nil
    }
    @MainActor final class Coordinator: NSObject, @preconcurrency MTKViewDelegate {
      var surface: MetalPreviewSurface?
      var renderer: SavedSplatRenderer?
      private var inFlight = false
      private var queue: (any MTLCommandQueue)?
      func configure(_ view: MTKView) { queue = view.device?.makeCommandQueue() }
      func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
      func draw(in view: MTKView) {
        guard !inFlight, let drawable = view.currentDrawable,
          let command = queue?.makeCommandBuffer() else { return }
        if let surface {
          guard drawable.texture.width == surface.width,
            drawable.texture.height == surface.height else { return }
          let blit = command.makeBlitCommandEncoder()
          blit?.copy(from: surface.texture, to: drawable.texture)
          blit?.endEncoding()
          command.addCompletedHandler { [surface] _ in _ = surface.width }
        } else if let renderer {
          renderer.render(to: drawable.texture, command: command, clear: true)
        } else if let pass = view.currentRenderPassDescriptor {
          let encoder = command.makeRenderCommandEncoder(descriptor: pass)
          encoder?.endEncoding()
        }
        inFlight = true
        command.addCompletedHandler { [weak self] _ in
          Task { @MainActor [weak self] in self?.inFlight = false }
        }
        command.present(drawable)
        command.commit()
      }
    }
  }
#endif
