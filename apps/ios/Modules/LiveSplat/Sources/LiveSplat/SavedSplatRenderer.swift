#if canImport(RealityKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import Metal
  @preconcurrency import RealityKit
  import Foundation
  import UIKit
  import simd

  /// Immutable geometry and pipeline, locked camera snapshots. Like MetalPreview,
  /// RealityKit owns AR compositing and its command buffer: no second AR session,
  /// camera readback, or new world origin. Saved PLY uses keepCrs world metres.
  public final class SavedSplatRenderer: @unchecked Sendable {
    private struct Uniforms {
      var view: simd_float4x4
      var projection: simd_float4x4
      var viewport: SIMD4<Float>
    }
    private let points: any MTLBuffer
    private let standard: any MTLRenderPipelineState
    private let hdr: any MTLRenderPipelineState
    private let count: Int
    private let lock = NSLock()
    private var liveCamera = false
    private var liveMatrices: (simd_float4x4, simd_float4x4)?
    // Shared index storage is reused only after every reader's command completes.
    private struct OrderSlot {
      let buffer: any MTLBuffer
      var readers = 0
    }
    private var orderBuffers: [OrderSlot] = []
    private var orderIndex: Int?
    private var uniforms = Uniforms(view: matrix_identity_float4x4,
      projection: matrix_identity_float4x4, viewport: SIMD4(1, 1, 0, 0))

    public init(scene: SavedSplat) throws {
      guard let device = MTLCreateSystemDefaultDevice() else { throw LiveSplatError.unavailable }
      let gaussianCount = scene.gaussians.count
      count = gaussianCount
      let buffer = scene.gaussians.withUnsafeBytes { bytes in
        device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let buffer else { throw LiveSplatError.unavailable }
      points = buffer
      var slots: [OrderSlot] = []
      for _ in 0..<3 {
        guard let order = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride,
          options: .storageModeShared) else { throw LiveSplatError.unavailable }
        slots.append(OrderSlot(buffer: order))
      }
      orderBuffers = slots
      let library = try device.makeDefaultLibrary(bundle: .module)
      func pipeline(_ format: MTLPixelFormat) throws -> any MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "savedSplatVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "savedSplatFragment")
        let attachment = descriptor.colorAttachments[0]!
        attachment.pixelFormat = format
        attachment.isBlendingEnabled = true
        attachment.sourceRGBBlendFactor = .one
        attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        attachment.sourceAlphaBlendFactor = .one
        attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        return try device.makeRenderPipelineState(descriptor: descriptor)
      }
      standard = try pipeline(.bgra8Unorm)
      hdr = try pipeline(.rgba16Float)
    }

    @discardableResult public func update(view: simd_float4x4, projection: simd_float4x4,
      size: SIMD2<Float>, opacity: Float, indices: Data? = nil) -> Bool {
      lock.withLock {
        uniforms = Uniforms(view: view, projection: projection,
          viewport: SIMD4(max(1,size.x), max(1,size.y), opacity, 0))
        guard let indices else { return true }
        guard indices.count == count * MemoryLayout<UInt32>.stride,
          let slot = orderBuffers.firstIndex(where: { $0.readers == 0 }) else { return false }
        indices.withUnsafeBytes { bytes in
          if let base = bytes.baseAddress {
            orderBuffers[slot].buffer.contents().copyMemory(from: base, byteCount: bytes.count)
          }
        }
        orderIndex = slot
        return true
      }
    }

    /// The ARView fills the screen; its bounds include the overlay's safe areas.
    public func setViewport(_ size: CGSize) {
      lock.withLock {
        let width = Float(max(1, size.width))
        let height = Float(max(1, size.height))
        if uniforms.viewport.x != width || uniforms.viewport.y != height {
          uniforms.viewport.x = width
          uniforms.viewport.y = height
          liveMatrices = nil
        }
      }
    }
    public func setLiveCamera(_ enabled: Bool) {
      lock.withLock { liveCamera = enabled; liveMatrices = nil }
    }
    /// The app is portrait-only. Retain only value matrices, never the ARFrame.
    /// Sorting can lag one request while projection follows every live frame.
    public func offer(_ frame: ARFrame) {
      let size = lock.withLock { uniforms.viewport }
      let viewport = CGSize(width: CGFloat(size.x), height: CGFloat(size.y))
      let view = frame.camera.viewMatrix(for: .portrait)
      let projection = frame.camera.projectionMatrix(for: .portrait, viewportSize: viewport,
        zNear: 0.02, zFar: 100)
      lock.withLock { if liveCamera { liveMatrices = (view, projection) } }
    }

    public func render(to target: any MTLTexture, command: any MTLCommandBuffer, clear: Bool) {
      var snapshot = lock.withLock { () -> (Uniforms, Int?) in
        var values = uniforms
        if let liveMatrices { values.view = liveMatrices.0; values.projection = liveMatrices.1 }
        if let orderIndex { orderBuffers[orderIndex].readers += 1 }
        return (values, orderIndex)
      }
      if let slot = snapshot.1 {
        command.addCompletedHandler { [self] _ in
          lock.withLock { orderBuffers[slot].readers -= 1 }
        }
      }
      let pass = MTLRenderPassDescriptor()
      pass.colorAttachments[0].texture = target
      pass.colorAttachments[0].loadAction = clear ? .clear : .load
      pass.colorAttachments[0].storeAction = .store
      pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.018, green: 0.03, blue: 0.035, alpha: 1)
      guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { return }
      guard let slot = snapshot.1 else { encoder.endEncoding(); return }
      let index = lock.withLock { orderBuffers[slot].buffer }
      encoder.setRenderPipelineState(target.pixelFormat == .rgba16Float ? hdr : standard)
      encoder.setVertexBuffer(points, offset: 0, index: 0)
      encoder.setVertexBuffer(index, offset: 0, index: 1)
      encoder.setVertexBytes(&snapshot.0, length: MemoryLayout<Uniforms>.stride, index: 2)
      encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: count)
      encoder.endEncoding()
    }
    public func composite(_ context: ARView.PostProcessContext) {
      let blit = context.commandBuffer.makeBlitCommandEncoder()
      blit?.copy(from: context.sourceColorTexture, to: context.targetColorTexture)
      blit?.endEncoding()
      render(to: context.targetColorTexture, command: context.commandBuffer, clear: false)
    }
  }
#endif
