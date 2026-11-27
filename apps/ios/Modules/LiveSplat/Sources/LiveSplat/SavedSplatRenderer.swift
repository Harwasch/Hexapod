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
    private let device: any MTLDevice
    private let points: any MTLBuffer
    private let standard: any MTLRenderPipelineState
    private let hdr: any MTLRenderPipelineState
    private let count: Int
    private let lock = NSLock()
    private var liveCamera = false
    private var liveMatrices: (simd_float4x4, simd_float4x4)?
    private var order: (any MTLBuffer)?
    private var uniforms = Uniforms(view: matrix_identity_float4x4,
      projection: matrix_identity_float4x4, viewport: SIMD4(1, 1, 0, 0))

    public init(scene: SavedSplat) throws {
      guard let device = MTLCreateSystemDefaultDevice() else { throw LiveSplatError.unavailable }
      self.device = device
      count = scene.gaussians.count
      let buffer = scene.gaussians.withUnsafeBytes { bytes in
        device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let buffer else { throw LiveSplatError.unavailable }
      points = buffer
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

    public func update(view: simd_float4x4, projection: simd_float4x4,
      size: SIMD2<Float>, opacity: Float, indices: Data) {
      let next = indices.withUnsafeBytes { bytes -> (any MTLBuffer)? in
        guard let base = bytes.baseAddress, bytes.count == count * 4 else { return nil }
        return device.makeBuffer(bytes: base, length: bytes.count, options: .storageModeShared)
      }
      lock.withLock {
        order = next
        uniforms = Uniforms(view: view, projection: projection,
          viewport: SIMD4(max(1,size.x), max(1,size.y), opacity, 0))
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
      var snapshot = lock.withLock { () -> (Uniforms, (any MTLBuffer)?) in
        var values = uniforms
        if let liveMatrices { values.view = liveMatrices.0; values.projection = liveMatrices.1 }
        return (values, order)
      }
      let pass = MTLRenderPassDescriptor()
      pass.colorAttachments[0].texture = target
      pass.colorAttachments[0].loadAction = clear ? .clear : .load
      pass.colorAttachments[0].storeAction = .store
      pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.018, green: 0.03, blue: 0.035, alpha: 1)
      guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { return }
      guard let index = snapshot.1 else { encoder.endEncoding(); return }
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
