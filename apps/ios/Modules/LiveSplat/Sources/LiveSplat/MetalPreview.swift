#if canImport(RealityKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import Metal
  @preconcurrency import RealityKit
  import CoreVideo
  import Foundation
  import UIKit
  import simd

  /// RealityKit owns the only display loop and command queue. One preview submission
  /// may be outstanding; a busy GPU receives the camera unchanged, never a queued frame.
  /// Mailbox/control/summary state is locked; GPU resources belong to the render callback.
  public final class ScanPreviewProcessor: @unchecked Sendable {
    private let lock = NSLock()
    private var input = PreviewInput()
    private var busy = false
    private var fraction = 0.0
    private var quality = PreviewQuality()
    private var failure = false
    public var coverageFraction: Double { lock.withLock { fraction } }
    public var unavailable: Bool { lock.withLock { failure } }
    private let device: MTLDevice
    private let cache: CVMetalTextureCache
    private let cells: MTLBuffer, locks: MTLBuffer, counters: MTLBuffer
    private let fuse: MTLComputePipelineState, composite: MTLComputePipelineState,
      coverage: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private var color: MTLTexture?, depth: MTLTexture?
    private var generation: UInt64 = .max
    private var lastFusion = -Double.infinity
    private var epoch: UInt32 = 0

    public init() throws {
      guard let device = MTLCreateSystemDefaultDevice() else { throw LiveSplatError.unavailable }
      self.device = device
      var cache: CVMetalTextureCache?
      guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess,
        let cache,
        let cells = device.makeBuffer(
          length: SurfelHash.capacity * MemoryLayout<SurfelGPUCell>.stride,
          options: .storageModePrivate),
        let locks = device.makeBuffer(
          length: SurfelHash.capacity / 4 * 4, options: .storageModePrivate),
        let counters = device.makeBuffer(length: 8, options: .storageModeShared)
      else { throw LiveSplatError.unavailable }
      self.cache = cache
      self.cells = cells
      self.locks = locks
      self.counters = counters
      let library = try device.makeDefaultLibrary(bundle: .module)
      func function(_ name: String) throws -> MTLFunction {
        guard let value = library.makeFunction(name: name) else { throw LiveSplatError.unavailable }
        return value
      }
      fuse = try device.makeComputePipelineState(function: function("fuseSurfels"))
      coverage = try device.makeComputePipelineState(function: function("coverageScanPreview"))
      composite = try device.makeComputePipelineState(function: function("compositeScanPreview"))
      let descriptor = MTLRenderPipelineDescriptor()
      descriptor.vertexFunction = try function("surfelVertex")
      descriptor.fragmentFunction = try function("surfelFragment")
      descriptor.colorAttachments[0].pixelFormat = .rgba16Float
      descriptor.colorAttachments[0].isBlendingEnabled = true
      descriptor.colorAttachments[0].sourceRGBBlendFactor = .one
      descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
      descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
      descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
      descriptor.depthAttachmentPixelFormat = .depth32Float
      renderPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
      let depthDescriptor = MTLDepthStencilDescriptor()
      depthDescriptor.depthCompareFunction = .less
      depthDescriptor.isDepthWriteEnabled = true
      guard let state = device.makeDepthStencilState(descriptor: depthDescriptor) else {
        throw LiveSplatError.unavailable
      }
      depthState = state
    }
    public func offer(_ frame: ARFrame) {
      lock.withLock { if input.active { input.frame = PreviewFrame(frame) } }
    }
    public func setActive(_ active: Bool, enabled: Bool) {
      lock.withLock {
        input.active = active
        input.enabled = enabled
        if !active {
          input.frame = nil
          fraction = 0
        }
      }
    }
    public func reset() {
      lock.withLock {
        input.generation &+= 1
        input.frame = nil
        fraction = 0
        quality = PreviewQuality()
      }
    }
    public func setViewport(_ size: CGSize, orientation: UIInterfaceOrientation) {
      lock.withLock {
        input.viewport = size
        input.orientation = orientation
      }
    }
    private func texture(_ buffer: CVPixelBuffer, format: MTLPixelFormat, plane: Int = 0) throws
      -> CVMetalTexture
    {
      let planar = CVPixelBufferIsPlanar(buffer)
      let w = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
      let h = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
      var value: CVMetalTexture?
      guard
        CVMetalTextureCacheCreateTextureFromImage(
          nil, cache, buffer, nil, format, w, h, plane, &value) == kCVReturnSuccess,
        let value, CVMetalTextureGetTexture(value) != nil
      else { throw LiveSplatError.unavailable }
      return value
    }
    private func targets(width: Int, height: Int) throws {
      guard color?.width != width || color?.height != height else { return }
      func make(_ format: MTLPixelFormat, usage: MTLTextureUsage) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
          pixelFormat: format, width: width, height: height, mipmapped: false)
        d.storageMode = .private
        d.usage = usage
        guard let result = device.makeTexture(descriptor: d) else {
          throw LiveSplatError.unavailable
        }
        return result
      }
      color = try make(.rgba16Float, usage: [.renderTarget, .shaderRead, .shaderWrite])
      depth = try make(.depth32Float, usage: .renderTarget)
    }
    private func copyCamera(_ context: ARView.PostProcessContext) {
      let blit = context.commandBuffer.makeBlitCommandEncoder()
      blit?.copy(from: context.sourceColorTexture, to: context.targetColorTexture)
      blit?.endEncoding()
    }
    public func render(_ context: ARView.PostProcessContext) {
      let state: (PreviewInput, PreviewQuality)? = lock.withLock {
        guard !busy, input.active, input.frame != nil else { return nil }
        busy = true
        return (input, quality)
      }
      guard let (snapshot, quality) = state, let transfer = snapshot.frame else {
        copyCamera(context)
        return
      }
      let frame = transfer.frame
      guard case .normal = frame.camera.trackingState,
        let sceneDepth = frame.sceneDepth, let confidence = sceneDepth.confidenceMap,
        snapshot.viewport.width > 0, snapshot.viewport.height > 0
      else {
        lock.withLock {
          busy = false
          fraction = 0
        }
        copyCamera(context)
        return
      }
      var retained: [CVMetalTexture] = []
      var fused = false
      var encodingFailed = false
      do {
        let map = sceneDepth.depthMap
        let rgb = frame.capturedImage
        let rgbFormat = CVPixelBufferGetPixelFormatType(rgb)
        guard
          rgbFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            || rgbFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
          CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
          CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8,
          CVPixelBufferGetPlaneCount(rgb) == 2
        else { throw LiveSplatError.invalidFrame }
        retained = try [
          texture(map, format: .r32Float), texture(confidence, format: .r8Unorm),
          texture(rgb, format: .r8Unorm, plane: 0), texture(rgb, format: .rg8Unorm, plane: 1),
        ]
        let textures = retained.map { CVMetalTextureGetTexture($0)! }
        let target = context.targetColorTexture
        let scale = min(1, Double(quality.longestEdge) / Double(max(target.width, target.height)))
        let width = max(1, Int(Double(target.width) * scale))
        let height = max(1, Int(Double(target.height) * scale))
        try targets(width: width, height: height)
        if generation != snapshot.generation {
          guard let blit = context.commandBuffer.makeBlitCommandEncoder() else {
            throw LiveSplatError.unavailable
          }
          blit.fill(buffer: cells, range: 0..<cells.length, value: 0)
          blit.fill(buffer: locks, range: 0..<locks.length, value: 0)
          blit.endEncoding()
          generation = snapshot.generation
          epoch = 0
          lastFusion = -.infinity
        }
        let t = frame.displayTransform(for: snapshot.orientation, viewportSize: snapshot.viewport)
        let inverse = t.inverted()
        let k = frame.camera.intrinsics
        let iw = Float(CVPixelBufferGetWidth(rgb))
        let ih = Float(CVPixelBufferGetHeight(rgb))
        let dw = textures[0].width
        let dh = textures[0].height
        guard dw > 0, dh > 0, dw <= 512, dh <= 512,
          textures[1].width == dw, textures[1].height == dh
        else { throw LiveSplatError.invalidFrame }
        let sampling = SurfelSampling(width: dw, height: dh)
        let stride = sampling.stride
        let sampleCount = sampling.count
        let matrix = CVBufferCopyAttachment(rgb, kCVImageBufferYCbCrMatrixKey, nil) as? String
        var uniforms = PreviewUniforms(
          world: frame.camera.transform, view: frame.camera.transform.inverse,
          intrinsics: SIMD4(k[0][0] / iw, k[1][1] / ih, k[2][0] / iw, k[2][1] / ih),
          displayX: SIMD4(Float(t.a), Float(t.c), Float(t.tx), 0),
          displayY: SIMD4(Float(t.b), Float(t.d), Float(t.ty), 0),
          inverseX: SIMD4(Float(inverse.a), Float(inverse.c), Float(inverse.tx), 0),
          inverseY: SIMD4(Float(inverse.b), Float(inverse.d), Float(inverse.ty), 0),
          dimensions: SIMD4(Float(width), Float(height), Float(dw), Float(dh)),
          settings: SIMD4(UInt32(sampleCount), epoch, UInt32(stride), snapshot.enabled ? 1 : 0),
          flags: SIMD4(
            UInt32(quality.visibleStride),
            CVPixelBufferGetPixelFormatType(rgb) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
              ? 1 : 0,
            matrix == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String) ? 1 : 0, 0))
        if frame.timestamp - lastFusion >= quality.fusionInterval {
          epoch &+= 1
          uniforms.settings.y = epoch
          guard let blit = context.commandBuffer.makeBlitCommandEncoder() else {
            throw LiveSplatError.unavailable
          }
          blit.fill(buffer: counters, range: 0..<8, value: 0)
          blit.fill(buffer: locks, range: 0..<locks.length, value: 0)
          blit.endEncoding()
          guard let encoder = context.commandBuffer.makeComputeCommandEncoder() else {
            throw LiveSplatError.unavailable
          }
          encoder.setComputePipelineState(fuse)
          encoder.setBuffer(cells, offset: 0, index: 0)
          encoder.setBuffer(locks, offset: 0, index: 1)
          encoder.setBuffer(counters, offset: 0, index: 2)
          encoder.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 3)
          for (i, texture) in textures.enumerated() { encoder.setTexture(texture, index: i) }
          encoder.dispatchThreads(
            MTLSize(width: sampleCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          encoder.endEncoding()
          lastFusion = frame.timestamp
          fused = true
        }
        guard let coverageEncoder = context.commandBuffer.makeComputeCommandEncoder() else {
          throw LiveSplatError.unavailable
        }
        coverageEncoder.setComputePipelineState(coverage)
        coverageEncoder.setBuffer(cells, offset: 0, index: 0)
        coverageEncoder.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 1)
        coverageEncoder.setTexture(color, index: 0)
        coverageEncoder.setTexture(textures[0], index: 1)
        coverageEncoder.setTexture(textures[1], index: 2)
        coverageEncoder.dispatchThreads(
          MTLSize(width: width, height: height, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        coverageEncoder.endEncoding()
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        guard let render = context.commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
          throw LiveSplatError.unavailable
        }
        render.setRenderPipelineState(renderPipeline)
        render.setDepthStencilState(depthState)
        render.setVertexBuffer(cells, offset: 0, index: 0)
        render.setVertexBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 1)
        render.setFragmentBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 1)
        render.setFragmentTexture(textures[0], index: 0)
        render.setFragmentTexture(textures[1], index: 1)
        if snapshot.enabled {
          render.drawPrimitives(
            type: .point, vertexStart: 0, vertexCount: SurfelHash.capacity / quality.visibleStride)
        }
        render.endEncoding()
        guard let encoder = context.commandBuffer.makeComputeCommandEncoder() else {
          throw LiveSplatError.unavailable
        }
        encoder.setComputePipelineState(composite)
        encoder.setTexture(context.sourceColorTexture, index: 0)
        encoder.setTexture(target, index: 1)
        encoder.setTexture(color, index: 2)
        encoder.dispatchThreads(
          MTLSize(width: target.width, height: target.height, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
      } catch {
        generation = .max
        copyCamera(context)
        encodingFailed = true
      }
      // Retain CV texture wrappers AND ARFrame until GPU reads finish. Only 8 summary bytes return.
      let lifetime = PreviewLifetime(frame: transfer, textures: retained)
      let didFuse = fused
      let failedToEncode = encodingFailed
      context.commandBuffer.addCompletedHandler { [self, lifetime] command in
        withExtendedLifetime(lifetime) {
          let counts = counters.contents().assumingMemoryBound(to: UInt32.self)
          let constrained =
            ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState == .serious
            || ProcessInfo.processInfo.thermalState == .critical
          lock.withLock {
            if input.generation == snapshot.generation {
              qualityRecord(
                seconds: command.gpuEndTime - command.gpuStartTime, constrained: constrained)
              if didFuse { fraction = counts[0] == 0 ? 0 : Double(counts[1]) / Double(counts[0]) }
              failure = failedToEncode || command.status == .error
              if command.status == .error { input.generation &+= 1 }
            }
            busy = false
          }
        }
      }
    }
    private func qualityRecord(seconds: Double, constrained: Bool) {
      quality.record(seconds: seconds, constrained: constrained)
    }
  }
  struct SurfelGPUCell {
    var key: SIMD4<Int32>
    var positionRadius: SIMD4<Float>
    var color: SIMD4<Float>
    var state: SIMD4<UInt32>
  }
  struct PreviewUniforms {
    var world: simd_float4x4, view: simd_float4x4
    var intrinsics: SIMD4<Float>, displayX: SIMD4<Float>, displayY: SIMD4<Float>
    var inverseX: SIMD4<Float>, inverseY: SIMD4<Float>, dimensions: SIMD4<Float>
    var settings: SIMD4<UInt32>, flags: SIMD4<UInt32>
  }
  private final class PreviewLifetime: @unchecked Sendable {
    let frame: PreviewFrame
    let textures: [CVMetalTexture]
    init(frame: PreviewFrame, textures: [CVMetalTexture]) {
      self.frame = frame
      self.textures = textures
    }
  }
#endif
