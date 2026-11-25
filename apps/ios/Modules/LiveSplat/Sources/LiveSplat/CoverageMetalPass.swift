#if canImport(RealityKit) && os(iOS)
  @preconcurrency import Metal
  import Foundation

  struct CoverageMetalParameters {
    var shape: SIMD4<Float>
    var detail: SIMD4<Float>
    var counts: SIMD4<UInt32>
    init(_ p: CoverageCreditParameters) {
      shape = SIMD4(cos(p.coneDeg * .pi / 180), p.sharpness, p.viewsToFill, p.targetMmPerPx ?? 0)
      detail = SIMD4(
        p.parallaxMinDeg.map { cos($0 * .pi / 180) } ?? -2, p.normalEMA, p.baseLength,
        p.parallaxStubFraction)
      counts = SIMD4(
        UInt32(p.hairs), UInt32(CoverageHash.capacity), UInt32(CoverageHash.maxDrawnHairs), 1)
    }
  }

  /// Owned by the preview's one-in-flight guard; no CPU geometry readback or growing buffers.
  final class CoverageMetalPass {
    private let cells: MTLBuffer, geometry: MTLBuffer, claims: MTLBuffer, history: MTLBuffer
    private let hairs: MTLBuffer, arguments: MTLBuffer
    private let observe: MTLComputePipelineState, credit: MTLComputePipelineState
    private let prepare: MTLComputePipelineState, finish: MTLComputePipelineState
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private var parameters: CoverageMetalParameters
    private var epoch: UInt32 = 0
    init(device: MTLDevice, library: MTLLibrary) throws {
      parameters = CoverageMetalParameters(try .bundled())
      func buffer(_ bytes: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
          throw LiveSplatError.unavailable
        }
        return b
      }
      cells = try buffer(CoverageHash.capacity * MemoryLayout<CoverageGPUCell>.stride)
      geometry = try buffer(CoverageHash.capacity * 32)
      claims = try buffer(CoverageHash.capacity / 4 * 4)
      history = try buffer(CoverageHash.capacity * 60)
      hairs = try buffer(CoverageHash.maxDrawnHairs * 8)
      arguments = try buffer(16)
      func function(_ name: String) throws -> MTLFunction {
        guard let f = library.makeFunction(name: name) else { throw LiveSplatError.unavailable }
        return f
      }
      observe = try device.makeComputePipelineState(function: function("observeCoverageCells"))
      credit = try device.makeComputePipelineState(function: function("creditCoverageCells"))
      prepare = try device.makeComputePipelineState(function: function("prepareCoverageHairs"))
      finish = try device.makeComputePipelineState(function: function("finishCoverageHairs"))
      let d = MTLRenderPipelineDescriptor()
      d.vertexFunction = try function("coverageHairVertex")
      d.fragmentFunction = try function("coverageHairFragment")
      d.colorAttachments[0].pixelFormat = .rgba16Float
      d.depthAttachmentPixelFormat = .depth32Float
      pipeline = try device.makeRenderPipelineState(descriptor: d)
      let depth = MTLDepthStencilDescriptor()
      depth.depthCompareFunction = .lessEqual
      depth.isDepthWriteEnabled = false
      guard let state = device.makeDepthStencilState(descriptor: depth) else {
        throw LiveSplatError.unavailable
      }
      depthState = state
    }
    func reset(_ command: MTLCommandBuffer) throws {
      guard let e = command.makeBlitCommandEncoder() else { throw LiveSplatError.unavailable }
      for b in [cells, geometry, claims, history, arguments] {
        e.fill(buffer: b, range: 0..<b.length, value: 0)
      }
      e.endEncoding()
    }
    private func bindings(_ e: MTLComputeCommandEncoder, _ u: inout PreviewUniforms) {
      e.setBuffer(cells, offset: 0, index: 0)
      e.setBuffer(geometry, offset: 0, index: 1)
      e.setBuffer(claims, offset: 0, index: 2)
      e.setBuffer(history, offset: 0, index: 3)
      e.setBytes(&u, length: MemoryLayout<PreviewUniforms>.stride, index: 4)
      e.setBytes(&parameters, length: MemoryLayout<CoverageMetalParameters>.stride, index: 5)
    }
    func fuse(
      _ command: MTLCommandBuffer, uniforms u: inout PreviewUniforms, textures: [MTLTexture]
    ) throws {
      epoch &+= 1
      parameters.counts.y = epoch
      guard let b = command.makeBlitCommandEncoder() else { throw LiveSplatError.unavailable }
      b.fill(buffer: claims, range: 0..<claims.length, value: 0)
      b.endEncoding()
      guard let e = command.makeComputeCommandEncoder() else { throw LiveSplatError.unavailable }
      e.setComputePipelineState(observe)
      bindings(e, &u)
      e.setTexture(textures[0], index: 0)
      e.setTexture(textures[1], index: 1)
      e.dispatchThreads(
        MTLSize(width: Int(u.settings.x), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      e.endEncoding()
      guard let c = command.makeComputeCommandEncoder() else { throw LiveSplatError.unavailable }
      c.setComputePipelineState(credit)
      bindings(c, &u)
      c.dispatchThreads(
        MTLSize(width: CoverageHash.capacity, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      c.endEncoding()
    }
    func prepareDraw(
      _ command: MTLCommandBuffer, uniforms u: inout PreviewUniforms, quality: PreviewQuality
    ) throws {
      // The existing 8 ms feedback decimates coverage with the preview and entirely
      // skips its drawing at the lowest quality. Credit still follows admitted keyframes.
      guard quality.level < 2 else { return }
      parameters.counts.w = UInt32(quality.visibleStride)
      parameters.counts.z = UInt32(CoverageHash.maxDrawnHairs / quality.visibleStride)
      guard let b = command.makeBlitCommandEncoder() else { throw LiveSplatError.unavailable }
      b.fill(buffer: arguments, range: 0..<16, value: 0)
      b.endEncoding()
      guard let e = command.makeComputeCommandEncoder() else { throw LiveSplatError.unavailable }
      e.setComputePipelineState(prepare)
      bindings(e, &u)
      e.setBuffer(hairs, offset: 0, index: 2)
      e.setBuffer(arguments, offset: 0, index: 3)
      e.dispatchThreads(
        MTLSize(width: CoverageHash.capacity / quality.visibleStride, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      e.memoryBarrier(scope: .buffers)
      e.setComputePipelineState(finish)
      e.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      e.endEncoding()
    }
    func draw(_ e: MTLRenderCommandEncoder, uniforms u: inout PreviewUniforms) {
      e.setRenderPipelineState(pipeline)
      e.setDepthStencilState(depthState)
      e.setVertexBuffer(cells, offset: 0, index: 0)
      e.setVertexBuffer(geometry, offset: 0, index: 1)
      e.setVertexBuffer(hairs, offset: 0, index: 2)
      e.setVertexBytes(&u, length: MemoryLayout<PreviewUniforms>.stride, index: 4)
      e.setVertexBytes(&parameters, length: MemoryLayout<CoverageMetalParameters>.stride, index: 5)
      e.setFragmentBytes(&u, length: MemoryLayout<PreviewUniforms>.stride, index: 4)
      e.drawPrimitives(type: .line, indirectBuffer: arguments, indirectBufferOffset: 0)
    }
  }
#endif
