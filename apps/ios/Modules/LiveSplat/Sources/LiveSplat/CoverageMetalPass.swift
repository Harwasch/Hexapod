#if canImport(RealityKit) && os(iOS)
  @preconcurrency import Metal
  import Foundation

  /// Fixed storage; high-confidence depth updates direction masks before surfel fusion.
  final class CoverageMetalPass {
    let cells: MTLBuffer, development: MTLBuffer
    private let claims: MTLBuffer, summary: MTLBuffer
    private let observe: MTLComputePipelineState, reduce: MTLComputePipelineState,
      finish: MTLComputePipelineState
    init(device: MTLDevice, library: MTLLibrary) throws {
      func buffer(_ length: Int, shared: Bool = false) throws -> MTLBuffer {
        guard
          let b = device.makeBuffer(
            length: length, options: shared ? .storageModeShared : .storageModePrivate)
        else { throw LiveSplatError.unavailable }
        return b
      }
      cells = try buffer(CoverageHash.capacity * MemoryLayout<CoverageGPUCell>.stride)
      development = try buffer(SurfelHash.capacity * MemoryLayout<CoverageDevelopment>.stride)
      claims = try buffer(CoverageHash.capacity)
      summary = try buffer(129 * MemoryLayout<CoverageSummary>.stride, shared: true)
      func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: name) else { throw LiveSplatError.unavailable }
        return try device.makeComputePipelineState(function: f)
      }
      observe = try pipeline("observeCoverageCells")
      reduce = try pipeline("reduceCoverageSummary")
      finish = try pipeline("finishCoverageSummary")
    }
    func reset(_ command: MTLCommandBuffer) throws {
      guard let b = command.makeBlitCommandEncoder() else { throw LiveSplatError.unavailable }
      for buffer in [cells, claims, summary] {
        b.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
      }
      b.endEncoding()
    }
    func fuse(
      _ command: MTLCommandBuffer, uniforms: inout PreviewUniforms,
      style: inout CoverageStyleUniforms, textures: [MTLTexture]
    ) throws {
      guard let b = command.makeBlitCommandEncoder() else { throw LiveSplatError.unavailable }
      b.fill(buffer: claims, range: 0..<claims.length, value: 0)
      b.endEncoding()
      guard let e = command.makeComputeCommandEncoder() else { throw LiveSplatError.unavailable }
      e.setComputePipelineState(observe)
      e.setBuffer(cells, offset: 0, index: 0)
      e.setBuffer(claims, offset: 0, index: 1)
      e.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 3)
      e.setTexture(textures[0], index: 0)
      e.setTexture(textures[1], index: 1)
      e.dispatchThreads(
        MTLSize(width: Int(uniforms.settings.x), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      e.endEncoding()
      guard let r = command.makeComputeCommandEncoder() else { throw LiveSplatError.unavailable }
      r.setComputePipelineState(reduce)
      r.setBuffer(cells, offset: 0, index: 0)
      r.setBuffer(summary, offset: 0, index: 1)
      r.setBytes(&style, length: MemoryLayout<CoverageStyleUniforms>.stride, index: 2)
      r.dispatchThreads(
        MTLSize(width: 128, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      r.memoryBarrier(scope: .buffers)
      r.setComputePipelineState(finish)
      r.setBuffer(summary, offset: 0, index: 0)
      r.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      r.endEncoding()
    }
    func sample(camera: SIMD3<Float>, timestamp: Double) -> ScanGuidanceSample? {
      let s = summary.contents().assumingMemoryBound(to: CoverageSummary.self)[128]
      guard s.centerCount.w > 0 else { return nil }
      return ScanGuidanceSample(
        center: SIMD3(s.centerCount.x, s.centerCount.y, s.centerCount.z),
        camera: camera, observedCells: Int(s.centerCount.w),
        shortSpreadFraction: Float(s.counts.x) / s.centerCount.w, timestamp: timestamp)
    }
  }
#endif
