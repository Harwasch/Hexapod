#if canImport(RealityKit) && os(iOS)
  @preconcurrency import Metal
  import Foundation
  import Testing
  import simd
  @testable import LiveSplat

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
  func coverageMetalCreditFillsOrbitAndRejectsExcludedViews() async throws {
    struct Geometry {
      var key: SIMD4<Int32>
      var position: SIMD4<Float>
    }
    #expect(MemoryLayout<CoverageGPUCell>.stride == 32)
    #expect(MemoryLayout<CoverageMetalParameters>.stride == 48)
    #expect(MemoryLayout<Geometry>.stride == 32)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let library = try await device.makeDefaultLibrary(bundle: .module)
    let function = try #require(library.makeFunction(name: "creditCoverageCells"))
    let pipeline = try await device.makeComputePipelineState(function: function)
    func buffer(_ length: Int) throws -> MTLBuffer {
      let b = try #require(device.makeBuffer(length: length, options: .storageModeShared))
      b.contents().initializeMemory(as: UInt8.self, repeating: 0, count: b.length)
      return b
    }
    let cells = try buffer(CoverageHash.capacity * 32)
    let geometry = try buffer(CoverageHash.capacity * 32)
    let history = try buffer(CoverageHash.capacity * 60)
    let pointer = cells.contents().assumingMemoryBound(to: CoverageGPUCell.self)
    let g = geometry.contents().assumingMemoryBound(to: Geometry.self)
    var parameters = CoverageMetalParameters(.init())
    parameters.counts.y = 123
    var uniforms = PreviewUniforms(
      world: matrix_identity_float4x4, view: matrix_identity_float4x4,
      intrinsics: SIMD4(1, 1, 0, 0), displayX: .zero, displayY: .zero,
      inverseX: .zero, inverseY: .zero, dimensions: SIMD4(1, 1, 1000, 1000),
      settings: SIMD4(1, 1, 2, 1), flags: SIMD4(1, 0, 0, 0))
    let normal = SIMD3<Float>(0, 0, 1)
    var orbit: [SIMD3<Float>] = []
    for _ in 0..<8 {
      for h in 0..<14 {
        orbit.append(CoverageCreditReference.hair(h, count: 14, normal: normal) * 2)
      }
    }
    let hair0 = CoverageCreditReference.hair(0, count: 14, normal: normal)
    let excluded = Array(repeating: SIMD3(-hair0.x, -hair0.y, hair0.z) * 2, count: 100)
    for cameras in [orbit, excluded] {
      pointer[0] = CoverageGPUCell()
      pointer[0].viewCount = 1
      g[0] = Geometry(key: SIMD4(0, 0, 0, 123), position: SIMD4(0, 0, 0, 1))
      // Same epoch but not observed by the current pass: must earn no credit.
      pointer[1] = pointer[0]
      g[1] = Geometry(key: .zero, position: SIMD4(0, 0, 0, 1))
      var reference = CoverageHash(capacity: 4)
      var epoch: UInt16 = 0
      let command = try #require(queue.makeCommandBuffer())
      for camera in cameras {
        epoch += 1
        reference.observe(
          position: .zero, normal: normal, camera: camera, focalPixels: 1000, epoch: epoch)
        uniforms.world.columns.3 = SIMD4(camera, 1)
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(cells, offset: 0, index: 0)
        encoder.setBuffer(geometry, offset: 0, index: 1)
        encoder.setBuffer(history, offset: 0, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 4)
        encoder.setBytes(
          &parameters, length: MemoryLayout<CoverageMetalParameters>.stride, index: 5)
        encoder.dispatchThreads(
          MTLSize(width: 2, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
      }
      await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        command.addCompletedHandler { _ in continuation.resume() }
        command.commit()
      }
      #expect(command.status == .completed)
      let slot = try #require(reference.keys.firstIndex { $0 != nil })
      if cameras.count == orbit.count {
        for h in 0..<14 {
          #expect(pointer[0][h] == 255 && pointer[0][h] == reference.cells[slot][h])
        }
      } else {
        #expect(pointer[0][0] == 0 && pointer[0][0] == reference.cells[slot][0])
      }
      for h in 0..<14 { #expect(pointer[1][h] == 0) }
    }
  }
#endif
