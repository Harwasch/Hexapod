#if canImport(RealityKit) && os(iOS)
  @preconcurrency import Metal
  import Foundation
  import Testing
  import simd
  @testable import LiveSplat

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
  func metalFusionMatchesReferenceForExactlyRepresentableFixture() async throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    // In an async context the iOS 26 SDK resolves this to its async overload.
    let library = try await device.makeDefaultLibrary(bundle: .module)
    let pipeline = try await device.makeComputePipelineState(
      function: #require(library.makeFunction(name: "fuseSurfels")))
    #expect(MemoryLayout<SurfelGPUCell>.stride == 64)
    #expect(MemoryLayout<PreviewUniforms>.stride == 256)
    let cells = try #require(
      device.makeBuffer(length: SurfelHash.capacity * 64, options: .storageModeShared))
    let claims = try #require(
      device.makeBuffer(length: SurfelHash.capacity, options: .storageModeShared))
    let counts = try #require(device.makeBuffer(length: 8, options: .storageModeShared))
    cells.contents().initializeMemory(as: UInt8.self, repeating: 0, count: cells.length)
    func texture(_ format: MTLPixelFormat, bytes: [UInt8]) throws -> MTLTexture {
      let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: 1, height: 1, mipmapped: false)
      descriptor.storageMode = .shared
      descriptor.usage = .shaderRead
      let value = try #require(device.makeTexture(descriptor: descriptor))
      bytes.withUnsafeBytes { p in
        value.replace(
          region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: p.baseAddress!,
          bytesPerRow: bytes.count)
      }
      return value
    }
    let textures = try [
      texture(.r32Float, bytes: [0, 0, 0, 64]), texture(.r8Unorm, bytes: [2]),
      texture(.r8Unorm, bytes: [128]), texture(.rg8Unorm, bytes: [128, 128]),
    ]
    var uniforms = PreviewUniforms(
      world: matrix_identity_float4x4, view: matrix_identity_float4x4,
      intrinsics: SIMD4(1, 1, 0, 0), displayX: SIMD4(1, 0, 0, 0), displayY: SIMD4(0, 1, 0, 0),
      inverseX: SIMD4(1, 0, 0, 0), inverseY: SIMD4(0, 1, 0, 0), dimensions: SIMD4(repeating: 1),
      settings: SIMD4(1, 4, 2, 1), flags: SIMD4(1, 0, 0, 0))
    for epoch: UInt32 in [4, 8] {
      claims.contents().initializeMemory(as: UInt8.self, repeating: 0, count: claims.length)
      counts.contents().initializeMemory(as: UInt8.self, repeating: 0, count: counts.length)
      uniforms.settings.y = epoch  // phase zero selects the only depth pixel
      let command = try #require(queue.makeCommandBuffer())
      let encoder = try #require(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      for (i, buffer) in [cells, claims, counts].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: i)
      }
      encoder.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 3)
      for (i, texture) in textures.enumerated() { encoder.setTexture(texture, index: i) }
      encoder.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      encoder.endEncoding()
      await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        command.addCompletedHandler { _ in continuation.resume() }
        command.commit()
      }
      #expect(command.status == .completed)
      let key = SurfelKey(position: SIMD3(0, 0, -2), depth: 2)
      let cell = cells.contents().assumingMemoryBound(to: SurfelGPUCell.self)[
        key.bucket(count: SurfelHash.capacity / 4) * 4]
      #expect(cell.key == SIMD4(key.x, key.y, key.z, key.level))
      #expect(cell.positionRadius == SIMD4(0, 0, -2, SurfelHash.radius(depth: 2, focalPixels: 1)))
      #expect(cell.state.x == epoch / 4)
      #expect(cell.state.y == epoch)
      #expect(counts.contents().assumingMemoryBound(to: UInt32.self)[0] == 1)
    }
  }
#endif
