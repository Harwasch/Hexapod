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
    let library = try await device.makeDefaultLibrary(bundle: .module)
    let coarsePipeline = try await device.makeComputePipelineState(
      function: #require(library.makeFunction(name: "observeCoverageCells")))
    let pipeline = try await device.makeComputePipelineState(
      function: #require(library.makeFunction(name: "fuseSurfels")))
    #expect(MemoryLayout<SurfelGPUCell>.stride == 64)
    #expect(MemoryLayout<PreviewUniforms>.stride == 256)
    let cells = try #require(
      device.makeBuffer(length: SurfelHash.capacity * 64, options: .storageModeShared))
    let attributes = try #require(
      device.makeBuffer(length: GaussianSeed.attributeBytes, options: .storageModeShared))
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
    #expect(MemoryLayout<SurfelGPUCell>.offset(of: \.key) == 0)
    #expect(MemoryLayout<SurfelGPUCell>.offset(of: \.positionRadius) == 16)
    #expect(MemoryLayout<SurfelGPUCell>.offset(of: \.color) == 32)
    #expect(MemoryLayout<SurfelGPUCell>.offset(of: \.state) == 48)
    let coarse = try #require(
      device.makeBuffer(length: CoverageHash.capacity * 32, options: .storageModeShared))
    let development = try #require(
      device.makeBuffer(length: SurfelHash.capacity * 16, options: .storageModeShared))
    coarse.contents().initializeMemory(as: UInt8.self, repeating: 0, count: coarse.length)
    development.contents().initializeMemory(as: UInt8.self, repeating: 0, count: development.length)
    var styleUniforms = CoverageStyleUniforms()
    for style: UInt32 in [0, 2] {
      cells.contents().initializeMemory(as: UInt8.self, repeating: 0, count: cells.length)
      uniforms.flags.w = style
      coarse.contents().initializeMemory(as: UInt8.self, repeating: 0, count: coarse.length)
      for epoch: UInt32 in [4, 8] {
        claims.contents().initializeMemory(as: UInt8.self, repeating: 0, count: claims.length)
        counts.contents().initializeMemory(as: UInt8.self, repeating: 0, count: counts.length)
        uniforms.settings.y = epoch  // phase zero selects the only depth pixel
        let command = try #require(queue.makeCommandBuffer())
        styleUniforms.timing.x = Float(epoch) / 4
        let coarseEncoder = try #require(command.makeComputeCommandEncoder())
        coarseEncoder.setComputePipelineState(coarsePipeline)
        coarseEncoder.setBuffer(coarse, offset: 0, index: 0)
        coarseEncoder.setBuffer(claims, offset: 0, index: 1)
        coarseEncoder.setBytes(&uniforms, length: MemoryLayout<PreviewUniforms>.stride, index: 3)
        coarseEncoder.setTexture(textures[0], index: 0)
        coarseEncoder.setTexture(textures[1], index: 1)
        coarseEncoder.dispatchThreads(
          MTLSize(width: 1, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        coarseEncoder.endEncoding()
        // Coarse and fine hashes use independent claim buffers in production.
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.fill(buffer: claims, range: 0..<claims.length, value: 0)
        blit.endEncoding()
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (i, buffer) in [cells, claims, counts].enumerated() {
          encoder.setBuffer(buffer, offset: 0, index: i)
        }
        encoder.setBuffer(attributes, offset: 0, index: 4)
        encoder.setBuffer(coarse, offset: 0, index: 5)
        encoder.setBuffer(development, offset: 0, index: 6)
        encoder.setBytes(
          &styleUniforms, length: MemoryLayout<CoverageStyleUniforms>.stride, index: 7)
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
        let coarseKey = SIMD3<Int32>(0, 0, -20)
        let coarseSlot = CoverageHash.bucket(coarseKey, count: CoverageHash.capacity / 4) * 4
        let coarseCell = coarse.contents().assumingMemoryBound(to: CoverageGPUCell.self)[coarseSlot]
        #expect(coarseCell.key == SIMD4(0, 0, -20, 0))
        #expect(coarseCell.views.z == epoch / 4)
        let animation = development.contents().assumingMemoryBound(to: SIMD4<Float>.self)[
          key.bucket(count: SurfelHash.capacity / 4) * 4]
        #expect(
          animation.y
            == CoverageAppearance.score(
              observations: Float(epoch / 4),
              confidentRun: Float(epoch / 4), azimuth: coarseCell.views.x.nonzeroBitCount,
              elevation: coarseCell.views.y.nonzeroBitCount))
        #expect(animation.z == styleUniforms.timing.x && animation.w == -1)
        #expect(cell.state.x == epoch / 4)
        #expect(cell.state.y == epoch)
        #expect(counts.contents().assumingMemoryBound(to: UInt32.self)[0] == 1)
        if style == 2 {
          let index = key.bucket(count: SurfelHash.capacity / 4) * 4
          #expect(
            attributes.contents().assumingMemoryBound(to: SIMD4<Float>.self)[index]
              == SIMD4(0, 0, 1, 0.85))
        }
      }
    }
  }
#endif
