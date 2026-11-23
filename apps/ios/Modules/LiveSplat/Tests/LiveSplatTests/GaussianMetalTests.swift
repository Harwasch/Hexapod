#if canImport(Metal)
  @preconcurrency import Metal
  import Foundation
  import Testing
  @testable import LiveSplat

  /// Apple-only execution test. Linux verifies the same network in GaussianTests;
  /// it cannot establish Metal compilation, cross-dispatch barriers or draw ordering.
  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
  func gpuGaussianSortMatchesPortableOrder() async throws {
    struct Entry {
      var depth: Float
      var index: UInt32
    }
    #expect(MemoryLayout<Entry>.stride == 8)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let library = try await device.makeDefaultLibrary(bundle: .module)
    let function = try #require(library.makeFunction(name: "bitonicGaussianSort"))
    let pipeline = try await device.makeComputePipelineState(function: function)
    let buffer = try #require(device.makeBuffer(length: 256 * 8, options: .storageModeShared))
    let pointer = buffer.contents().assumingMemoryBound(to: Entry.self)
    var expected: [GaussianSort.Entry] = []
    for i in 0..<256 {
      let d: Float = i % 13 == 0 ? -.infinity : Float((i * 37) % 19 + 1)
      pointer[i] = Entry(depth: d, index: UInt32(i))
      expected.append(.init(depth: d, index: UInt32(i)))
    }
    GaussianSort.sort(&expected)
    let command = try #require(queue.makeCommandBuffer())
    let encoder = try #require(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    var width = 2
    while width <= 256 {
      var distance = width / 2
      while distance > 0 {
        var stage = SIMD4<UInt32>(UInt32(width), UInt32(distance), 256, 0)
        encoder.setBytes(&stage, length: 16, index: 1)
        encoder.dispatchThreads(
          MTLSize(width: 256, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
        distance /= 2
      }
      width *= 2
    }
    encoder.endEncoding()
    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
      command.addCompletedHandler { _ in continuation.resume() }
      command.commit()
    }
    #expect(command.status == .completed)
    for i in 0..<256 {
      #expect(pointer[i].depth == expected[i].depth && pointer[i].index == expected[i].index)
    }
  }
#endif
