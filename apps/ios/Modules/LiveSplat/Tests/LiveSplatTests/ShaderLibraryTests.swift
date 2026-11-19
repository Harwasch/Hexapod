#if canImport(Metal)
  import Foundation
  import Metal
  import Testing

  @testable import LiveSplat

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
  func packageContainsAllLiveSplatComputeFunctions() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try device.makeDefaultLibrary(bundle: .module)
    for name in ["projectGaussians", "binAndSort", "compositeGaussians", "optimizeGaussians"] {
      let function = try #require(library.makeFunction(name: name))
      _ = try device.makeComputePipelineState(function: function)
    }
  }
#endif
