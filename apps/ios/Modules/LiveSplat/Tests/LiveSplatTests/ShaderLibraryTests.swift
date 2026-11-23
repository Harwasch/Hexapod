#if canImport(Metal)
  import Foundation
  import Metal
  import Testing

  @testable import LiveSplat

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
  func packageContainsAllLiveSplatComputeFunctions() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try device.makeDefaultLibrary(bundle: .module)
    for name in [
      "fuseSurfels", "coverageScanPreview", "compositeScanPreview", "prepareGaussianSort",
      "bitonicGaussianSort", "reduceScanScene", "finishScanScene", "compositeScanInset",
    ] {
      let function = try #require(library.makeFunction(name: name))
      _ = try device.makeComputePipelineState(function: function)
    }
    let render = MTLRenderPipelineDescriptor()
    render.vertexFunction = try #require(library.makeFunction(name: "surfelVertex"))
    render.fragmentFunction = try #require(library.makeFunction(name: "surfelFragment"))
    render.colorAttachments[0].pixelFormat = .rgba16Float
    render.depthAttachmentPixelFormat = .depth32Float
    _ = try device.makeRenderPipelineState(descriptor: render)
    for pair in [
      ("gaussianVertex", "gaussianFragment"), ("phoneFrustumVertex", "phoneFrustumFragment"),
    ] {
      render.vertexFunction = try #require(library.makeFunction(name: pair.0))
      render.fragmentFunction = try #require(library.makeFunction(name: pair.1))
      _ = try device.makeRenderPipelineState(descriptor: render)
    }
    #expect(library.makeFunction(name: "optimizeGaussians") == nil)
  }
#endif
