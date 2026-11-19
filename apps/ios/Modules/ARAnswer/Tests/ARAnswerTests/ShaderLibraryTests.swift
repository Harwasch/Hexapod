#if canImport(Metal)
  import Foundation
  import Metal
  import Testing

  @testable import ARAnswer

  @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
  func packageContainsSubjectHighlightComputeFunction() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try device.makeDefaultLibrary(bundle: .module)
    let function = try #require(library.makeFunction(name: "subjectHighlight"))
    _ = try device.makeComputePipelineState(function: function)
  }
#endif
