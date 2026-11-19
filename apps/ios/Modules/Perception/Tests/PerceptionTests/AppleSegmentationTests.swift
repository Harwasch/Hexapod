#if canImport(Vision) && canImport(CoreML) && canImport(CoreImage)
  import Testing

  @testable import Perception

  @Test func visionRejectsInvalidCameraBuffersBeforeInference() async {
    let segmenter = VisionSegmenter()
    for image in [
      CameraImage(width: 0, height: 1, rgba: []),
      CameraImage(width: 2, height: 2, rgba: [0, 0, 0, 255]),
    ] {
      await #expect(throws: PerceptionError.invalidGeometry) {
        try await segmenter.instances(image)
      }
      await #expect(throws: PerceptionError.invalidGeometry) {
        try await segmenter.segment(image, prompt: .point(.init(x: 0.5, y: 0.5)))
      }
    }
  }
#endif
