import Testing

@testable import ARAnswer

@Test func portraitTallViewportRoundTripsCameraPixelsWithoutAnOffset() throws {
  // Landscape 1920x1440 camera, rotated clockwise into a 390x844 aspect-fill view.
  let crop = (1 - (390.0 / 844) / (1440.0 / 1920)) / 2
  let span = 1 - 2 * crop
  let mapping = ViewportMapping(viewportToImage: [0, -span, 1, 0, 0, 1 - crop])
  let image = try #require(mapping.imagePoint(x: 0.25, y: 0.75))
  #expect(abs(image.x - 0.75) < 1e-12)
  #expect(abs(image.y - (1 - crop - 0.25 * span)) < 1e-12)
  let view = try #require(mapping.viewPoint(x: image.x, y: image.y))
  #expect(abs(view.x * 390 - 97.5) < 1e-9)
  #expect(abs(view.y * 844 - 633) < 1e-9)
}

@Test func maskUsesTallViewportAndBilinearEdges() throws {
  let mask = try #require(
    HighlightMask.project(
      width: 2, height: 1, pixels: [0, 255], viewportToImage: [1, 0, 0, 1, 0, 0],
      outputWidth: 4, outputHeight: 8))
  #expect(mask.width == 4 && mask.height == 8)
  #expect(Array(mask.pixels.prefix(4)) == [0, 64, 191, 255])
}
