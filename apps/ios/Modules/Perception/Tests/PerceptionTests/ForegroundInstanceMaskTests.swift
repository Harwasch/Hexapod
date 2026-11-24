import Testing

@testable import Perception

@Test func singlePassInstanceDecodingPreservesPixelsAspectAndNonconsecutiveIDs() throws {
  let masks = try ForegroundInstanceMasks.decode(
    width: 3, height: 2,
    labels: [3, 0, 7, 7, 3, 0], instances: [7, 3])
  #expect(masks.count == 2)
  #expect(masks[0].width == 3 && masks[0].height == 2)
  #expect(masks[0].pixels == [0, 0, 255, 255, 0, 0])
  #expect(masks[1].pixels == [255, 0, 0, 0, 255, 0])
}
@Test func instanceDecodingRejectsMalformedMapsAndNeverLabelsBackground() throws {
  #expect(
    try ForegroundInstanceMasks.decode(width: 1, height: 1, labels: [0], instances: []).isEmpty)
  #expect(throws: PerceptionError.self) {
    try ForegroundInstanceMasks.decode(width: 1, height: 1, labels: [], instances: [1])
  }
  #expect(throws: PerceptionError.self) {
    try ForegroundInstanceMasks.decode(width: 1, height: 1, labels: [0], instances: [0])
  }
}
