import Foundation
import XCTest

@testable import PhoneTrainPrep

final class PrepTests: XCTestCase {
  private let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

  private func frame(
    _ index: Int, x: Double = 0, angle: Double = 0,
    tracking: String = "normal"
  ) throws -> CaptureFrame {
    let a = angle * .pi / 180
    var pose = identity
    pose[0] = cos(a)
    pose[1] = -sin(a)
    pose[4] = sin(a)
    pose[5] = cos(a)
    pose[3] = x
    let object: [String: Any] = [
      "index": index, "timestamp": Double(index), "transform": pose,
      "intrinsics": [100, 0, 50, 0, 100, 40, 0, 0, 1],
      "imageWidth": 100, "imageHeight": 80, "trackingState": tracking,
    ]
    return try JSONDecoder().decode(
      CaptureFrame.self, from: JSONSerialization.data(withJSONObject: object))
  }

  func testLumaIsRec709AndIgnoresTrailingBytes() {
    let luma = Keyframes.luma(rgb: [255, 255, 255, 255, 0, 0, 0, 0, 0, 7])
    XCTAssertEqual(luma.count, 3)
    XCTAssertEqual(luma[0], 1, accuracy: 1e-12)
    XCTAssertEqual(luma[1], 0.2126, accuracy: 1e-12)
    XCTAssertEqual(luma[2], 0)
    XCTAssertTrue(Keyframes.luma(rgb: []).isEmpty)
  }

  func testSavedBundleLogValidation() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = Data(
      #"{"version":1,"format":"hex-capture","frameCount":2,"video":"rgb.mov","frameLog":"frames.jsonl"}"#
        .utf8)
    try manifest.write(to: root.appendingPathComponent("manifest.json"))
    let encoder = JSONEncoder()
    let log = try [frame(0), frame(1, x: 0.1)].map {
      String(decoding: try encoder.encode($0), as: UTF8.self)
    }.joined(separator: "\n")
    try log.write(
      to: root.appendingPathComponent("frames.jsonl"), atomically: true, encoding: .utf8)
    XCTAssertEqual(try CaptureFrame.read(from: root).count, 2)
    try "".write(to: root.appendingPathComponent("frames.jsonl"), atomically: true, encoding: .utf8)
    XCTAssertThrowsError(try CaptureFrame.read(from: root))
  }

  func testTranslationRotationTrackingAndRelativeBlur() throws {
    var config = PrepConfig()
    config.minimumKeyframes = 2
    let frames = try [
      frame(0), frame(1, x: 0.02), frame(2, x: 0.041),
      frame(3, x: 0.05, angle: 5), frame(4, x: 1, tracking: "limited:motion"),
      frame(5, x: 2), frame(6, x: 3),
    ]
    let selected = try Keyframes.select(
      frames: frames,
      scores: [0: 10, 1: 10, 2: 10, 3: 10, 5: 0.1, 6: 10], config: config)
    XCTAssertEqual(selected.map { $0.source.index }, [0, 2, 3, 6])
    XCTAssertEqual(
      selected[0].source.scaledIntrinsics(width: 50, height: 40),
      [50, 0, 25, 0, 50, 20, 0, 0, 1])
  }

  func testRelaxationThinningAndHoldoutMatchNativeSplit() throws {
    var config = PrepConfig()
    config.minimumKeyframes = 8
    config.maximumKeyframes = 10
    let frames = try (0..<30).map { try frame($0, x: Double($0) * 0.031) }
    let scores = Dictionary(uniqueKeysWithValues: frames.map { ($0.index, 1.0) })
    let selected = try Keyframes.select(frames: frames, scores: scores, config: config)
    XCTAssertEqual(selected.count, 10)
    XCTAssertEqual(selected.filter(\.isHoldout).map { $0.source.index }, [0, 24])
    XCTAssertEqual((0..<25).filter { Keyframes.isHoldout($0) }, [0, 8, 16, 24])
    config.minimumKeyframes = 20
    config.maximumKeyframes = 40
    XCTAssertEqual(try Keyframes.select(frames: frames, scores: scores, config: config).count, 30)
  }

  func testLaplacianSeparatesFlatAndSharp() throws {
    let flat = [Double](repeating: 0.5, count: 64)
    let sharp = (0..<64).map { Double(($0 / 8 + $0 % 8) % 2) }
    XCTAssertEqual(try Keyframes.blurScore(luma: flat, width: 8, height: 8), 0)
    XCTAssertGreaterThan(try Keyframes.blurScore(luma: sharp, width: 8, height: 8), 10)
    XCTAssertThrowsError(try Keyframes.blurScore(luma: [], width: 8, height: 8))
  }

  func testSVDParityWithServerFixture() throws {
    let input = [
      0.866112, -0.49994, 0.00002, 1,
      0.50003, 0.86600, -0.00001, 2,
      0.00001, 0.00002, 1.00003, 3,
      0.0000002, 0, 0, 1,
    ]
    // Computed with capture_bundle.world_to_camera(np.array(input).reshape(4,4));
    // corrected ARKit R = returned_rotation.T @ diag([1,-1,-1]). numpy SVD,
    // float64. 1e-12 compares algorithms, not an expanded input tolerance.
    let expected = [
      0.8660395473533415, -0.49997550170601096, 1.0669698318337586e-05,
      0.49997550181851247, 0.86603954720334, -1.6160444426194594e-05,
      -1.1605543905796453e-06, 1.9330171746481572e-05, 0.9999999998124991,
    ]
    let corrected = try Pose.corrected(input)
    for r in 0..<3 {
      for c in 0..<3 {
        XCTAssertEqual(corrected[r * 4 + c], expected[r * 3 + c], accuracy: 1e-12)
      }
    }
    XCTAssertEqual([corrected[3], corrected[7], corrected[11]], [1, 2, 3])
    XCTAssertEqual(Array(corrected[12..<16]), [0, 0, 0, 1])
    for (index, value) in [(0, 1.1), (1, 0.02), (0, -1.0), (12, 2e-6), (0, Double.nan)] {
      var bad = identity
      bad[index] = value
      XCTAssertThrowsError(try Pose.corrected(bad))
    }
  }

  func testPlaneBackProjectionConfidenceEdgesAndHalfDepth() throws {
    var pose = identity
    pose[3] = 1
    let depth = [Double](repeating: 2, count: 25)
    var confidence = [UInt8](repeating: 2, count: 25)
    confidence[6] = 1
    let rgb: [UInt8] = Array(repeating: [20, 40, 60], count: 25).flatMap { $0 }
    func project(_ d: [Double]) throws -> [SeedPoint] {
      try SeedPoints.backProject(
        depth: d, confidence: confidence, width: 5, height: 5,
        intrinsics: [2, 0, 2, 0, 2, 2, 0, 0, 1], pose: pose, rgb: rgb, rgbWidth: 5, rgbHeight: 5)
    }
    let points = try project(depth)
    XCTAssertEqual(points.count, 8)
    XCTAssertTrue(points.contains(SeedPoint(x: 1, y: 0, z: -2, rgb: [20, 40, 60])))
    XCTAssertTrue(points.allSatisfy { $0.z == -2 })
    var edge = depth
    edge[12] = 5
    XCTAssertEqual(try project(edge).count, 3)
    XCTAssertEqual(try SeedPoints.decodeDepth(Data([0, 64, 0, 60]), count: 2), [2, 1])
  }

  func testVoxelCountDeterminismAndBound() {
    let points = (0..<100).map { SeedPoint(x: Double($0) * 0.001, y: 0, z: -1, rgb: [1, 2, 3]) }
    XCTAssertEqual(SeedPoints.voxelDownsample(points, size: 0.015).count, 7)
    XCTAssertEqual(
      SeedPoints.voxelDownsample(points, size: 0.015),
      SeedPoints.voxelDownsample(points, size: 0.015))
    var bounded = VoxelSeeds(size: 0.001, maximum: 5)
    bounded.add(points)
    XCTAssertLessThanOrEqual(bounded.points.count, 5)
  }
}
