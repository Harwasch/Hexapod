import Foundation
import Testing

@testable import Perception

private actor ForegroundMasks: InstanceSegmenting {
  var calls = 0
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    throw PerceptionError.unsupportedModel
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] {
    calls += 1
    return [true, false].map { left in
      SubjectMask(width: 4, height: 4, pixels: (0..<16).map { (($0 % 4 < 2) == left) ? 255 : 0 })
    }
  }
}
private struct ForegroundLabel: ImageIdentifying {
  func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel? { nil }
}
@Test func oneForegroundRequestRefreshesAllSelectedWorldSurfaces() async throws {
  let masks = ForegroundMasks()
  let scene = PerceptionScene(segmenter: masks, instances: masks, identifier: ForegroundLabel())
  let frame = FrameSnapshot(
    image: CameraImage(width: 4, height: 4, rgba: Array(repeating: 255, count: 64)),
    depth: DepthImage(width: 4, height: 4, metres: Array(repeating: 2, count: 16)),
    camera: CameraGeometry(
      fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4,
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
  await scene.update(frame, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  let first = try await scene.foregroundSelection(tap: ImagePoint(x: 0.2, y: 0.5))
  #expect(first.surfaces.count == 1 && first.surfaces[0].mesh.points.count == 8)
  let second = try await scene.foregroundSelection(tap: ImagePoint(x: 0.8, y: 0.5))
  #expect(second.entities.count == 2 && second.surfaces.count == 2)
  await scene.update(frame, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 2)
  let refreshed = try await scene.foregroundSelection(tap: nil)
  #expect(refreshed.surfaces.count == 2 && refreshed.surfaces.allSatisfy { $0.capturedAt == 2 })
  #expect(await masks.calls == 3)
  let removed = try await scene.foregroundSelection(tap: ImagePoint(x: 0.2, y: 0.5))
  #expect(removed.entities.count == 1 && removed.surfaces.count == 1)
  let empty = try await scene.foregroundSelection(tap: ImagePoint(x: -1, y: 0))
  #expect(empty.entities.isEmpty && empty.surfaces.isEmpty)
}

private struct RingMask: InstanceSegmenting {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    throw PerceptionError.unsupportedModel
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] {
    [
      SubjectMask(
        width: 8, height: 8,
        pixels: (0..<64).map {
          let x = $0 % 8
          let y = $0 / 8
          return x < 2 || x >= 6 || y < 2 || y >= 6 ? 255 : 0
        })
    ]
  }
}
@Test func worldSurfaceKeepsAHollowSubjectSelectedWhenItsAnchorFallsInAHole() async throws {
  let segmenter = RingMask()
  let scene = PerceptionScene(
    segmenter: segmenter, instances: segmenter, identifier: ForegroundLabel())
  let frame = FrameSnapshot(
    image: CameraImage(width: 8, height: 8, rgba: Array(repeating: 255, count: 256)),
    depth: DepthImage(width: 8, height: 8, metres: Array(repeating: 2, count: 64)),
    camera: CameraGeometry(
      fx: 8, fy: 8, cx: 4, cy: 4, imageWidth: 8, imageHeight: 8,
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
  await scene.update(frame, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  _ = try await scene.foregroundSelection(tap: ImagePoint(x: 0.05, y: 0.5))
  let refreshed = try await scene.foregroundSelection(tap: nil)
  #expect(refreshed.entities.count == 1 && refreshed.surfaces.count == 1)
  let removed = try await scene.foregroundSelection(tap: ImagePoint(x: 0.05, y: 0.5))
  #expect(removed.entities.isEmpty && removed.surfaces.isEmpty)
}

private actor TemporaryMaskHole: InstanceSegmenting {
  private var visible = true
  func setVisible(_ value: Bool) { visible = value }
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    throw PerceptionError.unsupportedModel
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] {
    visible ? [SubjectMask(width: 4, height: 4, pixels: Array(repeating: 255, count: 16))] : []
  }
}
@Test func aMissingMaskNeverResetsTheAgeOfOldWorldPointsAndFreshDepthReplacesThem() async throws {
  let masks = TemporaryMaskHole()
  let scene = PerceptionScene(segmenter: masks, instances: masks, identifier: ForegroundLabel())
  let frame = FrameSnapshot(
    image: CameraImage(width: 4, height: 4, rgba: Array(repeating: 255, count: 64)),
    depth: DepthImage(width: 4, height: 4, metres: Array(repeating: 2, count: 16)),
    camera: CameraGeometry(
      fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4,
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
  await scene.update(frame, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  _ = try await scene.foregroundSelection(tap: ImagePoint(x: 0.5, y: 0.5))
  await masks.setVisible(false)
  await scene.update(
    frame, viewportToImage: [1, 0, 0, 1, 0, 0],
    capturedAt: 1 + PerceptionLimits.maximumSurfaceAge / 2)
  let held = try await scene.foregroundSelection(tap: nil)
  #expect(held.surfaces.first?.capturedAt == 1)
  await scene.update(
    frame, viewportToImage: [1, 0, 0, 1, 0, 0],
    capturedAt: 1 + PerceptionLimits.maximumSurfaceAge * 2)
  let expired = try await scene.foregroundSelection(tap: nil)
  #expect(expired.entities.count == 1 && expired.surfaces.isEmpty)
  await masks.setVisible(true)
  await scene.update(frame, viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 3)
  let fresh = try await scene.foregroundSelection(tap: nil)
  #expect(fresh.surfaces.first?.capturedAt == 3)
}
