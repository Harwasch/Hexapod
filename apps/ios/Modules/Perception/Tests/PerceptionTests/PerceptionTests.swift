import EntitySchema
import Foundation
import Testing

@testable import Perception

private let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
private func camera(_ transform: [Double] = identity) -> CameraGeometry {
  CameraGeometry(fx: 4, fy: 4, cx: 2, cy: 2, imageWidth: 4, imageHeight: 4, transform: transform)
}
private let fullMask = SubjectMask(width: 4, height: 4, pixels: [UInt8](repeating: 255, count: 16))
@Test func knownPlaneLiftsToARCoordinatesAndPercentileExtent() throws {
  let entity = try MaskLifter.lift(
    mask: fullMask, depth: .init(width: 4, height: 4, metres: [Float](repeating: 2, count: 16)),
    camera: camera(), label: "panel")
  #expect(entity.anchor.world == Vector3(x: 0, y: 0, z: -2))
  #expect(entity.extent == Vector3(x: 1.5, y: 1.5, z: 0))
}
@Test func translationRotationAndProjectionRoundTrip() throws {
  // 90 degrees about Y, translated by (3,1,4).
  let t: [Double] = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 3, 1, 4, 1]
  let cam = camera(t)
  let p = ImagePoint(x: 0.25, y: 0.75)
  let world = try #require(cam.worldPoint(pixel: p, depth: 2))
  #expect(world == Vector3(x: 1, y: 0.5, z: 4.5))
  #expect(cam.project(world) == p)
  #expect(cam.project(.init(x: 5, y: 1, z: 4)) == nil)
}
@Test func holesAndOutliersDoNotMoveMedianAnchor() throws {
  var depths = [Float](repeating: 2, count: 16)
  depths[0] = .nan
  depths[1] = 0
  depths[2] = .infinity
  depths[3] = 15
  let entity = try MaskLifter.lift(
    mask: fullMask, depth: .init(width: 4, height: 4, metres: depths),
    camera: camera(), label: "panel")
  #expect(entity.anchor.world.z == -2)
  #expect(entity.extent.z == 0)
  #expect(entity.confidence == 12.0 / 16.0)
}
@Test func malformedAndInsufficientDepthAreRejected() {
  #expect(throws: PerceptionError.self) {
    try MaskLifter.lift(
      mask: fullMask, depth: .init(width: 4, height: 4, metres: [0]), camera: camera(), label: "x")
  }
  #expect(throws: PerceptionError.self) {
    try MaskLifter.lift(
      mask: fullMask,
      depth: .init(width: 4, height: 4, metres: [Float](repeating: .nan, count: 16)),
      camera: camera(), label: "x")
  }
  #expect(SubjectMask(width: 0, height: 2, pixels: []).bounds == nil)
}
private func detection(_ x: Double, label: String = "tree", camera cam: CameraGeometry = camera())
  -> Detection
{
  let e = Entity(
    label: label, anchor: .init(world: .init(x: x, y: 0, z: -2)),
    extent: .init(x: 0.3, y: 0.5, z: 0.3),
    masks: [], confidence: 0.8, attrs: [:], source: .init(module: "test", model: "depth"))
  return Detection(entity: e, bounds: cam.projectedBounds(e)!)
}
@Test func associationPreservesIdentityAcrossOrderAndCameraMotion() throws {
  var tracker = EntityTracker()
  let original = tracker.update([detection(-0.3), detection(0.3)], camera: camera())
  var shifted = identity
  shifted[12] = 0.1
  let cam = camera(shifted)
  let updated = tracker.update(
    [detection(0.32, camera: cam), detection(-0.28, camera: cam)], camera: cam)
  #expect(updated.count == 2)
  #expect(updated[0].id == original[0].id && updated[0].entity.anchor.world.x == -0.28)
  #expect(updated[1].id == original[1].id && updated[1].entity.anchor.world.x == 0.32)
}
@Test func associationIsOneToOneAndExpiresStaleTracks() {
  var tracker = EntityTracker()
  tracker.maximumMissedFrames = 1
  let initial = tracker.update([detection(0)], camera: camera())
  let next = tracker.update(
    [detection(0.01), detection(0.02), detection(0, label: "rock")], camera: camera())
  #expect(next.count == 3)
  #expect(next.filter { $0.id == initial[0].id }.count == 1)
  _ = tracker.update([], camera: camera())
  #expect(tracker.update([], camera: camera()).isEmpty)
}
private actor Download: ModelDownloading {
  var calls = 0
  let data: Data
  init(_ data: Data) { self.data = data }
  func download(_ url: URL, maximumBytes: Int) async throws -> Data {
    calls += 1
    return data
  }
}
@Test func sha256KnownVectorsAndVerifiedCache() async throws {
  #expect(
    SHA256Digest.hex(Data()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
  let data = Data("abc".utf8)
  let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  #expect(SHA256Digest.hex(data) == digest)
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: directory) }
  let download = Download(data)
  let cache = ModelCache(directory: directory, downloader: download)
  let config = ModelConfiguration(
    url: URL(string: "https://example.invalid/model.mlmodel")!, sha256: digest)
  let path = try await cache.model(config)
  #expect(try await cache.model(config) == path)
  #expect(await download.calls == 1)
  try Data("corrupt".utf8).write(to: path)
  #expect(try await cache.model(config) == path)
  #expect(await download.calls == 2)
  let wrong = ModelConfiguration(url: config.url, sha256: String(repeating: "0", count: 64))
  await #expect(throws: PerceptionError.self) { try await cache.model(wrong) }
  #expect(
    !FileManager.default.fileExists(
      atPath: directory.appendingPathComponent(wrong.sha256 + ".mlmodel").path))
}
private struct FailingSegmenter: Segmenter {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    throw PerceptionError.unsupportedModel
  }
}
private struct FixedSegmenter: Segmenter {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask { fullMask }
}
@Test func missingModelFallsBackThroughInterface() async throws {
  let segmenter = FallbackSegmenter(primary: FailingSegmenter(), fallback: FixedSegmenter())
  #expect(
    try await segmenter.segment(
      .init(width: 1, height: 1, rgba: [0, 0, 0, 255]),
      prompt: .point(.init(x: 0.5, y: 0.5))) == fullMask)
}

@Test func depthAndMaskMayUseDifferentResolutions() throws {
  let mask = SubjectMask(width: 8, height: 8, pixels: [UInt8](repeating: 255, count: 64))
  let e = try MaskLifter.lift(
    mask: mask, depth: .init(width: 4, height: 4, metres: [Float](repeating: 2, count: 16)),
    camera: camera(), label: "panel")
  #expect(e.anchor.world == .init(x: 0, y: 0, z: -2))
  #expect(e.confidence == 1)
}
@Test func subjectIsolationRemovesBackgroundWithoutMovingCoordinates() throws {
  let image = CameraImage(width: 2, height: 1, rgba: [10, 20, 30, 255, 40, 50, 60, 255])
  let mask = SubjectMask(width: 2, height: 1, pixels: [255, 0])
  let isolated = try SubjectImage.isolated(image, mask: mask)
  #expect(isolated.rgba == [10, 20, 30, 255, 0, 0, 0, 255])
}
@Test func sha256MultiBlockVector() {
  #expect(
    SHA256Digest.hex(Data("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".utf8))
      == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
}

private struct SceneSegmenter: InstanceSegmenting {
  var masks: [SubjectMask] = [fullMask]
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask { masks[0] }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] { masks }
}
private struct SceneIdentifier: ImageIdentifying {
  var value: DeviceLabel? = DeviceLabel(label: "tree", confidence: 0.9)
  func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel? { value }
}
private func sceneFrame() -> FrameSnapshot {
  FrameSnapshot(
    image: .init(width: 4, height: 4, rgba: [UInt8](repeating: 255, count: 64)),
    depth: .init(width: 4, height: 4, metres: [Float](repeating: 2, count: 16)), camera: camera())
}
@Test func sceneSelectsVoiceSubjectAndKeepsMaskCameraPairing() async throws {
  let scene = PerceptionScene(
    segmenter: SceneSegmenter(), instances: SceneSegmenter(), identifier: SceneIdentifier())
  let transform: [Double] = [-1, 0, 0, 1, 1, 0]
  await scene.update(sceneFrame(), viewportToImage: transform, capturedAt: 10)
  let selected = try await scene.select(.init(x: 0.5, y: 0.5))
  #expect(selected.entities.first?.label == "tree")
  #expect(selected.entities.first?.attrs["identifiedOnDevice"] == .bool(true))
  #expect(selected.viewportToImage == transform && selected.capturedAt == 10)
  #expect(try await scene.inspect(subject: "trees", count: true).count == 1)
  #expect(try await scene.inspect(subject: "rock", count: true).isEmpty)
  await scene.clear()
  #expect(await scene.selection().entities.isEmpty)
}
@Test func sceneNeverAssignsVoiceLabelWithoutClassificationEvidence() async throws {
  let scene = PerceptionScene(
    segmenter: SceneSegmenter(), instances: SceneSegmenter(),
    identifier: SceneIdentifier(value: nil))
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  #expect(try await scene.inspect(subject: "oak", count: false).isEmpty)
  let selected = try await scene.select(.init(x: 0.5, y: 0.5))
  #expect(selected.entities.first?.label == "subject")
  #expect(selected.entities.first?.attrs["identifiedOnDevice"] == nil)
}
private actor SceneGate {
  var ready = false
  var arrived: CheckedContinuation<Void, Never>?
  var waiting: CheckedContinuation<Void, Never>?
  func block() async {
    ready = true
    arrived?.resume()
    arrived = nil
    await withCheckedContinuation { waiting = $0 }
  }
  func waitForArrival() async {
    if ready { return }
    await withCheckedContinuation { arrived = $0 }
  }
  func release() {
    waiting?.resume()
    waiting = nil
  }
}
private struct BlockingSegmenter: InstanceSegmenting {
  let gate: SceneGate
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    await gate.block()
    return fullMask
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] { [fullMask] }
}
@Test func clearingSceneInvalidatesAnInFlightSelection() async throws {
  let gate = SceneGate()
  let segmenter = BlockingSegmenter(gate: SceneGate())
  let blocking = BlockingSegmenter(gate: gate)
  let scene = PerceptionScene(
    segmenter: blocking, instances: segmenter, identifier: SceneIdentifier())
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  let pending = Task { try await scene.select(.init(x: 0.5, y: 0.5)) }
  await gate.waitForArrival()
  await scene.clear()
  await gate.release()
  await #expect(throws: CancellationError.self) { try await pending.value }
  #expect(await scene.selection().entities.isEmpty)
}

private actor IntermittentSegmenter: InstanceSegmenting {
  var calls = 0
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    calls += 1
    if calls == 2 { throw PerceptionError.noSubject }
    return fullMask
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] { [fullMask] }
}
@Test func sceneReacquiresATrackAfterATemporaryMaskHole() async throws {
  let segmenter = IntermittentSegmenter()
  let scene = PerceptionScene(
    segmenter: segmenter, instances: segmenter, identifier: SceneIdentifier())
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  _ = try await scene.select(.init(x: 0.5, y: 0.5))
  #expect(try await scene.refreshSelection().masks.isEmpty)
  #expect(try await scene.refreshSelection().entities.first?.label == "tree")
}

private struct TapSegmenter: InstanceSegmenting {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    guard case .point(let point) = prompt, point.y < 0.8 else { throw PerceptionError.noSubject }
    let left = point.x < 0.5
    return SubjectMask(
      width: 4, height: 4,
      pixels: (0..<16).map { i in
        ((i % 4 < 2) == left) ? 255 : 0
      })
  }
  func instances(_ image: CameraImage) async throws -> [SubjectMask] { [] }
}
@Test func tapsAccumulateToggleAndEmptySpaceClearsSelection() async throws {
  let scene = PerceptionScene(
    segmenter: TapSegmenter(), instances: TapSegmenter(), identifier: SceneIdentifier())
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  let first = try await scene.select(.init(x: 0.2, y: 0.3))
  #expect(first.entities.count == 1)
  let second = try await scene.select(.init(x: 0.7, y: 0.3))
  #expect(second.entities.count == 2 && second.masks.count == 2)
  let toggled = try await scene.select(.init(x: 0.2, y: 0.3))
  #expect(toggled.entities.count == 1)
  #expect(toggled.entities.first?.anchor.world.x == second.entities.last?.anchor.world.x)
  let empty = try await scene.select(.init(x: 0.7, y: 0.9))
  #expect(empty.entities.isEmpty && empty.masks.isEmpty)
}

@Test func multiSelectionRefreshRetainsOrderAndTogglingAfterCameraUpdateRemovesOnlyOne()
  async throws
{
  let scene = PerceptionScene(
    segmenter: TapSegmenter(), instances: TapSegmenter(), identifier: SceneIdentifier())
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 1)
  _ = try await scene.select(.init(x: 0.2, y: 0.3))
  _ = try await scene.select(.init(x: 0.7, y: 0.3))
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 2)
  let refreshed = try await scene.refreshSelection()
  #expect(refreshed.entities.count == 2 && refreshed.masks.count == 2)
  #expect(refreshed.entities[0].anchor.world.x < refreshed.entities[1].anchor.world.x)
  await scene.update(sceneFrame(), viewportToImage: [1, 0, 0, 1, 0, 0], capturedAt: 3)
  let toggled = try await scene.select(.init(x: 0.2, y: 0.3))
  #expect(toggled.entities.count == 1 && toggled.entities[0].anchor.world.x > 0)
}
