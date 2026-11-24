import Capture
import EntitySchema
import Foundation

public struct FixtureTruth: Decodable, Sendable {
  public struct Object: Decodable, Sendable {
    public let label: String
    public let centre: [Double]
    public let extents: [Double]
  }
  public struct Probe: Decodable, Sendable {
    public let frame: Int
    public let label: String
    public let pixel: [Int]
    public let world: [Double]
  }
  public let synthetic: Bool
  public let objects: [Object]
  public let counts: [String: Int]
  public let treeHeight: Double
  public let normalFrames: Int
  public let probes: [Probe]
}

/// A fixture folder contains bundle/ and ground_truth.json. Reads committed depth,
/// confidence and poses. Semantic colours are reconstructed from ground-truth bounds;
/// this deliberately tests geometry and routing, not HEVC decoding or segmentation quality.
public struct FixtureReplay: Sendable {
  public let directory: URL
  public let truth: FixtureTruth
  public let frames: [CaptureFrame]
  public init(directory: URL) throws {
    self.directory = directory
    truth = try JSONDecoder().decode(
      FixtureTruth.self,
      from: Data(contentsOf: directory.appendingPathComponent("ground_truth.json")))
    guard truth.synthetic,
      truth.objects.allSatisfy({ $0.centre.count == 3 && $0.extents.count == 3 })
    else { throw PerceptionError.invalidGeometry }
    let data = try Data(contentsOf: directory.appendingPathComponent("bundle/frames.jsonl"))
    frames = try data.split(separator: 10).map {
      try JSONDecoder().decode(CaptureFrame.self, from: Data($0))
    }
  }
  public func snapshot(at index: Int) throws -> FrameSnapshot {
    guard frames.indices.contains(index), let ref = frames[index].depth,
      ref.width > 0, ref.height > 0, CaptureBundleWriter.safeName(ref.file),
      CaptureBundleWriter.safeName(ref.confidence)
    else { throw PerceptionError.invalidGeometry }
    let frame = frames[index]
    guard frame.transform.count == 16, frame.intrinsics.count == 9,
      frame.imageWidth > 0, frame.imageHeight > 0
    else { throw PerceptionError.invalidGeometry }
    let folder = directory.appendingPathComponent("bundle")
    let bytes = [UInt8](try Data(contentsOf: folder.appendingPathComponent(ref.file)))
    let confidence = [UInt8](try Data(contentsOf: folder.appendingPathComponent(ref.confidence)))
    guard bytes.count == ref.width * ref.height * 2, confidence.count == ref.width * ref.height
    else { throw PerceptionError.invalidGeometry }
    // Spelled out step by step: the iOS SDK's type checker times out on the one-expression form.
    var values = [Float](repeating: .infinity, count: confidence.count)
    for i in 0..<confidence.count where confidence[i] != 0 {
      let low = UInt16(bytes[i * 2])
      let high = UInt16(bytes[i * 2 + 1]) << 8
      values[i] = Float(Float16(bitPattern: low | high))
    }
    let sx = Double(ref.width) / Double(frame.imageWidth)
    let sy = Double(ref.height) / Double(frame.imageHeight)
    let fx: Double = frame.intrinsics[0] * sx
    let fy: Double = frame.intrinsics[4] * sy
    // The fixtures use integer pixel centres; MaskLifter uses normalized texel centres.
    let cx: Double = frame.intrinsics[2] * sx + 0.5
    let cy: Double = frame.intrinsics[5] * sy + 0.5
    var transform: [Double] = []
    transform.reserveCapacity(16)
    for col in 0..<4 {
      for row in 0..<4 { transform.append(frame.transform[row * 4 + col]) }
    }
    let camera = CameraGeometry(
      fx: fx, fy: fy, cx: cx, cy: cy,
      imageWidth: ref.width, imageHeight: ref.height, transform: transform)
    var rgba = [UInt8](repeating: 0, count: values.count * 4)
    for i in values.indices {
      let u: Double = (Double(i % ref.width) + 0.5) / Double(ref.width)
      let v: Double = (Double(i / ref.width) + 0.5) / Double(ref.height)
      let point = camera.worldPoint(pixel: .init(x: u, y: v), depth: Double(values[i]))
      var label = "sky"
      if let p = point {
        let coordinates: [Double] = [p.x, p.y, p.z]
        for object in truth.objects where Self.contains(object, coordinates) {
          label = object.label
          break
        }
      }
      let colour: [UInt8] = FakeSegmenter.palette[label] ?? [130, 180, 225]
      rgba.replaceSubrange(i * 4..<i * 4 + 4, with: colour + [255])
    }
    return FrameSnapshot(
      image: .init(width: ref.width, height: ref.height, rgba: rgba),
      depth: .init(width: ref.width, height: ref.height, metres: values), camera: camera)
  }

  /// Whether a world point lies inside an object's ground-truth box. Float16 depth at < 10 m
  /// rounds by <= 4 mm; 1 cm of slack covers that in world axes.
  static func contains(_ object: FixtureTruth.Object, _ point: [Double]) -> Bool {
    for axis in 0..<3 {
      let half: Double = object.extents[axis] / 2 + 0.01
      if abs(point[axis] - object.centre[axis]) > half { return false }
    }
    return true
  }
}

/// Known flat fixture colours, also usable on decoded RGB. Trunk and crown form one tree.
public struct FakeSegmenter: InstanceSegmenting, ImageIdentifying {
  public static let palette: [String: [UInt8]] = [
    "ground": [145, 120, 75], "shed": [190, 65, 45],
    "trunk": [95, 55, 25], "crown": [40, 145, 55], "sky": [130, 180, 225],
  ]
  public init() {}
  private func label(_ image: CameraImage, at index: Int) -> String {
    let colour = Array(image.rgba[index * 4..<index * 4 + 3])
    let nearest = Self.palette.min { a, b in
      zip(a.value, colour).reduce(0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
        < zip(b.value, colour).reduce(0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
    }!.key
    return ["trunk", "crown"].contains(nearest) ? "tree" : nearest
  }
  public func instances(_ image: CameraImage) async throws -> [SubjectMask] {
    guard image.width > 0, image.height > 0, image.rgba.count == image.width * image.height * 4
    else { throw PerceptionError.invalidGeometry }
    return ["shed", "tree"].compactMap { subject in
      let pixels = (0..<image.width * image.height).map {
        label(image, at: $0) == subject ? UInt8(255) : 0
      }
      return pixels.contains(255)
        ? SubjectMask(width: image.width, height: image.height, pixels: pixels) : nil
    }
  }
  public func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    let masks = try await instances(image)
    let point: ImagePoint
    switch prompt {
    case .point(let p): point = p
    case .box(let b): point = .init(x: (b.min.x + b.max.x) / 2, y: (b.min.y + b.max.y) / 2)
    }
    guard (0..<1).contains(point.x), (0..<1).contains(point.y),
      let mask = masks.first(where: {
        $0.pixels[Int(point.y * Double($0.height)) * $0.width + Int(point.x * Double($0.width))] > 0
      })
    else { throw PerceptionError.noSubject }
    return mask
  }
  public func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel? {
    guard mask.isValid, mask.width == image.width, mask.height == image.height,
      image.rgba.count == image.width * image.height * 4,
      let first = mask.pixels.firstIndex(where: { $0 > 0 })
    else { throw PerceptionError.invalidGeometry }
    return .init(label: label(image, at: first), confidence: 1)
  }
}

public protocol PerceptionServing: Sendable {
  func update(_ frame: FrameSnapshot, viewportToImage: [Double], capturedAt: TimeInterval) async
  func clear() async
  func selection() async -> SceneSelection
  func select(_ point: ImagePoint) async throws -> SceneSelection
  func foregroundSelection(tap: ImagePoint?) async throws -> SceneSelection
  func refreshSelection(isQuery: Bool) async throws -> SceneSelection
  func inspect(subject: String, count: Bool) async throws -> [Entity]
}
extension PerceptionScene: PerceptionServing {}
extension PerceptionServing {
  public func foregroundSelection(tap: ImagePoint?) async throws -> SceneSelection {
    if let tap { return try await select(tap) }
    return try await refreshSelection(isQuery: false)
  }
}

/// A fake service uses the real lifting/tracking code with deterministic semantic masks.
public struct FakePerception: PerceptionServing {
  private let scene: PerceptionScene
  public init() { scene = Self.makeScene() }
  public func update(_ frame: FrameSnapshot, viewportToImage: [Double], capturedAt: TimeInterval)
    async
  {
    await scene.update(frame, viewportToImage: viewportToImage, capturedAt: capturedAt)
  }
  public func clear() async { await scene.clear() }
  public func selection() async -> SceneSelection { await scene.selection() }
  public func select(_ point: ImagePoint) async throws -> SceneSelection {
    try await scene.select(point)
  }
  public func foregroundSelection(tap: ImagePoint?) async throws -> SceneSelection {
    try await scene.foregroundSelection(tap: tap)
  }
  public func refreshSelection(isQuery: Bool = false) async throws -> SceneSelection {
    try await scene.refreshSelection(isQuery: isQuery)
  }
  public func inspect(subject: String, count: Bool) async throws -> [Entity] {
    try await scene.inspect(subject: subject, count: count)
  }
  public static func makeScene() -> PerceptionScene {
    let fake = FakeSegmenter()
    return PerceptionScene(segmenter: fake, instances: fake, identifier: fake)
  }
}

#if canImport(SwiftUI)
  import SwiftUI
  public struct PerceptionDebugScreen: View {
    private let directory: URL?
    @State private var result = "Simulated fixture perception"
    public init(directory: URL? = nil) { self.directory = directory }
    public var body: some View {
      VStack {
        Text(result)
        Button("Replay yard fixture") {
          Task {
            guard let directory else {
              result = "Select a fixture directory in the debug menu"
              return
            }
            do {
              let replay = try FixtureReplay(directory: directory)
              let scene = FakePerception.makeScene()
              for frame in replay.frames where frame.trackingState == "normal" {
                await scene.update(
                  try replay.snapshot(at: frame.index), viewportToImage: [1, 0, 0, 1, 0, 0],
                  capturedAt: frame.timestamp)
              }
              let entities = try await scene.inspect(subject: "subject", count: true)
              result =
                "Simulated: "
                + entities.map { "\($0.label): \($0.extent.y)m visible height" }.joined(
                  separator: "\n")
            } catch { result = String(describing: error) }
          }
        }
      }.padding()
    }
  }
  #Preview {
    PerceptionDebugScreen(
      directory:
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/bundles/yard-small"))
  }
#endif
