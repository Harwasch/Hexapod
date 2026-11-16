import ARAnswer
import EntitySchema
import Foundation
import Perception
import Router
import Voice

protocol CameraCaptureSource: Sendable {
  var usesLiveSession: Bool { get }
  func frames() -> AsyncThrowingStream<FrameSnapshot, Error>
}
struct LiveCaptureSource: CameraCaptureSource {
  let usesLiveSession = true
  func frames() -> AsyncThrowingStream<FrameSnapshot, Error> { AsyncThrowingStream { $0.finish() } }
}
struct FixtureCaptureSource: CameraCaptureSource {
  let directory: URL
  let usesLiveSession = false
  func frames() -> AsyncThrowingStream<FrameSnapshot, Error> {
    AsyncThrowingStream { c in
      let task = Task.detached {
        do {
          let replay = try FixtureReplay(directory: directory)
          for frame in replay.frames where frame.trackingState == "normal" {
            try Task.checkCancellation()
            c.yield(try replay.snapshot(at: frame.index))
            try await Task.sleep(for: .milliseconds(83))
          }
          c.finish()
        } catch { c.finish(throwing: error) }
      }
      c.onTermination = { _ in task.cancel() }
    }
  }
}

/// Local plug-in seam until the sibling modules merge. A provider owns its renderer
/// or source and reports the result of starting it. The menu can exercise it alone.
@MainActor protocol DebugPieceService: AnyObject {
  func run() async throws -> String
}
@MainActor final class FakeDebugPiece: DebugPieceService {
  private let label: String
  private(set) var runs = 0
  init(_ label: String) { self.label = label }
  func run() async throws -> String {
    runs += 1
    return "Simulated \(label), run \(runs)"
  }
}
@MainActor final class DisabledDebugPiece: DebugPieceService {
  func run() async throws -> String { "Off" }
}

enum ProviderError: Error, CustomStringConvertible {
  case unavailable(String)
  case fixtureNotSelected
  var description: String {
    switch self {
    case .unavailable(let name): "Provider unavailable: \(name)"
    case .fixtureNotSelected: "Choose a fixture folder containing bundle/ and ground_truth.json"
    }
  }
}

/// Typed, open registration. Registering a sibling implementation is one line;
/// selecting an unavailable provider throws visibly instead of substituting a fake.
@MainActor final class PieceProviders<Value> {
  private var factories: [String: () throws -> Value] = [:]
  func register(_ name: String, _ factory: @escaping () throws -> Value) {
    factories[name] = factory
  }
  func resolve(_ name: String) throws -> Value {
    guard let factory = factories[name] else { throw ProviderError.unavailable(name) }
    return try factory()
  }
  func contains(_ name: String) -> Bool { factories[name] != nil }
}

@MainActor struct CameraProviders {
  let voice: any VoiceInput
  let planner: any Planning
  let perception: any PerceptionServing
  let deep: any DeepAnswering
  let capture: any CameraCaptureSource
  let fakeOverlay: Bool
  let simulated: Bool
}

@MainActor final class PieceRegistry {
  let voice = PieceProviders<any VoiceInput>()
  let planner = PieceProviders<any Planning>()
  let perception = PieceProviders<any PerceptionServing>()
  let deep = PieceProviders<any DeepAnswering>()
  let capture = PieceProviders<any CameraCaptureSource>()
  let liveSplat = PieceProviders<any DebugPieceService>()
  let globe = PieceProviders<any DebugPieceService>()
  let defaults: UserDefaults
  static let options: [(key: String, title: String, choices: [String])] = [
    ("capture", "Capture source", ["live", "yard-small", "yard-limited"]),
    ("voice", "Voice", ["real", "fake"]),
    ("planner", "Planner", ["real", "rules", "fake"]),
    ("perception", "Perception", ["real", "fake"]),
    ("deep", "Deep answer", ["offline", "fake", "cloud"]),
    ("liveSplat", "Live splat", ["off", "fake", "real"]),
    ("globe", "Globe data", ["fake", "real"]),
    ("overlay", "Answer overlay", ["real", "fake"]),
  ]
  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    defaults.register(defaults: [
      "hex.piece.capture": "live", "hex.piece.voice": "real", "hex.piece.planner": "real",
      "hex.piece.perception": "real", "hex.piece.deep": "offline", "hex.piece.liveSplat": "off",
      "hex.piece.globe": "fake", "hex.piece.overlay": "real",
    ])
    voice.register("fake") { FakeVoice() }
    planner.register("fake") { FakePlanner() }
    planner.register("rules") { RulePlanner() }
    perception.register("fake") { FakePerception() }
    deep.register("offline") { OfflineDeepAnswer() }
    deep.register("fake") { FakeDeepAnswer() }
    capture.register("live") { LiveCaptureSource() }
    for name in ["yard-small", "yard-limited"] {
      capture.register(name) { [weak self] in
        guard let path = self?.defaults.string(forKey: "hex.fixture.\(name)") else {
          throw ProviderError.fixtureNotSelected
        }
        return FixtureCaptureSource(directory: URL(fileURLWithPath: path))
      }
    }
    liveSplat.register("off") { DisabledDebugPiece() }
    let fakeSplat = FakeDebugPiece("live splat")
    let fakeGlobe = FakeDebugPiece("globe data")
    liveSplat.register("fake") { fakeSplat }
    globe.register("fake") { fakeGlobe }
    #if os(iOS)
      voice.register("real") { SpeechVoice() }
      planner.register("real") { ApplePlanner() }
      perception.register("real") {
        PerceptionScene(
          segmenter: RuntimeSegmenter(), instances: VisionSegmenter(),
          identifier: VisionIdentifier())
      }
    #endif
    // TODO(merge): deep.register("cloud") { CloudAnswerer(...) }
    // TODO(merge): liveSplat.register("real") { LiveSplatDebugAdapter(renderer: ...) }
    // TODO(merge): globe.register("real") { GlobeDebugAdapter(source: ...) }
  }
  func choice(_ key: String) -> String { defaults.string(forKey: "hex.piece.\(key)") ?? "" }
  func choose(_ value: String, for key: String) { defaults.set(value, forKey: "hex.piece.\(key)") }
  func makeCamera(useTestChoices: Bool = true) throws -> CameraProviders {
    // Stored TestFlight choices must not leak into a later App Store launch or
    // remain active after the testing setting is disabled. Preserve them for re-entry.
    let selected: (String) -> String = { key in
      if useTestChoices { return self.choice(key) }
      return [
        "capture": "live", "voice": "real", "planner": "real", "perception": "real",
        "deep": "offline", "overlay": "real",
      ][key] ?? ""
    }
    return try CameraProviders(
      voice: voice.resolve(selected("voice")), planner: planner.resolve(selected("planner")),
      perception: perception.resolve(selected("perception")), deep: deep.resolve(selected("deep")),
      capture: capture.resolve(selected("capture")), fakeOverlay: selected("overlay") == "fake",
      simulated: selected("capture") != "live"
        || ["voice", "planner", "perception", "deep", "overlay"].contains { selected($0) == "fake" }
    )
  }
  var fixtureDirectory: URL? {
    defaults.string(forKey: "hex.fixture.\(choice("capture"))").map { URL(fileURLWithPath: $0) }
  }
  /// DEBUG or an actual TestFlight receipt plus an explicit in-app setting.
  static func menuAvailable(debug: Bool, receipt: String?, enabled: Bool) -> Bool {
    debug || (receipt == "sandboxReceipt" && enabled)
  }
}
