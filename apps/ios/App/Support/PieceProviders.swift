import ARAnswer
import Cloud
import EntitySchema
import Foundation
import Globe
import LiveSplat
import Perception
import Router
import Voice

protocol CameraCaptureSource: Sendable {
  var usesLiveSession: Bool { get }
  var fixtureDirectory: URL? { get }
  func frames() -> AsyncThrowingStream<FrameSnapshot, Error>
}
extension CameraCaptureSource { var fixtureDirectory: URL? { nil } }

struct LiveCaptureSource: CameraCaptureSource {
  let usesLiveSession = true
  func frames() -> AsyncThrowingStream<FrameSnapshot, Error> { AsyncThrowingStream { $0.finish() } }
}
struct FixtureCaptureSource: CameraCaptureSource {
  let directory: URL
  var fixtureDirectory: URL? { directory }
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

/// A provider owns its renderer
/// or source and reports the result of starting it. The menu can exercise it alone.
@MainActor protocol DebugPieceService: AnyObject {
  func run() async throws -> String
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
  let liveSplat: (any LiveSplatRenderer)?
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
  let splatRenderers = PieceProviders<(any LiveSplatRenderer)?>()
  let globeSources = PieceProviders<any GlobeDataSource>()
  let configuration: ServiceConfiguration
  let defaults: UserDefaults
  let liveSplatPreference: LiveSplatPreference
  static let options: [(key: String, title: String, choices: [String])] = [
    ("capture", "Capture source", ["live", "yard-small", "yard-limited"]),
    ("voice", "Voice", ["real", "fake"]),
    ("planner", "Planner", ["real", "rules", "fake"]),
    ("perception", "Perception", ["real", "fake"]),
    ("deep", "Deep answer", ["offline", "fake", "cloud"]),
    ("liveSplat", "Scan preview", ["off", "fake", "real"]),
    ("globe", "Globe data", ["fake", "real"]),
    ("overlay", "Answer overlay", ["real", "fake"]),
  ]
  init(
    defaults: UserDefaults = .standard, configuration: ServiceConfiguration? = nil,
    liveSplatPreference: LiveSplatPreference = .shared
  ) {
    self.liveSplatPreference = liveSplatPreference
    self.defaults = defaults
    let config = configuration ?? ServiceConfiguration(defaults: defaults)
    self.configuration = config
    defaults.register(defaults: [
      "hex.piece.capture": "live", "hex.piece.voice": "real", "hex.piece.planner": "real",
      "hex.piece.perception": "real",
      "hex.piece.deep": config.cloudConfigured ? "cloud" : "offline", "hex.piece.liveSplat": "real",
      "hex.piece.globe": "real", "hex.piece.overlay": "real",
    ])
    voice.register("fake") { FakeVoice() }
    planner.register("fake") { FakePlanner() }
    planner.register("rules") { RulePlanner() }
    perception.register("fake") { FakePerception() }
    deep.register("offline") { OfflineDeepAnswer() }
    deep.register("fake") { try FakeDeepAnswering.yardSmall() }
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
    #if os(iOS)
      voice.register("real") { SpeechVoice() }
      planner.register("real") { ApplePlanner() }
      perception.register("real") {
        PerceptionScene(
          segmenter: RuntimeSegmenter(), instances: VisionSegmenter(),
          identifier: VisionIdentifier())
      }
    #endif
    deep.register("cloud") {
      guard let base = config.cloudBase, config.cloudConfigured else {
        throw ProviderError.unavailable("cloud configuration")
      }
      return CloudAnswerer(baseURL: base, writeToken: config.writeToken)
    }
    // Developer off/fake selects the provider first. Only real follows the user setting.
    // Keep one real camera renderer across provider reapplication; probes stay isolated.
    let realSplat = DefaultLiveSplatRenderer(preference: liveSplatPreference)
    splatRenderers.register("off") { nil }
    splatRenderers.register("fake") {
      FakeLiveSplat(preference: LiveSplatPreference(store: DebugSplatPreferenceStore()))
    }
    splatRenderers.register("real") { realSplat }
    liveSplat.register("real") {
      SplatProbe(renderer: DefaultLiveSplatRenderer(preference: liveSplatPreference))
    }
    liveSplat.register("fake") {
      SplatProbe(
        renderer: FakeLiveSplat(preference: LiveSplatPreference(store: DebugSplatPreferenceStore()))
      )
    }
    globeSources.register("fake") { FakeGlobeDataSource() }
    globeSources.register("real") {
      guard let root = config.apiRoot else { throw ProviderError.unavailable("API URL") }
      return APIGlobeDataSource(baseURL: root)
    }
    globe.register("real") { [globeSources] in
      GlobeProbe(source: try globeSources.resolve("real"))
    }
    globe.register("fake") { GlobeProbe(source: FakeGlobeDataSource(), simulated: true) }
  }
  func choice(_ key: String) -> String { defaults.string(forKey: "hex.piece.\(key)") ?? "" }
  func choose(_ value: String, for key: String) { defaults.set(value, forKey: "hex.piece.\(key)") }
  func makeCamera(useTestChoices: Bool = false) throws -> CameraProviders {
    #if DEBUG
      // UI automation is an explicit DEBUG launch; this path is absent from Release.
      if let index = ProcessInfo.processInfo.arguments.firstIndex(of: "-uiTestFixture"),
        ProcessInfo.processInfo.arguments.indices.contains(index + 1),
        ProcessInfo.processInfo.arguments[index + 1] == "yard-small",
        let directory = Bundle.main.url(
          forResource: "yard-small", withExtension: nil,
          subdirectory: "UITestFixtures")
      {
        return CameraProviders(
          voice: FakeVoice(), planner: RulePlanner(), perception: FakePerception(),
          liveSplat: try splatRenderers.resolve("fake"), deep: try FakeDeepAnswering.yardSmall(),
          capture: FixtureCaptureSource(directory: directory), fakeOverlay: true, simulated: true)
      }
    #endif
    // Stored TestFlight choices must not leak into a later App Store launch or
    // remain active after the testing setting is disabled. Preserve them for re-entry.
    let selected: (String) -> String = { key in
      if useTestChoices { return self.choice(key) }
      return [
        "capture": "live", "voice": "real", "planner": "real", "perception": "real",
        "deep": self.configuration.cloudConfigured ? "cloud" : "offline", "overlay": "real",
        "liveSplat": "real",
      ][key] ?? ""
    }
    return try CameraProviders(
      voice: voice.resolve(selected("voice")), planner: planner.resolve(selected("planner")),
      perception: perception.resolve(selected("perception")),
      liveSplat: splatRenderers.resolve(selected("liveSplat")),
      deep: deep.resolve(selected("deep")),
      capture: capture.resolve(selected("capture")), fakeOverlay: selected("overlay") == "fake",
      simulated: selected("capture") != "live"
        || ["voice", "planner", "perception", "deep", "overlay", "liveSplat"].contains {
          selected($0) == "fake"
        }
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

/// The API origin (or /api/v1 root) and write token are supplied by the host's settings.
struct ServiceConfiguration: Sendable {
  let baseURL: URL?
  private let tokenOverride: String?
  var writeToken: String { tokenOverride ?? APICredentials.token }
  let webBaseURL: URL?
  init(baseURL: URL?, writeToken: String = "", webBaseURL: URL? = nil) {
    self.baseURL = baseURL
    self.tokenOverride = writeToken
    self.webBaseURL = webBaseURL
  }
  init(defaults: UserDefaults) {
    func setting(_ name: String, _ info: String) -> String {
      let value =
        defaults.string(forKey: name)
        ?? (Bundle.main.object(forInfoDictionaryKey: info) as? String ?? "")
      return value.contains("$(") ? "" : value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    baseURL = URL(string: setting("hex.api.baseURL", "HEX_API_BASE_URL"))
    tokenOverride = nil
    webBaseURL = URL(string: setting("hex.web.baseURL", "HEX_WEB_BASE_URL"))
  }
  var cloudConfigured: Bool { cloudBase != nil && !writeToken.isEmpty }
  var cloudBase: URL? {
    guard var url = baseURL, ["http", "https"].contains(url.scheme), url.host != nil else {
      return nil
    }
    if url.lastPathComponent == "v1" { url.deleteLastPathComponent() }
    return url
  }
  var apiRoot: URL? {
    guard let base = cloudBase else { return nil }
    return base.lastPathComponent == "api"
      ? base.appendingPathComponent("v1") : base.appendingPathComponent("api/v1")
  }
}

@MainActor final class SplatProbe: DebugPieceService {
  let renderer: any LiveSplatRenderer
  init(renderer: any LiveSplatRenderer) {
    self.renderer = renderer
  }
  func run() async throws -> String {
    renderer.start()
    defer { renderer.stop() }
    let source = try FakeFrameSource.plane(frameCount: 1)
    if let event = try await source.next() { renderer.ingest(event) }
    return "Simulated probe · \(renderer.coverageFraction)"
  }
}
@MainActor final class GlobeProbe: DebugPieceService {
  let source: any GlobeDataSource
  let simulated: Bool
  init(source: any GlobeDataSource, simulated: Bool = false) {
    self.source = source
    self.simulated = simulated
  }
  func run() async throws -> String {
    let sites = try await source.sites()
    return "\(simulated ? "Simulated · " : "")\(sites.count)"
  }
}

/// Simulated scan preview is a developer override, independent of the persisted user setting.
@MainActor private final class DebugSplatPreferenceStore: LiveSplatPreferenceStore {
  private var enabled = true
  func bool(forKey key: String) -> Bool? { enabled }
  func set(_ value: Bool, forKey key: String) { enabled = value }
}
