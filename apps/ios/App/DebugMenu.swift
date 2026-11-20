import ARAnswer
import Foundation
import Globe
import Perception
import Router
import SwiftUI
import UniformTypeIdentifiers
import Voice

#if DEBUG
  /// Explicit UI launches intercept control requests before URLSession.shared starts.
  /// The server reports an already uploaded fixture, exercising the uploader's resume
  /// path without a background object-store transfer or any production API request.
  final class UIFixtureAPIProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool {
      UIFixtureLaunch.active && ["https", "http"].contains(request.url?.scheme ?? "")
    }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
      guard let url = request.url else { return }
      let path = url.path
      let captureID = FakeGlobeDataSource.scanID.uuidString.lowercased()
      let siteID = FakeGlobeDataSource.siteID.uuidString.lowercased()
      var status = 200
      let payload: Any
      if path.contains("/captures/bundles/"),
        (body()["siteId"] as? String)?.lowercased() == siteID
      {
        payload = ["id": captureID]
      } else if path.hasSuffix("/captures/" + captureID + "/bundle-upload"),
        let bytes = body()["bytes"] as? NSNumber
      {
        payload = ["id": "00000000-0000-0000-0000-000000000005", "status": "complete", "bytes": bytes] as [String: Any]
      } else if path.hasSuffix("/captures/" + captureID + "/process-bundle") {
        payload = ["id": "00000000-0000-0000-0000-000000000006"]
      } else if path.hasSuffix("/captures/" + captureID) {
        payload = ["status": "complete", "siteId": siteID]
      } else if path.hasSuffix("/sites") {
        payload = [["id": siteID, "name": "Simulated property"]]
      } else {
        // Unknown requests fail locally. They must never fall through to the network.
        status = 503
        payload = ["detail": "Fixture unavailable"]
      }
      guard let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": "application/json"]),
        let data = try? JSONSerialization.data(withJSONObject: payload)
      else { return }
      client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      client?.urlProtocol(self, didLoad: data)
      client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { /* Responses are synchronous and own no task. */ }
    private func body() -> [String: Any] {
      var data = request.httpBody ?? Data()
      if data.isEmpty, let stream = request.httpBodyStream {
        stream.open()
        defer { stream.close() }
        var bytes = [UInt8](repeating: 0, count: 4096)
        while data.count < 65_536 {
          let count = stream.read(&bytes, maxLength: bytes.count)
          guard count > 0 else { break }
          data.append(contentsOf: bytes.prefix(count))
        }
      }
      return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
  }
#endif

@MainActor struct CameraHost: View {
  let registry: PieceRegistry
  @State private var model: CameraModel?
  @State private var revision = UUID()
  @State private var showSettings = false
  @State private var showCaptures = false
  @State private var showGlobe = false
  @AppStorage("hex.testflight.debugEnabled") private var testFlightDebug = false
  private var menuEnabled: Bool {
    #if DEBUG
      let debug = true
    #else
      let debug = false
    #endif
    return PieceRegistry.menuAvailable(
      debug: debug, receipt: Bundle.main.appStoreReceiptURL?.lastPathComponent,
      enabled: testFlightDebug)
  }
  var body: some View {
    Group {
      if let model {
        CameraScreen(model: model, onCaptures: { showCaptures = true },
          onSettings: { showSettings = true }, onGlobe: { showGlobe = true }).id(revision)
      } else {
        VStack(spacing: 20) {
          Label("Camera unavailable", systemImage: "camera.fill")
          Button("Retry") { apply() }.frame(minHeight: 44)
          Button("Captures", systemImage: "square.stack") { showCaptures = true }
            .frame(minHeight: 44).accessibilityIdentifier("home.captures")
          Button("Globe", systemImage: "globe.americas") { showGlobe = true }
            .frame(minHeight: 44).accessibilityIdentifier("home.globe")
          Button("Settings", systemImage: "gearshape") { showSettings = true }
            .frame(minHeight: 44).accessibilityIdentifier("home.settings")
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }
    .onAppear { apply() }
    .onChange(of: testFlightDebug) { _, _ in apply() }
    .navigationDestination(isPresented: $showSettings) { SettingsScreen(registry: registry) }
    .navigationDestination(isPresented: $showCaptures) { CapturesScreen(registry: registry) }
    .navigationDestination(isPresented: $showGlobe) { GlobeScreen(registry: registry) }
  }
  private func apply() {
    let pendingCapture = model?.savedCapture
    let duration = model?.captureDuration ?? 0
    model?.suspend()
    model = nil
    do {
      let refreshed = PieceRegistry(defaults: registry.defaults, configuration: registry.configuration,
        liveSplatPreference: registry.liveSplatPreference)
      model = CameraModel(providers: try refreshed.makeCamera(useTestChoices: menuEnabled))
      if UIFixtureLaunch.active {
        // The recorded replay has no device GPS. An explicit fixture site supplies
        // placement without changing the user's saved current-site preferences.
        model?.siteID = FakeGlobeDataSource.siteID
        model?.siteName = "Simulated property"
      }
      model?.savedCapture = pendingCapture
      model?.captureDuration = duration
    } catch { model = nil }
    revision = UUID()
  }
}

/// The host gates access in DEBUG/TestFlight. Module screens require SwiftUI.
@MainActor struct PieceDebugMenu: View {
  let registry: PieceRegistry
  let apply: () -> Void
  @State private var choices: [String: String] = [:]
  @State private var result = ""
  @AppStorage("hex.testflight.debugEnabled") private var testingEnabled = false
  @AppStorage("globeImplementation") private var globeImplementation = GlobeImplementation.defaultValue.rawValue
  @State private var importFixture = false
  var body: some View {
    NavigationStack {
      Form {
        if Bundle.main.appStoreReceiptURL?.lastPathComponent == "sandboxReceipt" {
          Toggle("Enable testing", isOn: $testingEnabled)
        }
        Section("Independent providers · fake data is Simulated") {
          ForEach(PieceRegistry.options, id: \.key) { option in
            Picker(
              option.title,
              selection: Binding(
                get: { choices[option.key] ?? registry.choice(option.key) },
                set: {
                  choices[option.key] = $0
                  registry.choose($0, for: option.key)
                }
              )
            ) { ForEach(option.choices, id: \.self) { Text($0).tag($0) } }
          }
          Button("Import selected fixture folder") { importFixture = true }
          Text(
            "For replay, select yard-small or yard-limited above, then import that folder (bundle/ and ground_truth.json)."
          )
          Button("Apply camera providers", action: apply)
          Button("Use real providers") {
            for (key, value) in [
              "capture": "live", "voice": "real", "planner": "real", "perception": "real",
              "deep": registry.configuration.cloudConfigured ? "cloud" : "offline",
              "liveSplat": "real", "globe": "real", "overlay": "real",
            ] { registry.choose(value, for: key) }
            choices = [:]
            apply()
          }
        }
        Section("Globe renderer") {
          // Cesium (bundled web globe) is the default; Apple Maps 3D is the second
          // implementation behind the same Globe protocol. GlobeView reads this key.
          Picker("Renderer", selection: $globeImplementation) {
            ForEach(GlobeImplementation.allCases, id: \.rawValue) { Text($0.rawValue).tag($0.rawValue) }
          }
          NavigationLink("Cesium globe with fixture data") { GlobeDebugView() }
        }
        Section("Run pieces alone") {
          NavigationLink("Voice") { VoiceDebugScreen() }
          NavigationLink("Router") { RouterDebugScreen() }
          NavigationLink("Perception") {
            PerceptionDebugScreen(directory: registry.fixtureDirectory)
          }
          NavigationLink("ARAnswer") { ARAnswerDebugScreen() }
          Button("Probe selected LiveSplat") {
            Task {
              do {
                result = try await registry.liveSplat.resolve(registry.choice("liveSplat")).run()
              } catch { result = String(describing: error) }
            }
          }
          Button("Probe selected Globe data") {
            Task {
              do { result = try await registry.globe.resolve(registry.choice("globe")).run() } catch
              { result = String(describing: error) }
            }
          }
          Text(result)
        }
      }.navigationTitle("Piece testing")
    }
    .fileImporter(isPresented: $importFixture, allowedContentTypes: [.folder]) { selection in
      do {
        let source = try selection.get()
        let access = source.startAccessingSecurityScopedResource()
        defer { if access { source.stopAccessingSecurityScopedResource() } }
        let name = registry.choice("capture")
        guard ["yard-small", "yard-limited"].contains(name) else {
          throw ProviderError.fixtureNotSelected
        }
        // Copy once; future launches do not rely on a transient security-scoped URL.
        let destination = FileManager.default.urls(
          for: .applicationSupportDirectory, in: .userDomainMask)[0]
          .appendingPathComponent("DebugFixtures/\(UUID().uuidString)")
        try FileManager.default.createDirectory(
          at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: destination)
        _ = try FixtureReplay(directory: destination)
        registry.defaults.set(destination.path, forKey: "hex.fixture.\(name)")
        result = "Imported Simulated \(name)"
      } catch { result = String(describing: error) }
    }
  }
}
