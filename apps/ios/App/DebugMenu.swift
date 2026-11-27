import ARAnswer
import Foundation
import Globe
import Perception
import Router
import SwiftUI
import UniformTypeIdentifiers
import Voice

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
        CameraScreen(model: model, registry: registry, onCaptures: { showCaptures = true },
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
        #if canImport(Msplat)
          if testingEnabled {
            Section {
              NavigationLink {
                TrainBenchmarkScreen()
              } label: {
                Label("Benchmark train", systemImage: "timer")
              }
            }
          }
        #endif
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
      }
      // Standalone probes intentionally run real services; fixture automation uses the app seams.
      .disabled(LaunchMode.current.isUITest)
      .navigationTitle("Piece testing")
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
