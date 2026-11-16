import ARAnswer
import Foundation
import Perception
import Router
import SwiftUI
import UniformTypeIdentifiers
import Voice

@MainActor struct CameraHost: View {
  @State private var registry = PieceRegistry()
  @State private var model: CameraModel?
  @State private var error = ""
  @State private var revision = UUID()
  @State private var showMenu = false
  @State private var showSettings = false
  @AppStorage("hex.testflight.debugEnabled") private var testFlightDebug = false
  private var isTestFlight: Bool {
    Bundle.main.appStoreReceiptURL?.lastPathComponent == "sandboxReceipt"
  }
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
    ZStack(alignment: .top) {
      if let model {
        CameraScreen(model: model).id(revision)
      } else {
        Text(error).padding().frame(maxWidth: .infinity, maxHeight: .infinity)
      }
      HStack {
        Text("Current scan")
          .padding(12).background(.ultraThinMaterial, in: Capsule())
          .onLongPressGesture { if menuEnabled { showMenu = true } }
          .accessibilityHint(menuEnabled ? "Long press for piece testing" : "")
        Spacer()
        if isTestFlight {
          Button {
            showSettings = true
          } label: {
            Image(systemName: "gearshape")
          }
          .accessibilityLabel("TestFlight settings")
        }
      }.padding()
    }
    .task { if model == nil { apply() } }
    .onChange(of: testFlightDebug) { _, _ in apply() }
    .sheet(isPresented: $showSettings) {
      Form { Toggle("Enable piece testing menu", isOn: $testFlightDebug) }
        .presentationDetents([.medium])
    }
    .sheet(isPresented: $showMenu) {
      PieceDebugMenu(registry: registry) {
        apply()
        showMenu = false
      }
    }
  }
  private func apply() {
    model?.suspend()
    model = nil
    do {
      model = CameraModel(providers: try registry.makeCamera(useTestChoices: menuEnabled))
      error = ""
    } catch { self.error = String(describing: error) }
    revision = UUID()
  }
}

/// The host gates access in DEBUG/TestFlight. Module screens require SwiftUI.
@MainActor struct PieceDebugMenu: View {
  let registry: PieceRegistry
  let apply: () -> Void
  @State private var choices: [String: String] = [:]
  @State private var result = ""
  @State private var importFixture = false
  var body: some View {
    NavigationStack {
      Form {
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
