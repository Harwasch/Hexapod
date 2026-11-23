import Foundation
import LiveSplat
import Perception
import Router
import Testing
import Voice

@testable import AppSupport

@Test @MainActor func liveSplatDefaultsToPreviewAndSwitchesWithoutReplacingRenderer() throws {
  let suite = "hex.splat.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  let registry = PieceRegistry(defaults: defaults, liveSplatPreference: preference)
  useOfflineCameraProviders(registry)
  let renderer = try #require(
    try registry.makeCamera(useTestChoices: true).liveSplat as? DefaultLiveSplatRenderer)
  #expect(renderer.preference === registry.liveSplatPreference)
  #expect(renderer.preference === preference)
  renderer.start()
  defer { renderer.stop() }
  #expect(renderer.status == .scanPreview)
  preference.previewEnabled = false
  #expect(renderer.status == .coverageOnly)
  #expect(try registry.splatRenderers.resolve("real") === renderer)
  preference.previewEnabled = true
  #expect(renderer.status == .scanPreview)
}

@Test @MainActor func debugSplatOverridesWinOverUserPreference() throws {
  let suite = "hex.splat.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  let registry = PieceRegistry(defaults: defaults, liveSplatPreference: preference)
  useOfflineCameraProviders(registry)
  for enabled in [false, true] {
    preference.previewEnabled = enabled
    registry.choose("off", for: "liveSplat")
    #expect(try registry.makeCamera(useTestChoices: true).liveSplat == nil)
    registry.choose("fake", for: "liveSplat")
    let fake = try #require(
      try registry.makeCamera(useTestChoices: true).liveSplat as? FakeLiveSplat)
    fake.start()
    #expect(fake.status == .scanPreview)
    preference.previewEnabled.toggle()
    #expect(fake.status == .scanPreview)
    fake.stop()
  }
}

@Test @MainActor func productionRegistriesShareLiveSplatPreference() throws {
  let first = PieceRegistry()
  let second = PieceRegistry()
  #expect(first.liveSplatPreference === LiveSplatPreference.shared)
  #expect(first.liveSplatPreference === second.liveSplatPreference)
}

@MainActor private func useOfflineCameraProviders(_ registry: PieceRegistry) {
  registry.voice.register("real") { FakeVoice() }
  registry.planner.register("real") { RulePlanner() }
  registry.perception.register("real") { FakePerception() }
}
