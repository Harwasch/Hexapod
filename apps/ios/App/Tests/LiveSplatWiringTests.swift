import Foundation
import LiveSplat
import Perception
import Router
import Testing
import Voice

@testable import AppSupport

@Test @MainActor func liveSplatDefaultsToTrainingAndSwitchesWithoutReplacingRenderer() throws {
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
  #expect(renderer.status == .training)
  preference.trainingEnabled = false
  #expect(renderer.status == .pointPreview)
  #expect(try registry.splatRenderers.resolve("real") === renderer)
  preference.trainingEnabled = true
  #expect(renderer.status == .training)
}

@Test @MainActor func debugSplatOverridesWinOverUserPreference() throws {
  let suite = "hex.splat.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  let registry = PieceRegistry(defaults: defaults, liveSplatPreference: preference)
  useOfflineCameraProviders(registry)
  for enabled in [false, true] {
    preference.trainingEnabled = enabled
    registry.choose("off", for: "liveSplat")
    #expect(try registry.makeCamera(useTestChoices: true).liveSplat == nil)
    registry.choose("fake", for: "liveSplat")
    let fake = try #require(
      try registry.makeCamera(useTestChoices: true).liveSplat as? FakeLiveSplat)
    fake.start()
    #expect(fake.status == .training)
    preference.trainingEnabled.toggle()
    #expect(fake.status == .training)
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
