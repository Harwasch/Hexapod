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

@Test @MainActor func previewChoicesPersistIndependentlyAndSeedAnisotropicReference() async throws {
  let domain = "hex.preview.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: domain))
  defer { defaults.removePersistentDomain(forName: domain) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(
    preference.previewStyle == .discs && !preference.insetEnabled && preference.previewEnabled)
  preference.previewStyle = .splat
  preference.insetEnabled = true
  let restored = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(restored.previewStyle == .splat && restored.insetEnabled)
  #expect(restored.previewEnabled)
  preference.insetEnabled = false
  #expect(restored.previewStyle == .splat && !restored.insetEnabled)
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  let source = try FakeFrameSource.plane(frameCount: 2)
  renderer.ingest(try #require(await source.next()))
  let g = try #require(renderer.splats.first)
  #expect(g.covariance[2, 2] < g.covariance[0, 0])
  #expect(renderer.status == .scanPreview)
  preference.previewStyle = .discs
  renderer.ingest(try #require(await source.next()))
  let disc = try #require(renderer.splats.first)
  #expect(disc.covariance[2, 2] == disc.covariance[0, 0])
  renderer.stop()
}

@Test @MainActor func coverageToggleUsesSharedPersistedPreferenceWithoutReplacingCameraRenderer()
  throws
{
  let suite = "hex.coverage.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  let registry = PieceRegistry(defaults: defaults, liveSplatPreference: preference)
  let renderer = try #require(
    try registry.splatRenderers.resolve("real") as? DefaultLiveSplatRenderer)
  #expect(!renderer.preference.coverageEnabled)
  renderer.start()
  preference.coverageEnabled = true
  let restored = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(restored.coverageEnabled && renderer.preference.coverageEnabled)
  #expect(restored.previewEnabled && !restored.insetEnabled && restored.previewStyle == .discs)
  #expect(try registry.splatRenderers.resolve("real") === renderer)
  #expect(renderer.status == .scanPreview)
  preference.coverageEnabled = false
  #expect(!restored.coverageEnabled)
  renderer.stop()
}
