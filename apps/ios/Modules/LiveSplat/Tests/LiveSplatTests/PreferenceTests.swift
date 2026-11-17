import Foundation
import Testing

@testable import LiveSplat

@MainActor private final class MemoryPreferenceStore: LiveSplatPreferenceStore {
  var values: [String: Bool] = [:]
  func bool(forKey key: String) -> Bool? { values[key] }
  func set(_ value: Bool, forKey key: String) { values[key] = value }
}

@Test @MainActor func trainingIsOnByDefaultAndPreferenceRoundTrips() {
  let store = MemoryPreferenceStore()
  let preference = LiveSplatPreference(store: store)
  #expect(preference.trainingEnabled)
  #expect(LiveSplatConfiguration().mode == .training)
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  #expect(renderer.status == .training)
  preference.trainingEnabled = false
  #expect(store.values["hex.liveSplat.training"] == false)
  #expect(!LiveSplatPreference(store: store).trainingEnabled)
  preference.trainingEnabled = true
  #expect(LiveSplatPreference(store: store).trainingEnabled)
}

@Test @MainActor func preferenceSwitchesRunningRendererWithoutResettingScan() async throws {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  let source = try FakeFrameSource.plane()
  renderer.ingest(try #require(await source.next()))
  let session = renderer.sessionGeneration
  let splats = renderer.splats
  preference.trainingEnabled = false
  #expect(renderer.status == .pointPreview)
  renderer.activateGaussian()  // A display tick must not undo the user's choice.
  #expect(renderer.status == .pointPreview)
  preference.trainingEnabled = true
  #expect(renderer.status == .training)
  #expect(renderer.sessionGeneration == session)
  #expect(renderer.splats.count == splats.count)
  #expect(renderer.latestFrame != nil)
  renderer.stop()
  preference.trainingEnabled = false
  #expect(renderer.status == .stopped)
  renderer.start()
  #expect(renderer.status == .pointPreview)
}

@Test @MainActor func fallbackPreservesPreferenceAndReasonAcrossToggles() {
  for reason in ["Metal device unsupported", "GPU budget exceeded"] {
    let store = MemoryPreferenceStore()
    let preference = LiveSplatPreference(store: store)
    let renderer = DefaultLiveSplatRenderer(preference: preference)
    renderer.start()
    if reason == "GPU budget exceeded" {
      renderer.acceptGPU([], ids: [], seconds: 0.009, session: renderer.sessionGeneration)
    } else {
      renderer.fallback(reason)
    }
    #expect(renderer.status == .fallback(reason))
    #expect(preference.trainingEnabled)
    #expect(store.values.isEmpty)
    preference.trainingEnabled = false
    #expect(renderer.status == .pointPreview)
    preference.trainingEnabled = true
    renderer.activateGaussian()
    #expect(renderer.status == .fallback(reason))
    #expect(store.values[LiveSplatPreference.key] == true)
  }
}

@Test @MainActor func fakeHonoursPreferenceAtStartAndWhileRunning() {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  let fake = FakeLiveSplat(preference: preference)
  fake.start()
  #expect(fake.status == .training)
  preference.trainingEnabled = false
  #expect(fake.status == .pointPreview)
  fake.stop()
  preference.trainingEnabled = true
  #expect(fake.status == .stopped)
  fake.start()
  #expect(fake.status == .training)
}

@Test @MainActor func preferencePersistsThroughUserDefaultsStore() throws {
  let domain = "LiveSplatTests.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: domain))
  defer { defaults.removePersistentDomain(forName: domain) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(preference.trainingEnabled)
  preference.trainingEnabled = false
  let restored = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(!restored.trainingEnabled)
  restored.trainingEnabled = true
  #expect(preference.trainingEnabled)
}

@Test @MainActor func pointPreviewConfigurationRemainsPreviewWhenPreferenceIsOn() {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  var configuration = LiveSplatConfiguration()
  configuration.mode = .pointPreview
  let renderer = DefaultLiveSplatRenderer(configuration: configuration, preference: preference)
  renderer.start()
  #expect(renderer.status == .pointPreview)
  renderer.activateGaussian()
  #expect(renderer.status == .pointPreview)
  #expect(preference.trainingEnabled)
}

@Test @MainActor func inFlightBudgetOverrunStillLatchesWhileUserChoosesPoints() {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  preference.trainingEnabled = false
  renderer.acceptGPU([], ids: [], seconds: 0.009, session: renderer.sessionGeneration)
  #expect(renderer.status == .pointPreview)
  #expect(!preference.trainingEnabled)
  preference.trainingEnabled = true
  #expect(renderer.status == .fallback("GPU budget exceeded"))
}
