import Foundation
import Testing

@testable import LiveSplat

#if canImport(Combine)
  import Combine

  @Test @MainActor func preferencePublishesThroughObservableObjectBeforeChanging() {
    func publisher<Object: ObservableObject>(_ object: Object) -> Object.ObjectWillChangePublisher {
      object.objectWillChange
    }
    let preference = LiveSplatPreference(store: MemoryPreferenceStore())
    var previousValues: [Bool] = []
    let subscription = publisher(preference).sink {
      previousValues.append(preference.previewEnabled)
    }
    preference.previewEnabled = false
    preference.previewEnabled = false
    preference.previewEnabled = true
    #expect(previousValues == [true, false])
    withExtendedLifetime(subscription) {}
  }
#endif

@MainActor private final class MemoryPreferenceStore: LiveSplatPreferenceStore {
  var values: [String: Bool] = [:]
  func bool(forKey key: String) -> Bool? { values[key] }
  func set(_ value: Bool, forKey key: String) { values[key] = value }
}

@Test @MainActor func previewIsOnByDefaultAndPreferenceRoundTrips() {
  let store = MemoryPreferenceStore()
  let preference = LiveSplatPreference(store: store)
  #expect(preference.previewEnabled)
  #expect(LiveSplatConfiguration().mode == .scanPreview)
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  #expect(renderer.status == .scanPreview)
  preference.previewEnabled = false
  #expect(store.values["hex.liveSplat.training"] == false)
  #expect(!LiveSplatPreference(store: store).previewEnabled)
  preference.previewEnabled = true
  #expect(LiveSplatPreference(store: store).previewEnabled)
}

@Test @MainActor func preferenceSwitchesRunningRendererWithoutResettingScan() async throws {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  let renderer = DefaultLiveSplatRenderer(preference: preference)
  renderer.start()
  let source = try FakeFrameSource.plane()
  renderer.ingest(try #require(await source.next()))
  let session = renderer.sessionGeneration
  let splats = renderer.splats
  preference.previewEnabled = false
  #expect(renderer.status == .coverageOnly)
  #expect(renderer.status == .coverageOnly)
  preference.previewEnabled = true
  #expect(renderer.status == .scanPreview)
  #expect(renderer.sessionGeneration == session)
  #expect(renderer.splats.count == splats.count)
  #expect(renderer.latestFrame != nil)
  renderer.stop()
  preference.previewEnabled = false
  #expect(renderer.status == .stopped)
  renderer.start()
  #expect(renderer.status == .coverageOnly)
}

@Test @MainActor func fakeHonoursPreferenceAtStartAndWhileRunning() {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  let fake = FakeLiveSplat(preference: preference)
  fake.start()
  #expect(fake.status == .scanPreview)
  preference.previewEnabled = false
  #expect(fake.status == .coverageOnly)
  fake.stop()
  preference.previewEnabled = true
  #expect(fake.status == .stopped)
  fake.start()
  #expect(fake.status == .scanPreview)
}

@Test @MainActor func preferencePersistsThroughUserDefaultsStore() throws {
  let domain = "LiveSplatTests.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: domain))
  defer { defaults.removePersistentDomain(forName: domain) }
  let preference = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(preference.previewEnabled)
  preference.previewEnabled = false
  let restored = LiveSplatPreference(store: UserDefaultsLiveSplatStore(defaults: defaults))
  #expect(!restored.previewEnabled)
  restored.previewEnabled = true
  #expect(preference.previewEnabled)
}

@Test @MainActor func coverageOnlyConfigurationRemainsPreviewWhenPreferenceIsOn() {
  let preference = LiveSplatPreference(store: MemoryPreferenceStore())
  var configuration = LiveSplatConfiguration()
  configuration.mode = .coverageOnly
  let renderer = DefaultLiveSplatRenderer(configuration: configuration, preference: preference)
  renderer.start()
  #expect(renderer.status == .coverageOnly)
  #expect(renderer.status == .coverageOnly)
  #expect(preference.previewEnabled)
}
