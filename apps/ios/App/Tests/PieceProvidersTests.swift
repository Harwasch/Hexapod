#if !os(iOS)
  import Foundation
  import Voice
  import Router
  import Perception
  import Testing
  @testable import AppSupport

  @Test @MainActor func independentChoicesPersistAndSiblingProvidersRegisterInOneLine() throws {
    let suite = "hex.test.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let registry = PieceRegistry(defaults: defaults)
    for (key, value) in [
      ("voice", "fake"), ("planner", "fake"), ("perception", "fake"), ("overlay", "fake"),
    ] {
      registry.choose(value, for: key)
    }
    registry.choose("cloud", for: "deep")
    #expect(throws: ProviderError.self) { try registry.makeCamera() }
    registry.deep.register("cloud") { FakeDeepAnswerForRegistry() }
    #expect(try registry.makeCamera().fakeOverlay)
    #expect(registry.choice("capture") == "live")
    // Release uses real/offline defaults even if TestFlight left fake selections stored.
    registry.voice.register("real") { Voice.FakeVoice() }
    registry.planner.register("real") { Router.RulePlanner() }
    registry.perception.register("real") { Perception.FakePerception() }
    let release = try registry.makeCamera(useTestChoices: false)
    #expect(!release.simulated && !release.fakeOverlay && release.capture.usesLiveSession)
    #expect(registry.choice("voice") == "fake")
    let restored = PieceRegistry(defaults: defaults)
    #expect(restored.choice("voice") == "fake")
    #expect(restored.choice("deep") == "cloud")
    #expect(!restored.deep.contains("cloud"))
    registry.choose("yard-small", for: "capture")
    #expect(throws: ProviderError.self) { try registry.makeCamera() }
    #expect(!PieceRegistry.menuAvailable(debug: false, receipt: "receipt", enabled: true))
    #expect(!PieceRegistry.menuAvailable(debug: false, receipt: "sandboxReceipt", enabled: false))
    #expect(PieceRegistry.menuAvailable(debug: false, receipt: "sandboxReceipt", enabled: true))
    #expect(PieceRegistry.menuAvailable(debug: true, receipt: nil, enabled: false))
  }
  import EntitySchema
  private struct FakeDeepAnswerForRegistry: DeepAnswering {
    func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
      AsyncThrowingStream { $0.finish() }
    }
  }
#endif
