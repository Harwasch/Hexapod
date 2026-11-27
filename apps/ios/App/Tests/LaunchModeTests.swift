import Foundation
import Testing

@testable import AppSupport

@Test func launchModeIsOffByDefaultAndRejectsMissingFixtureNames() {
  #expect(!LaunchMode.current.isUITest)
  #expect(LaunchMode.current.captureAPI == nil)
  for arguments in [
    [], ["Hex"], ["Hex", "-uiTestFixture"],
    ["Hex", "-uiTestFixture", ""], ["Hex", "-uiTestFixture", "-other"],
  ] {
    let mode = LaunchMode(arguments: arguments)
    #expect(!mode.isUITest)
    #expect(mode.fixtureName == nil)
    #expect(mode.captureAPI == nil)
    #expect(!mode.resetCaptures && !mode.emptyGlobe)
  }
}

@Test func launchModeParsesFixtureOnceAndFreezesItsClock() {
  let mode = LaunchMode(arguments: ["Hex", "-uiTestFixture", "yard-small"])
  #expect(mode.isUITest)
  #expect(mode.usesFixtureProviders)
  #expect(mode.fixtureName == "yard-small")
  #expect(mode.date(at: Date(timeIntervalSince1970: 10)) == mode.date(at: Date()))
  let ordinary = LaunchMode(arguments: [])
  let now = Date()
  #expect(ordinary.date(at: now) == now)
  #expect(
    mode.captureRoot
      == ordinary.captureRoot.deletingLastPathComponent()
      .appendingPathComponent("UITest/Captures"))
}

@Test func fixtureControlFlagsRequireExplicitUITestLaunch() {
  let flags = ["-uiTestResetCaptures", "-uiTestGlobeEmpty"]
  let ordinary = LaunchMode(arguments: flags)
  #expect(!ordinary.resetCaptures && !ordinary.emptyGlobe)
  let fixture = LaunchMode(arguments: ["-uiTestFixture", "yard-small"] + flags)
  #expect(fixture.resetCaptures && fixture.emptyGlobe)
}

private enum TrainerFactoryProbe: Error { case realSelected }
@Test @MainActor func ordinaryTrainingFactoryCallsTheRealConstructor() async {
  await #expect(throws: TrainerFactoryProbe.realSelected) {
    try await PhoneTrainingFactory.make(outputDirectory: URL(fileURLWithPath: "/unused")) {
      throw TrainerFactoryProbe.realSelected
    }
  }
}

@Test func productionProviderProbeStillUsesQuietRendering() {
  let mode = LaunchMode(arguments: ["Hex", "-uiTestFixture", "production-providers"])
  #expect(mode.isUITest)
  #expect(!mode.usesFixtureProviders)
}
