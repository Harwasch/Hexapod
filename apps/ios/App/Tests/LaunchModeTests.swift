import Foundation
import Testing

@testable import AppSupport

@Test func launchModeIsOffByDefaultAndRejectsMissingFixtureNames() {
  #expect(!LaunchMode.current.isUITest)
  for arguments in [
    [], ["Hex"], ["Hex", "-uiTestFixture"],
    ["Hex", "-uiTestFixture", ""], ["Hex", "-uiTestFixture", "-other"],
  ] {
    let mode = LaunchMode(arguments: arguments)
    #expect(!mode.isUITest)
    #expect(mode.fixtureName == nil)
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
}

@Test func productionProviderProbeStillUsesQuietRendering() {
  let mode = LaunchMode(arguments: ["Hex", "-uiTestFixture", "production-providers"])
  #expect(mode.isUITest)
  #expect(!mode.usesFixtureProviders)
}
