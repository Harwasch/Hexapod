import XCTest

/// These drive the real screens with DEBUG-only recorded capture/API providers.
/// An ordinary launch below exercises the production provider selection instead.
@MainActor final class HexUITests: XCTestCase {
  private var app: XCUIApplication!

  override func setUp() async throws {
    continueAfterFailure = false
    app = XCUIApplication()
  }

  override func tearDown() async throws {
    if app.launchArguments.contains("-uiTestFixture") {
      app.terminate()
      app.launch()
      visible("home.settings")
      tap("home.settings", shot: "Credential cleanup")
      if app.buttons["settings.signOut"].exists {
        tap("settings.signOut", shot: "Signed out")
      }
    }
    app.terminate()
  }

  private func launchFixture(signIn: Bool = true) {
    app.launchArguments = ["-uiTestFixture", "yard-small"]
    app.launch()
    visible("home.record")
    screenshot("Home")
    if signIn {
      tap("home.settings", shot: "Fixture sign-in")
      if app.buttons["settings.signOut"].exists { app.buttons["settings.signOut"].tap() }
      tap("settings.token", shot: "Fixture token field")
      app.secureTextFields["settings.token"].typeText("ui-test-token\n")
      tap("settings.signIn", shot: "Fixture signed in")
      app.navigationBars.buttons.firstMatch.tap()
      visible("home.record")
      screenshot("Fixture home")
    }
  }
  private func visible(_ id: String, timeout: TimeInterval = 20,
    file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertTrue(app.descendants(matching: .any)[id].firstMatch.waitForExistence(timeout: timeout),
      "Missing \(id)", file: file, line: line)
  }
  private func tap(_ id: String, shot: String,
    file: StaticString = #filePath, line: UInt = #line) {
    visible(id, file: file, line: line)
    let element = app.descendants(matching: .any)[id].firstMatch
    XCTAssertTrue(element.isHittable, "Not hittable: \(id)", file: file, line: line)
    element.tap()
    screenshot(shot)
  }
  private func screenshot(_ name: String) {
    let attachment = XCTAttachment(screenshot: app.screenshot())
    attachment.name = name
    attachment.lifetime = .keepAlways
    add(attachment)
  }

  func testRecordUploadAndOpenCapture() {
    launchFixture()
    tap("home.record", shot: "Recording")
    visible("scan.stop")
    screenshot("Scanning")
    tap("scan.stop", shot: "Saving")
    visible("upload.done")
    let progress = app.descendants(matching: .any).matching(
      NSPredicate(format: "label == %@", "Uploaded 100 percent")).firstMatch
    XCTAssertTrue(progress.waitForExistence(timeout: 20), "Fixture API must finish the resumed upload")
    screenshot("Upload")
    tap("upload.done", shot: "Done")
    tap("home.captures", shot: "Captures")
    let row = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "capture.row.")).firstMatch
    XCTAssertTrue(row.waitForExistence(timeout: 20))
    screenshot("Capture list")
    row.tap()
    visible("capture.detail")
    screenshot("Capture detail")
    visible("capture.share")
    visible("capture.delete")
    tap("capture.delete", shot: "Delete confirmation")
    app.buttons["Delete local scan"].tap()
    visible("captures.back")
    screenshot("Local fixture removed")
  }

  func testGlobeSelectsAnotherScan() {
    launchFixture()
    tap("home.globe", shot: "Globe")
    visible("globe.scanPicker")
    screenshot("Globe loaded")
    tap("globe.scanPicker", shot: "Scan picker")
    tap("globe.scan.00000000-0000-0000-0000-000000000004", shot: "Selected second scan")
    XCTAssertTrue(app.buttons["globe.scanPicker"].label.contains("2026-01-02"))
    screenshot("Second scan on globe")
  }

  func testAskAnswerAndEntityCard() {
    launchFixture()
    visible("ask.microphone")
    let mic = app.descendants(matching: .any)["ask.microphone"].firstMatch
    mic.press(forDuration: 1)
    screenshot("Question asked")
    visible("answer.pin.1")
    screenshot("Answer and pins")
    tap("answer.pin.1", shot: "Entity selected")
    visible("entity.card")
    screenshot("Entity card")
  }

  func testSettingsAndKeychainSignIn() {
    launchFixture(signIn: false)
    tap("home.settings", shot: "Settings")
    visible("settings.server")
    tap("settings.token", shot: "Token field")
    app.secureTextFields["settings.token"].typeText("ui-test-token\n")
    tap("settings.signIn", shot: "Signed in")
    XCTAssertEqual(app.buttons["settings.signIn"].label, "Saved")
    tap("settings.signOut", shot: "Credential removed")
  }

  func testOrdinaryLaunchUsesEveryRealProvider() {
    app.launchArguments = []
    app.launch()
    visible("home.settings")
    screenshot("Ordinary launch")
    // Settings exposes the resolved production selection as an accessibility value.
    tap("home.settings", shot: "Production settings")
    visible("settings.providers")
    let providers = app.staticTexts["settings.providers"].value as? String ?? ""
    for selection in ["capture:live", "voice:real", "planner:real", "perception:real",
      "liveSplat:real", "globe:real", "overlay:real"] {
      XCTAssertTrue(providers.split(separator: ",").contains(Substring(selection)), providers)
    }
    XCTAssertTrue(providers.contains("deep:cloud") || providers.contains("deep:offline"), providers)
    XCTAssertFalse(providers.contains("fake"))
    XCTAssertFalse(app.staticTexts["Simulated"].exists)
    screenshot("Real providers")
  }
}
