import XCTest

/// These drive the real screens with DEBUG-only recorded capture/API providers.
/// The production-provider flow uses quiet rendering with real provider selection.
@MainActor final class HexUITests: XCTestCase {
  private var app: XCUIApplication!

  override func setUp() async throws {
    continueAfterFailure = false
    app = XCUIApplication()
  }

  override func tearDown() async throws {
    if app.launchArguments.contains("-uiTestFixture") {
      app.terminate()
      launch("yard-small")
      visible("home.settings")
      screenshot("Cleanup home")
      tap("home.settings", shot: "Credential cleanup")
      if app.buttons["settings.signOut"].exists {
        tap("settings.signOut", shot: "Signed out")
      }
    }
    app.terminate()
  }

  private func launch(_ fixture: String, emptyGlobe: Bool = false, resetCaptures: Bool = true) {
    app.launchArguments = ["-uiTestFixture", fixture]
    if emptyGlobe { app.launchArguments.append("-uiTestGlobeEmpty") }
    if resetCaptures { app.launchArguments.append("-uiTestResetCaptures") }
    app.launch()
  }

  private func launchFixture(signIn: Bool = true, emptyGlobe: Bool = false) {
    launch("yard-small", emptyGlobe: emptyGlobe)
    visible("home.record")
    visible("fixture.still")
    screenshot("Home")
    if signIn {
      tap("home.settings", shot: "Fixture sign-in")
      if app.buttons["settings.signOut"].exists {
        tap("settings.signOut", shot: "Previous credential removed")
      }
      tap("settings.token", shot: "Fixture token field")
      app.secureTextFields["settings.token"].typeText("ui-test-token\n")
      screenshot("Credential entered")
      tap("settings.signIn", shot: "Fixture signed in")
      tap("settings.back", shot: "Back to camera")
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
    let ready = NSPredicate(format: "exists == true AND hittable == true AND enabled == true")
    let readiness = XCTNSPredicateExpectation(predicate: ready, object: element)
    let outcome = XCTWaiter.wait(for: [readiness], timeout: 20)
    guard outcome == .completed else {
      XCTFail("Not ready to tap: \(id)", file: file, line: line)
      return
    }
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
    visible("splat.staticPreview")
    visible("scan.tip")
    screenshot("Scanning")
    tap("scan.stop", shot: "Saving")
    visible("upload.done")
    visible("upload.complete")
    screenshot("Upload")
    tap("upload.done", shot: "Done")
    tap("home.captures", shot: "Captures")
    let row = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH %@", "capture.row.")).firstMatch
    XCTAssertTrue(row.waitForExistence(timeout: 20))
    screenshot("Capture list")
    row.tap()
    visible("capture.detail")
    screenshot("Capture detail")
    visible("capture.share")
    visible("capture.delete")
    tap("capture.delete", shot: "Delete confirmation")
    tap("capture.confirmDelete", shot: "Local scan deleted")
    visible("captures.back")
    screenshot("Local fixture removed")
  }

  func testPhoneTrainingRefinesThroughTheFixtureAPIAndPersists() {
    launchFixture()
    tap("home.record", shot: "Training scan started")
    tap("scan.stop", shot: "Training scan saved")
    visible("upload.complete")
    tap("train.start", shot: "Training presets")
    tap("train.quick", shot: "Quick selected")
    tap("train.begin", shot: "Training started")
    visible("result.share")
    visible("result.orbit")
    XCTAssertTrue(app.staticTexts["Simulated"].exists)
    XCTAssertFalse(app.descendants(matching: .any)["inplace.open"].exists)
    let status = app.descendants(matching: .any)["refine.progress"].firstMatch
    visible("refine.progress")
    XCTAssertTrue(status.label.contains("Stored in cloud"))
    screenshot("Phone result")
    tap("refine.start", shot: "Refine cost confirmation")
    tap("refine.cancel", shot: "Refinement declined")
    XCTAssertTrue(status.label.contains("Stored in cloud"))
    tap("refine.start", shot: "Refine confirmed cost")
    tap("refine.confirm", shot: "Refinement requested")
    let ready = NSPredicate(format: "label CONTAINS %@", "Ready")
    expectation(for: ready, evaluatedWith: status)
    waitForExpectations(timeout: 20)
    XCTAssertTrue(app.buttons["result.cloud"].isSelected)
    screenshot("Cloud ready")
    tap("result.phone", shot: "Phone fallback")
    tap("result.close", shot: "Result closed")
    tap("upload.done", shot: "Capture finished")
    tap("home.captures", shot: "Trained capture list")
    visible("captures.badge.phone")
    visible("captures.badge.cloud")
    screenshot("Both quality badges")
    app.terminate()
    launch("yard-small", resetCaptures: false)
    tap("home.captures", shot: "Saved training after relaunch")
    visible("captures.badge.phone")
    let row = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH %@", "capture.row.")).firstMatch
    XCTAssertTrue(row.waitForExistence(timeout: 20))
    row.tap()
    tap("train.start", shot: "Saved phone result opened")
    visible("result.share")
    visible("result.orbit")
    XCTAssertFalse(app.descendants(matching: .any)["inplace.open"].exists)
    screenshot("Saved simulated result")
  }

  func testGlobeSelectsAnotherScan() {
    launchFixture()
    tap("home.globe", shot: "Globe")
    visible("globe.scanPicker")
    screenshot("Globe loaded")
    XCTAssertEqual(app.navigationBars.count, 0)
    XCTAssertEqual(app.buttons.matching(identifier: "globe.back").count, 1)
    tap("globe.scanPicker", shot: "Scan picker")
    tap("globe.scan.00000000-0000-0000-0000-000000000004", shot: "Selected second scan")
    XCTAssertTrue(app.buttons["globe.scanPicker"].label.contains("Simulated property"))
    XCTAssertFalse(app.buttons["globe.scanPicker"].label.contains("2026-01-02"))
    screenshot("Second scan on globe")
  }

  func testGlobeWithoutScansKeepsTheMapAndOneBackControl() {
    launchFixture(emptyGlobe: true)
    tap("home.globe", shot: "Empty globe")
    visible("globe.empty")
    visible("globe.scanPicker")
    XCTAssertEqual(app.navigationBars.count, 0)
    XCTAssertEqual(app.buttons.matching(identifier: "globe.back").count, 1)
    tap("globe.back", shot: "Home after empty globe")
    visible("home.globe")
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
    screenshot("Credential entered")
    tap("settings.signIn", shot: "Signed in")
    XCTAssertEqual(app.buttons["settings.signIn"].label, "Saved")
    tap("settings.signOut", shot: "Credential removed")
  }

  func testProductionProviderSelectionWithQuietRendering() {
    launch("production-providers")
    visible("home.settings")
    visible("fixture.still")
    screenshot("Production providers with quiet rendering")
    // The still image is a fixture even while all selected providers are real.
    XCTAssertTrue(app.staticTexts["Simulated"].exists)
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
    screenshot("Real providers")
  }
}
