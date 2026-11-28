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
    // Debug fixture credentials are process-local. Termination clears them;
    // launching a fresh process to sign out can only inspect an empty store.
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
      waitFor(NSPredicate(format: "label == %@", "Saved"), on: app.buttons["settings.signIn"])
      XCTAssertEqual(app.buttons["settings.signIn"].label, "Saved")
      tap("settings.back", shot: "Back to camera")
      visible("home.record")
      screenshot("Fixture home")
    }
  }
  private func visible(_ id: String, timeout: TimeInterval = 30,
    file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertTrue(app.descendants(matching: .any)[id].firstMatch.waitForExistence(timeout: timeout),
      "Missing \(id)", file: file, line: line)
  }
  private func tap(_ id: String, shot: String,
    file: StaticString = #filePath, line: UInt = #line) {
    visible(id, file: file, line: line)
    let element = id == "settings.token"
      ? app.secureTextFields[id].firstMatch : app.buttons[id].firstMatch
    waitUntilReady(element, file: file, line: line)
    element.tap()
    screenshot(shot)
  }
  private func waitFor(_ predicate: NSPredicate, on element: XCUIElement,
    timeout: TimeInterval = 30, file: StaticString = #filePath, line: UInt = #line) {
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
    XCTAssertEqual(XCTWaiter.wait(for: [expectation], timeout: timeout), .completed,
      "State not reached: \(predicate) for \(element)", file: file, line: line)
  }
  private func waitUntilReady(_ element: XCUIElement,
    file: StaticString = #filePath, line: UInt = #line) {
    let ready = NSPredicate(format: "exists == true AND hittable == true AND enabled == true")
    waitFor(ready, on: element, file: file, line: line)
  }
  private func scrollTo(_ id: String, file: StaticString = #filePath, line: UInt = #line) {
    visible(id, file: file, line: line)
    let element = app.buttons[id].firstMatch
    let scroll = app.scrollViews.firstMatch
    XCTAssertTrue(scroll.waitForExistence(timeout: 30), file: file, line: line)
    for _ in 0..<4 {
      if element.isHittable { break }
      scroll.swipeUp()
    }
    waitUntilReady(element, file: file, line: line)
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
    XCTAssertTrue(row.waitForExistence(timeout: 30))
    screenshot("Capture list")
    waitUntilReady(row)
    row.tap()
    visible("capture.detail")
    screenshot("Capture detail")
    visible("capture.share")
    visible("capture.delete")
    scrollTo("capture.delete")
    XCTAssertTrue(app.buttons["capture.share"].isHittable)
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
    visible("result.share", timeout: 60)
    visible("result.orbit")
    XCTAssertTrue(app.staticTexts["Simulated"].exists)
    XCTAssertFalse(app.descendants(matching: .any)["inplace.open"].exists)
    let status = app.descendants(matching: .any)["refine.progress"].firstMatch
    visible("refine.progress")
    waitFor(NSPredicate(format: "label CONTAINS %@", "Stored in cloud"), on: status)
    XCTAssertTrue(status.label.contains("Stored in cloud"))
    screenshot("Phone result")
    tap("refine.start", shot: "Refine cost confirmation")
    tap("refine.cancel", shot: "Refinement declined")
    waitUntilReady(app.buttons["refine.start"])
    XCTAssertTrue(status.label.contains("Stored in cloud"))
    tap("refine.start", shot: "Refine confirmed cost")
    tap("refine.confirm", shot: "Refinement requested")
    let ready = NSPredicate(format: "label CONTAINS %@", "Ready")
    waitFor(ready, on: status, timeout: 60)
    waitFor(NSPredicate(format: "selected == true"), on: app.buttons["result.cloud"])
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
    XCTAssertTrue(row.waitForExistence(timeout: 30))
    waitUntilReady(row)
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
    let secondScan = app.buttons["globe.scan.00000000-0000-0000-0000-000000000004"]
    XCTAssertTrue(secondScan.waitForExistence(timeout: 30))
    XCTAssertFalse(secondScan.isSelected)
    tap("globe.scan.00000000-0000-0000-0000-000000000004", shot: "Selected second scan")
    waitUntilReady(app.buttons["globe.scanPicker"])
    waitFor(NSPredicate(format: "value == %@", "00000000-0000-0000-0000-000000000004"),
      on: app.buttons["globe.scanPicker"])
    XCTAssertEqual(app.buttons["globe.scanPicker"].value as? String,
      "00000000-0000-0000-0000-000000000004")
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
    waitUntilReady(mic)
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
    waitFor(NSPredicate(format: "label == %@", "Saved"), on: app.buttons["settings.signIn"])
    XCTAssertEqual(app.buttons["settings.signIn"].label, "Saved")
    tap("settings.signOut", shot: "Credential removed")
    waitFor(NSPredicate(format: "label == %@ AND enabled == false", "Sign in"),
      on: app.buttons["settings.signIn"])
    waitFor(NSPredicate(format: "exists == false"), on: app.buttons["settings.signOut"])
    XCTAssertEqual(app.buttons["settings.signIn"].label, "Sign in")
    XCTAssertFalse(app.buttons["settings.signIn"].isEnabled)
    XCTAssertFalse(app.buttons["settings.signOut"].exists)
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
