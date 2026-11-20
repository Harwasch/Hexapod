import Foundation

/// One DEBUG-only launch switch, shared by provider selection and quiet rendering.
struct LaunchMode: Equatable, Sendable {
  static let current: LaunchMode = {
    #if DEBUG
      LaunchMode(arguments: ProcessInfo.processInfo.arguments)
    #else
      LaunchMode(arguments: [])
    #endif
  }()

  let fixtureName: String?
  var isUITest: Bool { fixtureName != nil }
  // The provider-defaults flow needs quiet rendering while resolving real providers.
  var usesFixtureProviders: Bool { fixtureName == "yard-small" }

  init(arguments: [String]) {
    if let index = arguments.firstIndex(of: "-uiTestFixture"),
      arguments.indices.contains(index + 1), !arguments[index + 1].isEmpty,
      !arguments[index + 1].hasPrefix("-")
    {
      fixtureName = arguments[index + 1]
    } else {
      fixtureName = nil
    }
  }

  /// UI tests stop time; ordinary recording continues to use wall time.
  func date(at now: Date = Date()) -> Date {
    isUITest ? Date(timeIntervalSince1970: 1_767_225_600) : now
  }

  var previewDirectory: URL? {
    guard isUITest else { return nil }
    return Bundle.main.url(
      forResource: "yard-small", withExtension: nil, subdirectory: "UITestFixtures")?
      .appendingPathComponent("bundle")
  }
}
