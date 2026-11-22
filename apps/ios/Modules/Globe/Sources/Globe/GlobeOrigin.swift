import Foundation

/// Navigation permits one HTTPS origin. Only explicit fixture mode trusts one bundled page.
public struct GlobeOrigin: Sendable {
  public static let deployed = URL(string: "https://twin-web-f57.pages.dev")!
  public let page: URL
  private let fixturePage: URL?
  public init(webBaseURL: URL = Self.deployed, fixturePage: URL? = nil) {
    self.fixturePage = fixturePage
    page = fixturePage ?? webBaseURL.appendingPathComponent("globe-embed.html")
  }
  public func allowsNavigation(_ url: URL?) -> Bool {
    url?.absoluteString == "about:blank" || allowsMessage(url)
  }
  public func allowsMessage(_ url: URL?) -> Bool {
    guard let url, url.user == nil, url.password == nil else { return false }
    if let fixturePage { return url.isFileURL && url == fixturePage }
    return url.scheme == "https" && page.scheme == "https" && url.host != nil
      && url.host?.lowercased() == page.host?.lowercased()
      && (url.port ?? 443) == (page.port ?? 443)
  }
  public func allowsSecurityOrigin(scheme: String, host: String, port: Int) -> Bool {
    if fixturePage != nil { return scheme == "file" }
    return scheme == "https" && page.scheme == "https"
      && host.lowercased() == page.host?.lowercased()
      && (port == 0 ? 443 : port) == (page.port ?? 443)
  }
}
