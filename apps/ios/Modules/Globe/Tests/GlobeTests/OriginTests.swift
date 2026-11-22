import Foundation
import Testing

@testable import Globe

@Test func hostedOriginIsExactAndBlankCannotSendBridgeEvents() {
  let policy = GlobeOrigin()
  #expect(policy.page.absoluteString == "https://twin-web-f57.pages.dev/globe-embed.html")
  for value in [
    "https://twin-web-f57.pages.dev/globe-embed.html",
    "https://twin-web-f57.pages.dev:443/view.html",
  ] {
    #expect(policy.allowsNavigation(URL(string: value)))
    #expect(policy.allowsMessage(URL(string: value)))
  }
  #expect(policy.allowsNavigation(URL(string: "about:blank")))
  #expect(!policy.allowsMessage(URL(string: "about:blank")))
  for value in [
    "http://twin-web-f57.pages.dev", "https://twin-web-f57.pages.dev:444",
    "https://twin-web-f57.pages.dev.evil.test", "https://evil.test", "file:///tmp/globe-embed.html",
    "https://user@twin-web-f57.pages.dev", "data:text/html,hello",
  ] {
    #expect(!policy.allowsNavigation(URL(string: value)))
  }
  #expect(!policy.allowsNavigation(nil))
  #expect(policy.allowsSecurityOrigin(scheme: "https", host: "twin-web-f57.pages.dev", port: 443))
  #expect(!policy.allowsSecurityOrigin(scheme: "https", host: "evil.test", port: 443))
  #expect(!policy.allowsSecurityOrigin(scheme: "file", host: "", port: 0))
}

@Test func fixtureTrustsOnlyItsBundledDocument() {
  let page = URL(fileURLWithPath: "/bundle/GlobeWeb/globe-embed.html")
  let policy = GlobeOrigin(fixturePage: page)
  #expect(policy.allowsMessage(page))
  #expect(!policy.allowsMessage(URL(fileURLWithPath: "/bundle/GlobeWeb/other.html")))
  #expect(!policy.allowsMessage(GlobeOrigin.deployed))
  #expect(policy.allowsSecurityOrigin(scheme: "file", host: "", port: 0))
}
