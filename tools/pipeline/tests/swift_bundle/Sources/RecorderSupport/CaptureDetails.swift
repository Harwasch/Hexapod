import Foundation

// Same portable value type as App/Support/LocalCaptures.swift; avoid importing Upload.
struct CaptureDetails: Codable, Sendable {
  var siteName: String?
  var siteID: UUID?
  var duration: TimeInterval
}
