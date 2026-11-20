import Foundation

/// A single property opens directly; multiple properties need a deliberate choice.
public func globeInitialRoute(sites: [GlobeSite]) -> GlobeRoute? {
  sites.count == 1 ? GlobeRoute(siteID: sites[0].id) : nil
}

/// Public catalog read, including the site assigned by reconstruction after upload.
/// An upload's queued job alone is never evidence that its scan is ready.
public struct CatalogCaptureState: Decodable, Sendable {
  public let status: String
  public let siteId: UUID?
  public var label: String {
    switch status {
    case "awaiting-files", "not-started": "Queued"
    case "in-progress": "Refining"
    case "complete": "Ready"
    case "error": "Failed"
    default: "Unavailable"
    }
  }
  public static func fetch(
    captureID: UUID, baseURL: URL,
    transport: any GlobeTransport = URLSessionGlobeTransport()
  ) async throws -> Self {
    try JSONDecoder().decode(
      Self.self, from: await transport.get(
        baseURL.appendingPathComponent("captures/\(captureID.uuidString)")))
  }
}
