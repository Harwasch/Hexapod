import Networking
import EntitySchema
import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// App A may push this route from Home or a capture's siteId/captureId.
public struct GlobeRoute: Hashable, Codable, Sendable {
  public let siteID: UUID
  public let scanID: UUID?
  public init(siteID: UUID, scanID: UUID? = nil) {
    self.siteID = siteID
    self.scanID = scanID
  }
}

public struct GlobeSite: Codable, Identifiable, Sendable {
  public let id: UUID
  public let name: String
  public init(id: UUID, name: String) {
    self.id = id
    self.name = name
  }
}

public struct GlobeFootprint: Codable, Sendable {
  public let type: String
  /// Always polygons -> rings (outer, holes) -> longitude/latitude positions.
  public let polygons: [[[[Double]]]]

  enum CodingKeys: String, CodingKey { case type, coordinates }
  public init(polygons: [[[[Double]]]]) {
    self.type = "MultiPolygon"
    self.polygons = polygons
  }
  public init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    type = try c.decode(String.self, forKey: .type)
    switch type {
    case "Polygon": polygons = [try c.decode([[[Double]]].self, forKey: .coordinates)]
    case "MultiPolygon": polygons = try c.decode([[[[Double]]]].self, forKey: .coordinates)
    default:
      throw DecodingError.dataCorruptedError(
        forKey: .type, in: c, debugDescription: "Expected polygon")
    }
    for polygon in polygons {
      guard !polygon.isEmpty else { throw GlobeError.invalidResponse }
      for ring in polygon {
        guard ring.count >= 4, ring.first == ring.last,
          ring.allSatisfy({
            $0.count >= 2 && $0[0].isFinite && $0[1].isFinite && abs($0[0]) <= 180
              && abs($0[1]) <= 90
          })
        else { throw GlobeError.invalidResponse }
      }
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(type, forKey: .type)
    if type == "Polygon" {
      try c.encode(polygons[0], forKey: .coordinates)
    } else {
      try c.encode(polygons, forKey: .coordinates)
    }
  }
}

public struct GlobeScan: Codable, Identifiable, Sendable {
  public let id: UUID
  public let captureId: UUID?
  public let observedAt: String
  public let currentAssetId: UUID?
  public let footprint: GlobeFootprint
  public init(
    id: UUID, captureId: UUID?, observedAt: String, footprint: GlobeFootprint,
    currentAssetId: UUID? = nil
  ) {
    self.id = id
    self.currentAssetId = currentAssetId
    self.captureId = captureId
    self.observedAt = observedAt
    self.footprint = footprint
  }
}

public struct GlobeAsset: Codable, Sendable {
  public let id: UUID
  public let epochId: UUID?
  public let representation: String
  public let source: Source
  public struct Source: Codable, Sendable {
    public let type: String
    public let url: String?
  }
}

public struct GlobeProperty: Codable, Identifiable, Sendable {
  public let id: UUID
  public let name: String
  public let boundary: GlobeFootprint
  public let assets: [GlobeAsset]?
  public let epochs: [GlobeScan]
  public init(
    id: UUID, name: String, boundary: GlobeFootprint, epochs: [GlobeScan],
    assets: [GlobeAsset]? = nil
  ) {
    self.id = id
    self.name = name
    self.boundary = boundary
    self.epochs = epochs
    self.assets = assets
  }
}

/// Metadata envelope from the property endpoint. Entity and Ask shapes are unchanged.
public struct GlobePin: Codable, Identifiable, Sendable {
  public let id: UUID
  public let siteId: UUID
  public let scanId: UUID
  public let entity: Entity
  public let observedAt: String
  public let uncertaintyM: Double?
  public init(
    id: UUID, siteId: UUID, scanId: UUID, entity: Entity, observedAt: String, uncertaintyM: Double?
  ) {
    self.id = id
    self.siteId = siteId
    self.scanId = scanId
    self.entity = entity
    self.observedAt = observedAt
    self.uncertaintyM = uncertaintyM
  }
  public var coordinate: GeoCoordinate? {
    guard let geo = entity.anchor.geo, geo.lat.isFinite, geo.lon.isFinite, geo.alt.isFinite,
      abs(geo.lat) <= 90, abs(geo.lon) <= 180
    else { return nil }
    return geo
  }
}

public struct GlobeSnapshot: Sendable {
  public let property: GlobeProperty
  public let pins: [GlobePin]
  public init(property: GlobeProperty, pins: [GlobePin]) {
    self.property = property
    self.pins = pins
  }
}

public protocol GlobeDataSource: Sendable {
  func sites() async throws -> [GlobeSite]
  func snapshot(siteID: UUID) async throws -> GlobeSnapshot
}

public enum GlobeError: Error, Equatable {
  case httpStatus(Int)
  case invalidResponse, paginationLimit
}

public protocol GlobeTransport: Sendable {
  func get(_ url: URL) async throws -> Data
}

public struct URLSessionGlobeTransport: GlobeTransport {
  public init() {}
  public func get(_ url: URL) async throws -> Data {
    do { return try await APIClient().data(for: URLRequest(url: url)) }
    catch APIError.httpStatus(let status) { throw GlobeError.httpStatus(status) }
  }
}

public struct APIGlobeDataSource: GlobeDataSource {
  /// API root including /api/v1. No token is needed for the existing public reads.
  public let baseURL: URL
  private let transport: any GlobeTransport
  public init(baseURL: URL, transport: any GlobeTransport = URLSessionGlobeTransport()) {
    self.baseURL = baseURL
    self.transport = transport
  }
  public func sites() async throws -> [GlobeSite] {
    try JSONDecoder().decode(
      [GlobeSite].self, from: await transport.get(baseURL.appendingPathComponent("sites")))
  }
  public func snapshot(siteID: UUID) async throws -> GlobeSnapshot {
    let site = try JSONDecoder().decode(
      GlobeProperty.self,
      from: await transport.get(baseURL.appendingPathComponent("sites/\(siteID.uuidString)")))
    guard site.id == siteID else { throw GlobeError.invalidResponse }
    var offset = 0
    var pins: [UUID: GlobePin] = [:]
    for _ in 0..<200 {
      try Task.checkCancellation()
      var url = URLComponents(
        url: baseURL.appendingPathComponent("properties/\(siteID.uuidString)/entities"),
        resolvingAgainstBaseURL: false)!
      url.queryItems = [
        URLQueryItem(name: "limit", value: "500"),
        URLQueryItem(name: "offset", value: String(offset)),
      ]
      let page = try JSONDecoder().decode(PinPage.self, from: await transport.get(url.url!))
      guard page.items.allSatisfy({ $0.siteId == siteID }) else { throw GlobeError.invalidResponse }
      for pin in page.items { pins[pin.id] = pin }
      guard let next = page.nextOffset else {
        return GlobeSnapshot(
          property: site, pins: pins.values.sorted { $0.id.uuidString < $1.id.uuidString })
      }
      guard next > offset else { throw GlobeError.invalidResponse }
      offset = next
    }
    throw GlobeError.paginationLimit
  }
  private struct PinPage: Decodable {
    let items: [GlobePin]
    let nextOffset: Int?
  }
}

public struct FakeGlobeDataSource: GlobeDataSource {
  public static let siteID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
  public static let scanID = UUID(uuidString: "00000000-0000-0000-0000-000000000002")!
  public init() {}
  public func sites() async throws -> [GlobeSite] {
    [GlobeSite(id: Self.siteID, name: "Simulated property")]
  }
  public func snapshot(siteID: UUID) async throws -> GlobeSnapshot {
    guard siteID == Self.siteID else { throw GlobeError.httpStatus(404) }
    let footprint = GlobeFootprint(polygons: [
      [
        [
          [-122.14, 47.64], [-122.139, 47.64], [-122.139, 47.641], [-122.14, 47.641],
          [-122.14, 47.64],
        ]
      ]
    ])
    let property = GlobeProperty(
      id: siteID, name: "Simulated property", boundary: footprint,
      epochs: [
        GlobeScan(
          id: Self.scanID, captureId: Self.scanID, observedAt: "2026-01-01T00:00:00Z",
          footprint: footprint),
        GlobeScan(
          id: UUID(uuidString: "00000000-0000-0000-0000-000000000004")!,
          captureId: UUID(uuidString: "00000000-0000-0000-0000-000000000004")!,
          observedAt: "2026-01-02T00:00:00Z", footprint: footprint)
      ])
    let entity = Entity(
      label: "Simulated oak",
      anchor: Anchor(
        world: Vector3(x: 0, y: 0, z: 0), geo: GeoCoordinate(lat: 47.6405, lon: -122.1395, alt: 0)),
      extent: Vector3(x: 3, y: 10, z: 3), masks: [], confidence: 0.8, attrs: [:],
      source: EntitySource(module: "GlobeFixture", model: "fixture"))
    return GlobeSnapshot(
      property: property,
      pins: [
        GlobePin(
          id: UUID(uuidString: "00000000-0000-0000-0000-000000000003")!, siteId: siteID,
          scanId: Self.scanID, entity: entity, observedAt: "2026-01-01T00:00:00Z", uncertaintyM: 3)
      ])
  }
}
