import Foundation
import Testing

@testable import Globe

actor FixtureTransport: GlobeTransport {
  var responses: [Data]
  var urls: [URL] = []
  init(_ responses: [Data]) { self.responses = responses }
  func get(_ url: URL) async throws -> Data {
    urls.append(url)
    guard !responses.isEmpty else { throw GlobeError.httpStatus(404) }
    return responses.removeFirst()
  }
}

@Suite(.serialized)
struct GlobeTests {
  @Test func fakeAndRouteRoundTrip() async throws {
    let source = FakeGlobeDataSource()
    let sites = try await source.sites()
    #expect(sites.count == 1 && sites[0].name.contains("Simulated"))
    let snapshot = try await source.snapshot(siteID: sites[0].id)
    #expect(snapshot.pins[0].coordinate?.lat == 47.6405)
    #expect(snapshot.property.epochs[0].captureId == FakeGlobeDataSource.scanID)
    let route = GlobeRoute(siteID: sites[0].id, scanID: FakeGlobeDataSource.scanID)
    #expect(try JSONDecoder().decode(GlobeRoute.self, from: JSONEncoder().encode(route)) == route)
  }

  @Test func liveAdapterPaginationAndExistingSiteShapes() async throws {
    let fixture = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let encoder = JSONEncoder()
    struct Page: Encodable {
      let items: [GlobePin]
      let nextOffset: Int?
    }
    let transport = FixtureTransport([
      try encoder.encode(fixture.property),
      try encoder.encode(Page(items: fixture.pins, nextOffset: 500)),
      Data("{\"items\":[],\"nextOffset\":null}".utf8),
    ])
    let source = APIGlobeDataSource(
      baseURL: URL(string: "https://example.invalid/api/v1")!, transport: transport)
    let snapshot = try await source.snapshot(siteID: fixture.property.id)
    #expect(snapshot.pins.count == 1)
    let urls = await transport.urls
    #expect(urls[0].path.hasPrefix("/api/v1/sites/"))
    #expect(urls[2].query == "limit=500&offset=500")
  }

  @Test func refusesWrongSiteAndInvalidPagination() async throws {
    let fixture = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let data = try JSONEncoder().encode(fixture.property)
    let transport = FixtureTransport([data, Data("{\"items\":[],\"nextOffset\":0}".utf8)])
    let source = APIGlobeDataSource(
      baseURL: URL(string: "https://example.invalid/api/v1")!, transport: transport)
    await #expect(throws: GlobeError.invalidResponse) {
      try await source.snapshot(siteID: fixture.property.id)
    }
    let wrong = APIGlobeDataSource(
      baseURL: URL(string: "https://example.invalid/api/v1")!, transport: FixtureTransport([data]))
    await #expect(throws: GlobeError.invalidResponse) { try await wrong.snapshot(siteID: UUID()) }
  }

  @Test func polygonsPreserveHolesAndRejectBadCoordinates() throws {
    let json = """
      {"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]],[[0.1,0.1],[0.2,0.1],[0.2,0.2],[0.1,0.1]]]}
      """
    let polygon = try JSONDecoder().decode(GlobeFootprint.self, from: Data(json.utf8))
    #expect(polygon.polygons.count == 1 && polygon.polygons[0].count == 2)
    #expect(
      try JSONDecoder().decode(GlobeFootprint.self, from: JSONEncoder().encode(polygon)).polygons
        == polygon.polygons)
    #expect(throws: (any Error).self) {
      try JSONDecoder().decode(
        GlobeFootprint.self,
        from: Data(json.replacingOccurrences(of: "[1,0]", with: "[181,0]").utf8))
    }
  }

  @Test func openScanRefocusesTheEmbeddedGlobeWithoutReloadingItsSnapshot() async throws {
    let snapshot = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let wire = snapshot.bridgeSnapshot(fixture: true)
    let id = wire.scans[0].id
    var bridge = GlobeBridgeState()
    _ = bridge.update(snapshot: wire, scanID: id)
    #expect(bridge.openScan(id).isEmpty)
    _ = bridge.becomeReady()
    let focus: [BridgeMessage] = [.selectScan(scanId: id), .flyToScan(scanId: id)]
    #expect(bridge.openScan(id) == focus)
    #expect(bridge.openScan(id) == focus)
    #expect(bridge.openScan(UUID().uuidString).isEmpty)
  }

  @Test func siteListAndForeignPinRefusal() async throws {
    let fake = FakeGlobeDataSource()
    let snapshot = try await fake.snapshot(siteID: FakeGlobeDataSource.siteID)
    let encoder = JSONEncoder()
    var foreign = snapshot.pins[0]
    foreign = GlobePin(
      id: foreign.id, siteId: UUID(), scanId: foreign.scanId, entity: foreign.entity,
      observedAt: foreign.observedAt, uncertaintyM: foreign.uncertaintyM)
    struct Page: Encodable {
      let items: [GlobePin]
      let nextOffset: Int?
    }
    let transport = FixtureTransport([
      try encoder.encode(await fake.sites()), try encoder.encode(snapshot.property),
      try encoder.encode(Page(items: [foreign], nextOffset: nil)),
    ])
    let source = APIGlobeDataSource(
      baseURL: URL(string: "https://example.invalid/api/v1")!, transport: transport)
    #expect(try await source.sites().count == 1)
    await #expect(throws: GlobeError.invalidResponse) {
      try await source.snapshot(siteID: snapshot.property.id)
    }
  }

  @Test func unlocatedOrInvalidAnchorsAreNotPins() async throws {
    let original = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
      .pins[0]
    var entity = original.entity
    entity.anchor.geo?.lat = 91
    let invalid = GlobePin(
      id: original.id, siteId: original.siteId, scanId: original.scanId, entity: entity,
      observedAt: original.observedAt, uncertaintyM: nil)
    #expect(invalid.coordinate == nil)
    entity.anchor.geo = nil
    let unknown = GlobePin(
      id: original.id, siteId: original.siteId, scanId: original.scanId, entity: entity,
      observedAt: original.observedAt, uncertaintyM: nil)
    #expect(unknown.coordinate == nil)
  }

}
