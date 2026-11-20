import Foundation
import Testing

@testable import Globe

struct CatalogStateTests {
  @Test func onlyOneSiteSkipsPicker() {
    let site = GlobeSite(id: UUID(), name: "Field")
    #expect(globeInitialRoute(sites: []) == nil)
    #expect(globeInitialRoute(sites: [site]) == GlobeRoute(siteID: site.id))
    #expect(globeInitialRoute(sites: [site, site]) == nil)
  }

  @Test func processingNeverClaimsReadyBeforeServerCompletes() throws {
    for (raw, expected) in [
      ("awaiting-files", "Queued"), ("not-started", "Queued"),
      ("in-progress", "Refining"), ("complete", "Ready"), ("error", "Failed"),
      ("future-state", "Unavailable"),
    ] {
      let data = Data("{\"status\":\"\(raw)\",\"siteId\":null}".utf8)
      #expect(try JSONDecoder().decode(CatalogCaptureState.self, from: data).label == expected)
    }
  }

  @Test func readsServerAssignedSite() async throws {
    let siteID = UUID()
    let transport = FixtureTransport([
      Data("{\"status\":\"complete\",\"siteId\":\"\(siteID)\"}".utf8)
    ])
    let captureID = UUID()
    let state = try await CatalogCaptureState.fetch(
      captureID: captureID, baseURL: URL(string: "https://example.invalid/api/v1")!,
      transport: transport)
    #expect(state.siteId == siteID)
    let urls = await transport.urls
    #expect(urls.first?.path == "/api/v1/captures/\(captureID.uuidString)")
  }

  @Test func relativeTilesResolveOnWebHostAndRejectFileURLs() async throws {
    let fixture = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let asset = GlobeAsset(
      id: UUID(), epochId: fixture.property.epochs[0].id, representation: "gaussian-splat",
      source: GlobeAsset.Source(type: "3d-tiles-url", url: "/r2/scan/tileset.json"))
    let snapshot = GlobeSnapshot(property: GlobeProperty(
      id: fixture.property.id, name: fixture.property.name, boundary: fixture.property.boundary,
      epochs: fixture.property.epochs, assets: [asset]), pins: [])
    #expect(snapshot.resolvingAssetURLs(relativeTo: URL(string: "https://web.invalid")!)
      .bridgeSnapshot().scans[0].tilesetURL == "https://web.invalid/r2/scan/tileset.json")
    #expect(snapshot.resolvingAssetURLs(relativeTo: nil).bridgeSnapshot().scans[0].tilesetURL == nil)
  }
}
