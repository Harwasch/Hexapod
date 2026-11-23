import Foundation
import Testing

@testable import Globe

struct CatalogStateTests {
  // JobRead/JobStepRead shape, using the wrapped normalize error from
  // tools/pipeline/tests/test_quality_gates.py. No production request or capture data.
  private func jobFixture() throws -> [String: Any] {
    let url = Bundle.module.url(forResource: "normalize-error", withExtension: "json", subdirectory: "Fixtures")!
    return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
  }
  private func state(_ json: [String: Any]) throws -> CatalogCaptureState {
    try JSONDecoder().decode(CatalogCaptureState.self, from: JSONSerialization.data(withJSONObject: json))
  }
  @Test func normalizeFailurePreservesFullErrorAndUsesShortReason() throws {
    let value = try state(jobFixture())
    #expect(value.label == "Failed · Could not prepare capture")
    #expect(value.fullError?.contains("QualityGateError") == true)
    #expect(value.fullError?.contains("Traceback") == true)
  }
  @Test func onlyRunningTrainUsesValidServerProgress() throws {
    for (stage, progress, expected) in [
      ("normalize", ["done": 1, "total": 2], "Processing"),
      ("pose", ["done": 1, "total": 2], "Processing"),
      ("train", ["done": 11100, "total": 30000], "Training 37%"),
      ("train", [:], "Training"),
      ("train", ["done": 1, "total": 0], "Training"),
      ("train", ["done": -1, "total": 2], "Training"),
      ("train", ["done": 3, "total": 2], "Training"),
    ] {
      var json = try jobFixture()
      json["status"] = "in-progress"
      var job = (json["jobs"] as! [[String: Any]])[0]
      job["status"] = "in-progress"
      job["error"] = NSNull()
      var step = (job["steps"] as! [[String: Any]])[0]
      step["status"] = "in-progress"
      step["stageId"] = stage
      step["metrics"] = ["progress": progress, "unrelated": ["key": "value"]]
      job["steps"] = [step]
      json["jobs"] = [job]
      #expect(try state(json).label == expected)
      step["metrics"] = ["progress": progress, "preparing": true]
      job["steps"] = [step]
      json["jobs"] = [job]
      #expect(try state(json).label == "Processing")
      step["status"] = "complete"
      job["steps"] = [step]
      json["jobs"] = [job]
      #expect(try state(json).label == "Processing")
    }
  }
  @Test func latestRunWinsEvenWhenJobsArriveOutOfOrder() throws {
    var json = try jobFixture()
    let old = (json["jobs"] as! [[String: Any]])[0]
    var latest = old
    latest["createdAt"] = "2026-11-02T00:00:00Z"
    latest["status"] = "complete"
    latest["error"] = NSNull()
    latest["steps"] = []
    json["status"] = "complete"
    for jobs in [[old, latest], [latest, old]] {
      json["jobs"] = jobs
      #expect(try state(json).label == "Ready")
      #expect(try state(json).fullError == nil)
    }
    latest["status"] = "cancelled"
    json["jobs"] = [latest]
    #expect(try state(json).label == "Failed · Processing cancelled")
  }
  @Test func onlyOneSiteSkipsPicker() {
    let site = GlobeSite(id: UUID(), name: "Field")
    #expect(globeInitialRoute(sites: []) == nil)
    #expect(globeInitialRoute(sites: [site]) == GlobeRoute(siteID: site.id))
    #expect(globeInitialRoute(sites: [site, site]) == nil)
  }

  @Test func processingNeverClaimsReadyBeforeServerCompletes() throws {
    for (raw, expected) in [
      ("awaiting-files", "Processing"), ("not-started", "Processing"),
      ("in-progress", "Processing"), ("complete", "Ready"), ("error", "Failed"),
      ("future-state", "Connecting…"),
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
