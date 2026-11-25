import Foundation
import Testing

@testable import Globe

struct BridgeTests {
  @Test func sharedJSONRoundTrip() throws {
    let url = Bundle.module.url(
      forResource: "messages", withExtension: "json", subdirectory: "Fixtures")!
    let data = try Data(contentsOf: url)
    let messages = try JSONDecoder().decode([BridgeMessage].self, from: data)
    #expect(messages.count == 9)
    let encoded = try JSONEncoder().encode(messages)
    #expect(try JSONDecoder().decode([BridgeMessage].self, from: encoded) == messages)
    let original = try JSONSerialization.jsonObject(with: data) as! NSArray
    let roundTrip = try JSONSerialization.jsonObject(with: encoded) as! NSArray
    #expect(original == roundTrip)
    for invalid in [
      "{\"version\":2,\"type\":\"ready\"}", "{\"version\":1,\"type\":\"other\"}",
      "{\"version\":1,\"type\":\"pinTapped\"}",
    ] {
      #expect(throws: (any Error).self) {
        try JSONDecoder().decode(BridgeMessage.self, from: Data(invalid.utf8))
      }
    }
  }
  @Test func nativeDispatchChecksCurrentSnapshot() async throws {
    let snapshot = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let dispatcher = GlobeBridgeDispatcher()
    #expect(try dispatcher.dispatch(.ready, snapshot: nil) == .ready)
    let id = snapshot.pins[0].id
    #expect(
      try dispatcher.dispatch(.pinTapped(pinId: id.uuidString), snapshot: snapshot) == .pin(id))
    #expect(
      try dispatcher.dispatch(
        .scanSelected(scanId: FakeGlobeDataSource.scanID.uuidString), snapshot: snapshot)
        == .scan(FakeGlobeDataSource.scanID))
    #expect(throws: GlobeError.invalidResponse) {
      try dispatcher.dispatch(.pinTapped(pinId: UUID().uuidString), snapshot: snapshot)
    }
    #expect(throws: GlobeError.invalidResponse) {
      try dispatcher.dispatch(.selectScan(scanId: id.uuidString), snapshot: snapshot)
    }
    #expect(
      snapshot.captureRoute(for: FakeGlobeDataSource.scanID).scanID == FakeGlobeDataSource.scanID)
    #expect(GlobeImplementation.defaultValue == .cesium)
    #expect(snapshot.bridgeSnapshot().scans[0].tilesetURL == nil)
    #expect(
      snapshot.bridgeSnapshot(fixture: true).scans[0].tilesetURL
        == "./fixtures/globe-splat/tileset.json")
  }
  @Test func publishedAssetUsesExactEpochOrCurrentAsset() throws {
    let json = """
      {"id":"00000000-0000-0000-0000-000000000001","name":"Simulated","boundary":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]},"epochs":[{"id":"00000000-0000-0000-0000-000000000002","captureId":null,"observedAt":"2026-01-01","footprint":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}}],"assets":[{"id":"00000000-0000-0000-0000-000000000003","epochId":"00000000-0000-0000-0000-000000000002","representation":"gaussian-splat","source":{"type":"3d-tiles-url","url":"https://example.invalid/tileset.json"}}]}
      """
    let property = try JSONDecoder().decode(GlobeProperty.self, from: Data(json.utf8))
    let snapshot = GlobeSnapshot(property: property, pins: [])
    let bridge = snapshot.bridgeSnapshot()
    #expect(snapshot.captureRoute(for: property.epochs[0].id).scanID == nil)
    #expect(bridge.scans[0].tilesetURL == "https://example.invalid/tileset.json")
    #expect(bridge.scans[0].id == property.epochs[0].id.uuidString.lowercased())
  }
  @Test func readinessReplaysLatestStateAndDoesNotReloadUnchangedScan() async throws {
    let snapshot = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let wire = snapshot.bridgeSnapshot(fixture: true)
    let id = wire.scans[0].id
    var state = GlobeBridgeState()
    #expect(state.update(snapshot: wire, scanID: nil).isEmpty)
    let expected: [BridgeMessage] = [
      .snapshot(snapshot: wire), .selectScan(scanId: id), .flyToScan(scanId: id),
    ]
    #expect(state.becomeReady() == expected)
    #expect(state.update(snapshot: wire, scanID: id).isEmpty)
    state.invalidate()
    #expect(!state.ready)
    #expect(state.becomeReady() == expected)
    let empty = BridgeSnapshot(pins: [], scans: [], boundary: [])
    #expect(state.update(snapshot: empty, scanID: id) == [.snapshot(snapshot: empty)])
    state.invalidate()
    #expect(state.update(snapshot: wire, scanID: id).isEmpty)
    #expect(state.update(snapshot: empty, scanID: nil).isEmpty)
    #expect(state.becomeReady() == [.snapshot(snapshot: empty)])
  }

  @Test func missingScanNeverFocusesAnUnrelatedCapture() async throws {
    let snapshot = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let wire = snapshot.bridgeSnapshot(fixture: true)
    var state = GlobeBridgeState()
    _ = state.update(snapshot: wire, scanID: UUID().uuidString)
    #expect(state.becomeReady() == [.snapshot(snapshot: wire)])
  }

}
