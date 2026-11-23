#if canImport(UIKit) && canImport(WebKit)
  import Foundation
  import Testing
  import WebKit

  @testable import Globe

  /// Exercises the callback-based JavaScript overload and native event round trip on iOS.
  /// Disabled in package tests: they run with no host app, where WebKit cannot start its web
  /// content process, so the page never loads (CI run 44584652376 timed out here while every
  /// pure bridge test passed). Re-enable in an app-hosted test target or verify on a device.
  @Test(.disabled("WKWebView needs an app host; package tests have none"))
  @MainActor func localPageReceivesQueuedSnapshotAndFocus() async throws {
    let snapshot = try await FakeGlobeDataSource().snapshot(siteID: FakeGlobeDataSource.siteID)
    let session = CesiumGlobeSession()
    defer { session.webView.stopLoading() }
    let (events, continuation) = AsyncStream<UUID>.makeStream()
    defer { continuation.finish() }
    session.update(
      owner: UUID(), snapshot: snapshot, scanID: FakeGlobeDataSource.scanID, fixture: true,
      onPin: { _ in }, onScan: { continuation.yield($0) })
    let selected = try await withThrowingTaskGroup(of: UUID.self) { group in
      group.addTask {
        for await id in events { return id }
        throw CancellationError()
      }
      group.addTask {
        try await Task.sleep(for: .seconds(10))
        throw BridgeTimeout()
      }
      defer { group.cancelAll() }
      return try #require(try await group.next())
    }
    #expect(selected == FakeGlobeDataSource.scanID)
    #expect(session.error == nil)
  }

  private struct BridgeTimeout: Error {}
#endif
