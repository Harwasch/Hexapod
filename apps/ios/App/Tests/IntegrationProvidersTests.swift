import Cloud
import EntitySchema
import Foundation
import Globe
import LiveSplat
import Testing
import Upload

@testable import AppSupport

@Test @MainActor func configuredCloudAndSiblingProvidersResolve() async throws {
  let suite = "hex.integration.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let configuration = ServiceConfiguration(
    baseURL: URL(string: "https://fixture.invalid/api/v1"), writeToken: "fixture")
  let registry = PieceRegistry(defaults: defaults, configuration: configuration)
  #expect(registry.choice("deep") == "cloud")
  #expect(registry.choice("liveSplat") == "real")
  #expect(registry.choice("globe") == "real")
  #expect(try registry.deep.resolve("cloud") is CloudAnswerer)
  #expect(try registry.deep.resolve("fake") is FakeDeepAnswering)
  #expect(try registry.globeSources.resolve("real") is APIGlobeDataSource)
  #expect(configuration.cloudBase?.absoluteString == "https://fixture.invalid/api/")
  #expect(configuration.apiRoot?.absoluteString == "https://fixture.invalid/api/v1")
  #expect(try registry.splatRenderers.resolve("off") == nil)
  let fake = try #require(try registry.splatRenderers.resolve("fake"))
  fake.start()
  let source = try FakeFrameSource.plane(frameCount: 1)
  fake.ingest(try #require(try await source.next()))
  #expect(fake.coverageFraction > 0)
  fake.stop()
  let snapshot = try await registry.globeSources.resolve("fake").snapshot(
    siteID: FakeGlobeDataSource.siteID)
  #expect(snapshot.pins.count == 1 && snapshot.pins[0].coordinate != nil)
}

@Test func localCapturesExcludeIncompleteAndReadUploadStatus() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let complete = root.appendingPathComponent("Captures/complete")
  try FileManager.default.createDirectory(at: complete, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(
    at: root.appendingPathComponent("Captures/incomplete"), withIntermediateDirectories: true)
  try Data("{}".utf8).write(to: complete.appendingPathComponent("manifest.json"))
  var state = try UploadState(name: "Yard", totalBytes: 1)
  state.jobID = "fixture-job"
  let upload = root.appendingPathComponent("Uploads/complete")
  try FileManager.default.createDirectory(at: upload, withIntermediateDirectories: true)
  try state.save(to: upload.appendingPathComponent("upload.json"))
  let captures = try LocalCapture.list(at: root.appendingPathComponent("Captures"))
  #expect(captures.count == 1 && captures[0].status == "Uploaded")
}
