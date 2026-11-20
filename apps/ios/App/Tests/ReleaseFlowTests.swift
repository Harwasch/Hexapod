import Capture
import Foundation
import Testing
import Upload

@testable import AppSupport

@Test @MainActor func missingConfigurationKeepsRealGlobeAndIgnoresStoredFakes() throws {
  let suite = "hex.uxa.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  let registry = PieceRegistry(defaults: defaults, configuration: .init(baseURL: nil))
  #expect(registry.choice("globe") == "real")
  #expect(throws: ProviderError.self) { try registry.globeSources.resolve("real") }
  #expect(!registry.configuration.cloudConfigured)
}

@Test func plistConfigurationIgnoresLegacyPlaintextToken() throws {
  let suite = "hex.uxa.\(UUID().uuidString)"
  let defaults = try #require(UserDefaults(suiteName: suite))
  defer { defaults.removePersistentDomain(forName: suite) }
  defaults.set("legacy-secret", forKey: "hex.api.writeToken")
  let config = ServiceConfiguration(defaults: defaults)
  #expect(config.writeToken != "legacy-secret")
}

#if !os(iOS)
  @Test @MainActor func missingCredentialsKeepCaptureLocalAndShowSignIn() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let directory = root.appendingPathComponent("Captures/queued")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: directory.appendingPathComponent("manifest.json"))
    let capture = LocalCapture(directory: directory, upload: nil)
    let queue = CaptureUploadQueue()
    await queue.enqueue(capture)
    #expect(queue.issues[capture.id] == "Sign in")
    #expect(queue.states[capture.id] == nil)
    #expect(try LocalCapture.list(at: root.appendingPathComponent("Captures")).count == 1)
    #expect(!FileManager.default.fileExists(atPath: capture.uploadDirectory.path))
  }
#endif

@Test @MainActor func uploadRingUsesAcknowledgedBytesAndOnlyCompletesWhenAcknowledged() throws {
  var state = try UploadState(name: "Field", totalBytes: 12 * 1024 * 1024)
  state.partSize = 5 * 1024 * 1024
  #expect(CaptureUploadQueue.progress(state) == 0)
  try state.acknowledge(part: 1, etag: "one")
  #expect(CaptureUploadQueue.progress(state) == 5.0 / 12)
  try state.acknowledge(part: 3, etag: "three")
  #expect(CaptureUploadQueue.progress(state) == 7.0 / 12)
  state.jobID = "queued-for-refinement"
  #expect(CaptureUploadQueue.progress(state) == 1)
}

@Test func fixtureRecordingPublishesACompleteBundleThatCanBeArchived() async throws {
  let ios = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
  let source = ios.appendingPathComponent("Fixtures/bundles/yard-small/bundle")
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let destination = root.appendingPathComponent("Captures/fixture")
  let siteID = UUID()
  let saved = try await FixtureCaptureRecording.save(
    source: source, destination: destination,
    details: CaptureDetails(siteName: "Simulated field", siteID: siteID, duration: 3))
  let local = try #require(try LocalCapture.list(at: root.appendingPathComponent("Captures")).first)
  #expect(local.directory.standardizedFileURL.path == saved.standardizedFileURL.path)
  #expect(local.details?.siteID == siteID)
  #expect(
    try Data(contentsOf: saved.appendingPathComponent("rgb.mov"))
      == Data(contentsOf: source.appendingPathComponent("rgb.mov")))
  let archive = root.appendingPathComponent("fixture.hexcap")
  try CaptureArchive.write(bundle: saved, to: archive)
  let manifest = try CaptureArchive.readManifest(from: archive)
  #expect(manifest.frameCount > 0 && manifest.depthFrameCount == manifest.frameCount)
  #expect(manifest.device == "Simulated fixture")
}
