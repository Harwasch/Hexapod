import Foundation
import Testing
@testable import AppSupport

@Test func trainingSharpnessUsesOnlyFiniteHoldoutPSNR() {
  #expect(TrainingPresentation.sharpness(psnr: nil) == nil)
  #expect(TrainingPresentation.sharpness(psnr: .nan) == nil)
  #expect(TrainingPresentation.sharpness(psnr: .infinity) == nil)
  #expect(TrainingPresentation.sharpness(psnr: 5) == 0)
  #expect(TrainingPresentation.sharpness(psnr: 15) == 0)
  #expect(TrainingPresentation.sharpness(psnr: 25) == 50)
  #expect(TrainingPresentation.sharpness(psnr: 35) == 100)
  #expect(TrainingPresentation.sharpness(psnr: 50) == 100)
}
@Test func trainingLifecycleAndSwapRespectUserChoice() {
  var state = TrainingExperience()
  state.pause()
  #expect(state.stage == .choosing)
  state.begin(); state.prepared(); state.pause()
  #expect(state.stage == .paused)
  state.cool(); state.resume(); state.finish()
  #expect(state.stage == .settling)
  state.showResult()
  #expect(!state.stage.busy)
  var selection = SplatSelection()
  selection.select(.cloud)
  #expect(selection.source == .phone)
  selection.cloudReady()
  #expect(selection.source == .cloud)
  selection.select(.phone); selection.cloudReady()
  #expect(selection.source == .phone)
}
@Test func trainingETAAndBadgesAndIdentifiers() {
  #expect(TrainingPresentation.eta(.nan) == "—")
  #expect(TrainingPresentation.eta(1) == "~1 min")
  #expect(TrainingPresentation.eta(61) == "~2 min")
  #expect(CaptureQuality(phone: true, cloud: true).symbols == ["iphone", "cloud.fill"])
  #expect(CaptureQuality(phone: false, cloud: false).symbols.isEmpty)
  #expect(Set(TrainingID.all).count == TrainingID.all.count)
}
@Test func phoneResultIsDurableWithoutChangingCapture() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let bundle = root.appendingPathComponent("Captures/scan")
  try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
  let ply = root.appendingPathComponent("trained.ply")
  try Data("ply".utf8).write(to: ply)
  let capture = LocalCapture(directory: bundle, upload: nil)
  #expect(PhoneSplatRecord.load(for: capture) == nil)
  let first = try PhoneSplatRecord.save(ply: ply, sharpness: 70, checkpoint: nil, for: capture)
  let second = try PhoneSplatRecord.save(ply: ply, sharpness: 80, checkpoint: nil, for: capture)
  #expect(!FileManager.default.fileExists(atPath: first.url(for: capture).path))
  #expect(PhoneSplatRecord.load(for: capture)?.sharpness == 80)
  #expect(try Data(contentsOf: second.url(for: capture)) == Data("ply".utf8))
  #expect(try FileManager.default.contentsOfDirectory(atPath: bundle.path).isEmpty)
}

@Test func cloudConfirmationSurvivesUploadAndRelaunchUntilAcknowledged() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let capture = LocalCapture(directory: root.appendingPathComponent("Captures/scan"), upload: nil)
  #expect(!CloudRefineIntent.pending(for: capture))
  try CloudRefineIntent.confirm(for: capture)
  let reloaded = LocalCapture(directory: capture.directory, upload: nil)
  #expect(CloudRefineIntent.pending(for: reloaded))
  #expect(!FileManager.default.fileExists(atPath: capture.directory.path))
  try CloudRefineIntent.complete(for: reloaded)
  #expect(!CloudRefineIntent.pending(for: capture))
  try CloudRefineIntent.complete(for: reloaded)
}

@Test func cloudBadgeSurvivesOfflineAndSharesPublishedSite() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let capture = LocalCapture(directory: root.appendingPathComponent("Captures/scan"), upload: nil)
  let record = CloudSplatRecord(siteID: UUID(), scanID: UUID())
  try record.save(for: capture)
  let saved = try #require(CloudSplatRecord.load(for: capture))
  #expect(saved.scanID == record.scanID)
  let base = try #require(URL(string: "https://example.test"))
  let url = try #require(saved.shareURL(webBaseURL: base))
  #expect(url.path == "/view.html")
  #expect(url.fragment == record.siteID.uuidString.lowercased())
  #expect(capture.phoneSplat == nil)
}

#if !os(iOS)
@Test @MainActor func unconnectedCloudCannotRecordABillableIntent() async throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let capture = LocalCapture(directory: root.appendingPathComponent("Captures/scan"), upload: nil)
  let queue = CaptureUploadQueue()
  do {
    try await queue.requestRefinement(capture)
    Issue.record("Expected unavailable credentials")
  } catch {
    #expect(!CloudRefineIntent.pending(for: capture))
    #expect(queue.pendingRefinements.isEmpty)
  }
}
#endif

@Test func savedFixtureKeepsSimulationLabelInTrainingViewer() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  let capture = LocalCapture(directory: root, upload: nil)
  #expect(!capture.isSimulated)
  let manifest = root.appendingPathComponent("manifest.json")
  try Data("{\"device\":\"Simulated fixture\"}".utf8).write(to: manifest)
  #expect(capture.isSimulated)
  try Data("{\"device\":\"iPhone\"}".utf8).write(to: manifest)
  #expect(!capture.isSimulated)
}

@Test func inPlaceRequiresTheOriginalSessionAndSurvivesOrdinaryPause() {
  let first = NSObject()
  let replacement = NSObject()
  let capture = URL(fileURLWithPath: "/saved/scan")
  var world = SplatWorldOrigin()
  #expect(!world.permits(capture: capture, session: ObjectIdentifier(first)))
  world.record(capture: capture, session: ObjectIdentifier(first))
  // A pause creates no new session and performs no reset.
  #expect(world.permits(capture: capture, session: ObjectIdentifier(first)))
  #expect(!world.permits(capture: capture, session: ObjectIdentifier(replacement)))
  #expect(!world.permits(capture: URL(fileURLWithPath: "/saved/other"), session: ObjectIdentifier(first)))
  world.invalidate()
  #expect(!world.permits(capture: capture, session: ObjectIdentifier(first)))
}
