import Foundation
import Globe
import Testing
import Upload

@testable import AppSupport

#if DEBUG
  @Test func inProcessFixtureStoresThenRefinesOnlyOnRequestAndSurvivesRecreation() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let capture = LocalCapture(directory: root.appendingPathComponent("Captures/test"), upload: nil)
    let api: any FixtureCaptureServing = InProcessFixtureCaptureAPI()
    let stored = try await api.store(capture, siteName: "Simulated yard", siteID: nil)
    #expect(stored.uploadComplete && stored.jobID == nil)
    #expect(stored.name == "Simulated yard")
    #expect(stored.apiBase == nil)
    let before = try await api.status(capture)
    #expect(
      CaptureStatus.label(upload: stored, remote: before, unavailable: false) == "Stored in cloud")
    #expect(CaptureStatus.refineEnabled(upload: stored, remote: before, unavailable: false))
    let refined = try await api.refine(capture)
    #expect(refined.jobID != nil)
    let restarted = InProcessFixtureCaptureAPI()
    let resumed = try await restarted.store(capture, siteName: nil, siteID: nil)
    #expect(resumed == refined)
    let after = try await restarted.status(capture)
    #expect(CaptureStatus.label(upload: resumed, remote: after, unavailable: false) == "Ready")
    #expect(!CaptureStatus.refineVisible(upload: resumed, remote: after))
    #expect(
      CaptureStatus.trainedRoute(upload: resumed, remote: after)?.siteID
        == FakeGlobeDataSource.siteID)
  }
#endif

#if DEBUG
  import PhoneTrain

  @Test @MainActor func fixtureSessionResultAndQueueReachReadyWithoutConfiguredServices() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let capture = LocalCapture(directory: root.appendingPathComponent("Captures/test"), upload: nil)
    let api: any FixtureCaptureServing = InProcessFixtureCaptureAPI()
    let queue = CaptureUploadQueue(fixtureAPI: api)
    await queue.enqueue(capture)
    let stored = try #require(queue.states[capture.id])
    let config = ServiceConfiguration(baseURL: nil)
    let before = try #require(try await CaptureStatus.fetch(capture, configuration: config, fixture: api))
    #expect(CaptureStatus.label(upload: stored, remote: before, unavailable: false) == "Stored in cloud")
    let output = root.appendingPathComponent("Training")
    let png = Data([137, 80, 78, 71])
    let session = try await PhoneTrainingFactory.make(outputDirectory: output) {
      FixturePhoneTrainingSession(outputDirectory: output, previewPNG: png)
    }
    let result = try await session.run()
    let preview = try await session.previewImage()
    let record = try PhoneSplatRecord.save(ply: result.ply, sharpness: nil, checkpoint: nil,
      for: capture, previewPNG: preview, simulated: result.simulated)
    try await session.closeAndCleanup()
    #expect(record.simulated == true && record.sharpness == nil)
    #expect(capture.phoneSplat != nil)
    try await queue.requestRefinement(capture)
    let after = try #require(try await CaptureStatus.fetch(capture, configuration: config, fixture: api))
    #expect(CaptureStatus.label(upload: queue.states[capture.id], remote: after, unavailable: false) == "Ready")
    #expect(!CloudRefineIntent.pending(for: capture))
    #expect(queue.pendingRefinements.isEmpty)
    let route = try #require(CaptureStatus.trainedRoute(upload: queue.states[capture.id], remote: after))
    let scanID = try #require(route.scanID)
    try CloudSplatRecord(siteID: route.siteID, scanID: scanID).save(for: capture)
    #expect(CaptureQuality(phone: capture.phoneSplat != nil, cloud: capture.cloudSplat != nil).symbols.count == 2)
  }
#endif
