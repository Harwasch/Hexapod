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
