import Foundation
import Globe
import Testing
import Upload

@testable import AppSupport

@Test func storedUploadMapsToCloudCheckAndRefineWaitsForCompletion() throws {
  var upload = try UploadState(name: "Scan", totalBytes: 100)
  upload.captureID = UUID().uuidString
  #expect(!CaptureStatus.refineVisible(upload: upload, remote: nil))
  upload.storedInCloud = true
  let stored = try JSONDecoder().decode(
    CatalogCaptureState.self, from: Data("{\"status\":\"not-started\",\"jobs\":[]}".utf8))
  #expect(CaptureStatus.label(upload: upload, remote: stored, unavailable: false) == "Stored in cloud")
  #expect(CaptureStatus.symbol(label: "Stored in cloud") == "checkmark.icloud")
  #expect(CaptureStatus.refineVisible(upload: upload, remote: stored))
  #expect(CaptureStatus.refineEnabled(upload: upload, remote: stored, unavailable: false))
  #expect(!CaptureStatus.refineEnabled(upload: upload, remote: stored, unavailable: true))
  let local = LocalCapture(directory: URL(fileURLWithPath: "/fixture"), upload: upload)
  #expect(local.status == "Stored in cloud")
  for status in ["not-started", "in-progress"] {
    let running = try JSONDecoder().decode(
      CatalogCaptureState.self,
      from: Data("{\"status\":\"not-started\",\"jobs\":[{\"status\":\"\(status)\",\"createdAt\":\"2026\",\"steps\":[]}]}".utf8))
    #expect(CaptureStatus.refineVisible(upload: upload, remote: running))
    #expect(!CaptureStatus.refineEnabled(upload: upload, remote: running, unavailable: false))
    #expect(CaptureStatus.label(upload: upload, remote: running, unavailable: false) == "Processing")
  }
  upload.jobID = UUID().uuidString
  #expect(!CaptureStatus.refineEnabled(upload: upload, remote: nil, unavailable: false))
  let training = try JSONDecoder().decode(
    CatalogCaptureState.self,
    from: Data("{\"status\":\"in-progress\",\"jobs\":[{\"status\":\"in-progress\",\"createdAt\":\"2026\",\"steps\":[{\"stageId\":\"train\",\"status\":\"in-progress\",\"ordinal\":1,\"metrics\":{}}]}]}".utf8))
  #expect(CaptureStatus.label(upload: upload, remote: training, unavailable: false) == "Training")
  #expect(!CaptureStatus.refineEnabled(upload: upload, remote: training, unavailable: false))
  let ready = try JSONDecoder().decode(
    CatalogCaptureState.self, from: Data("{\"status\":\"complete\",\"jobs\":[]}".utf8))
  #expect(!CaptureStatus.refineVisible(upload: upload, remote: ready))
}

@Test func uploadPercentageNeverClaimsTrainingOrQueuesUnknownServerState() throws {
  #expect(CaptureStatus.label(upload: nil, remote: nil, unavailable: false) == "Connecting…")
  var upload = try UploadState(name: "Scan", totalBytes: 100)
  upload.partSize = 100
  #expect(CaptureStatus.label(upload: upload, remote: nil, unavailable: false) == "Uploading 0%")
  upload.completed = try JSONDecoder().decode(
    [CompletedPart].self, from: Data("[{\"partNumber\":1,\"etag\":\"fixture\"}]".utf8))
  #expect(CaptureStatus.label(upload: upload, remote: nil, unavailable: false) == "Uploading 100%")
  upload.jobID = UUID().uuidString
  #expect(CaptureStatus.label(upload: upload, remote: nil, unavailable: false) == "Connecting…")
  let ready = try JSONDecoder().decode(
    CatalogCaptureState.self, from: Data("{\"status\":\"complete\",\"siteId\":null}".utf8))
  #expect(CaptureStatus.label(upload: upload, remote: ready, unavailable: false) == "Ready")
  #expect(
    CaptureStatus.label(upload: upload, remote: ready, unavailable: true)
      == "Connection unavailable")
}

@Test func trainedScanRouteRequiresServerReadyAndUsesServerSite() throws {
  var upload = try UploadState(name: "Scan", totalBytes: 100)
  let scan = UUID()
  let site = UUID()
  upload.captureID = scan.uuidString
  upload.jobID = UUID().uuidString
  for status in ["created", "processing", "error", "complete"] {
    let remote = try JSONDecoder().decode(
      CatalogCaptureState.self,
      from: Data("{\"status\":\"\(status)\",\"siteId\":\"\(site)\"}".utf8))
    #expect(
      CaptureStatus.trainedRoute(upload: upload, remote: remote)
        == (status == "complete" ? GlobeRoute(siteID: site, scanID: scan) : nil))
  }
  #expect(CaptureStatus.trainedRoute(upload: upload, remote: nil) == nil)
}

@Test func failedCloudRefinementCanBeRequestedAgain() throws {
  var upload = try UploadState(name: "Scan", totalBytes: 100)
  upload.captureID = UUID().uuidString
  upload.storedInCloud = true
  upload.jobID = UUID().uuidString
  for status in ["error", "cancelled"] {
    let remote = try JSONDecoder().decode(
      CatalogCaptureState.self,
      from: Data("{\"status\":\"not-started\",\"jobs\":[{\"status\":\"\(status)\",\"createdAt\":\"2026\",\"steps\":[]}]}".utf8))
    #expect(CaptureStatus.refineEnabled(upload: upload, remote: remote, unavailable: false))
  }
}
