import Foundation
import Globe
import Testing
import Upload

@testable import AppSupport

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
