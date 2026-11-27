import Foundation
import Globe
import Upload

/// In-process substitute for capture upload, catalog reads and explicit cloud refinement.
/// No URLSession is constructed; durable upload state still drives the real screens.
protocol FixtureCaptureServing: Sendable {
  func store(_ capture: LocalCapture, siteName: String?, siteID: UUID?) async throws -> UploadState
  func refine(_ capture: LocalCapture) async throws -> UploadState
  func status(_ capture: LocalCapture) async throws -> CatalogCaptureState
}

#if DEBUG
  actor InProcessFixtureCaptureAPI: FixtureCaptureServing {
    static let shared = InProcessFixtureCaptureAPI()

    func store(_ capture: LocalCapture, siteName: String?, siteID: UUID?) throws -> UploadState {
      if let existing = try? UploadState.load(from: stateURL(capture)) { return existing }
      let id = FakeGlobeDataSource.scanID
      let details = capture.details
      var state = try UploadState(
        bundleID: id, name: siteName ?? details?.siteName ?? "Simulated capture", totalBytes: 1)
      state.captureID = id.uuidString.lowercased()
      state.siteID = siteID ?? details?.siteID ?? FakeGlobeDataSource.siteID
      state.storedInCloud = true
      try save(state, capture: capture)
      return state
    }

    func refine(_ capture: LocalCapture) throws -> UploadState {
      var state = try UploadState.load(from: stateURL(capture))
      guard state.uploadComplete else { throw UploadError.invalidState }
      state.jobID = "00000000-0000-0000-0000-000000000006"
      try save(state, capture: capture)
      return state
    }

    func status(_ capture: LocalCapture) throws -> CatalogCaptureState {
      let state = try UploadState.load(from: stateURL(capture))
      let refined = state.jobID != nil
      let payload: [String: Any] = [
        "status": refined ? "complete" : "not-started",
        "siteId": (state.siteID ?? FakeGlobeDataSource.siteID).uuidString,
        "jobs": [],
      ]
      return try JSONDecoder().decode(
        CatalogCaptureState.self, from: JSONSerialization.data(withJSONObject: payload))
    }

    private func stateURL(_ capture: LocalCapture) -> URL {
      capture.uploadDirectory.appendingPathComponent("upload.json")
    }
    private func save(_ state: UploadState, capture: LocalCapture) throws {
      try FileManager.default.createDirectory(
        at: capture.uploadDirectory, withIntermediateDirectories: true)
      try state.save(to: stateURL(capture))
    }
  }
#endif
