import Foundation
import Globe
import Upload

/// Upload bytes describe transfer only; reconstruction always comes from the server.
enum CaptureStatus {
  /// Display estimate per scan, not a quote; keep confirmation copy sourced here.
  static let cloudRefineEstimatedCost = "$1-3"

  private static func jobRunning(_ remote: CatalogCaptureState?) -> Bool {
    // Globe's catalog label already follows the latest job and active training step.
    remote?.jobs?.isEmpty == false
      && (remote?.label == "Processing" || remote?.label.hasPrefix("Training") == true)
  }
  static func refineVisible(upload: UploadState?, remote: CatalogCaptureState?) -> Bool {
    upload?.uploadComplete == true && upload?.captureID != nil && remote?.status != "complete"
  }
  static func refineEnabled(
    upload: UploadState?, remote: CatalogCaptureState?, unavailable: Bool
  ) -> Bool {
    refineVisible(upload: upload, remote: remote) && !unavailable && remote != nil
      && !jobRunning(remote)
      && (remote?.label.hasPrefix("Failed") == true
        || (remote?.status == "not-started" && upload?.jobID == nil))
  }
  static func symbol(label: String) -> String {
    label == "Stored in cloud" ? "checkmark.icloud" : "icloud"
  }
  static func trainedRoute(upload: UploadState?, remote: CatalogCaptureState?) -> GlobeRoute? {
    guard remote?.status == "complete", let siteID = remote?.siteId,
      let scanID = upload?.captureID.flatMap(UUID.init(uuidString:))
    else { return nil }
    return GlobeRoute(siteID: siteID, scanID: scanID)
  }
  static func label(upload: UploadState?, remote: CatalogCaptureState?, unavailable: Bool) -> String
  {
    guard let upload else { return "Connecting…" }
    if upload.uploadComplete {
      if unavailable { return "Connection unavailable" }
      if upload.jobID == nil, remote?.status == "not-started", remote?.jobs?.isEmpty == true {
        return "Stored in cloud"
      }
      return remote?.label ?? "Connecting…"
    }
    if upload.lastError != nil { return "Failed · Upload failed" }
    guard let size = upload.partSize, upload.totalBytes > 0 else { return "Uploading 0%" }
    let bytes = upload.completed.reduce(UInt64(0)) { result, part in
      let offset = UInt64(max(0, part.partNumber - 1)) * size
      return result + min(size, upload.totalBytes - min(offset, upload.totalBytes))
    }
    return "Uploading \(Int(min(1, Double(bytes) / Double(upload.totalBytes)) * 100))%"
  }
}
