import Globe
import Upload

/// Upload bytes describe transfer only; reconstruction always comes from the server.
enum CaptureStatus {
  static func label(upload: UploadState?, remote: CatalogCaptureState?, unavailable: Bool) -> String
  {
    guard let upload else { return "Connecting…" }
    if upload.jobID != nil {
      return unavailable ? "Connection unavailable" : (remote?.label ?? "Connecting…")
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
