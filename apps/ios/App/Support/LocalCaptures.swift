import Foundation
import Upload

struct CaptureDetails: Codable, Sendable {
  var siteName: String?
  var siteID: UUID?
  var duration: TimeInterval
}

struct LocalCapture: Identifiable, Sendable {
  let directory: URL
  let upload: UploadState?
  var details: CaptureDetails? {
    guard let data = try? Data(contentsOf: directory.appendingPathComponent("capture-info.json"))
    else { return nil }
    return try? JSONDecoder().decode(CaptureDetails.self, from: data)
  }
  var uploadDirectory: URL {
    directory.deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("Uploads/\(id)")
  }
  var id: String { directory.lastPathComponent }
  var status: String {
    if let upload {
      if upload.lastError != nil { return "Retry" }
      if upload.jobID != nil { return "Uploaded" }
      return "Uploading"
    }
    return "Local"
  }
  static func list(at root: URL) throws -> [LocalCapture] {
    guard FileManager.default.fileExists(atPath: root.path) else { return [] }
    return try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: nil
    ).filter {
      FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path)
    }.map {
      LocalCapture(
        directory: $0,
        upload: try? UploadState.load(
          from:
            root.deletingLastPathComponent().appendingPathComponent(
              "Uploads/\($0.lastPathComponent)/upload.json")))
    }.sorted { $0.id < $1.id }
  }
}
