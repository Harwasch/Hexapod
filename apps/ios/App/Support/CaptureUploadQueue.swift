import Capture
import Foundation
import Upload

#if os(iOS)
  import Observation
#endif

/// Retained independently of the camera screen. Archives, upload state and part PUTs
/// survive Done; relaunch reconnects URLSession using the same durable upload folder.
#if os(iOS)
  @Observable
#endif
@MainActor final class CaptureUploadQueue {
  static let shared = CaptureUploadQueue()
  private var uploaders: [String: BundleUploader] = [:]
  private var starting: Set<String> = []
  private(set) var states: [String: UploadState] = [:]
  private(set) var issues: [String: String] = [:]

  func enqueue(_ capture: LocalCapture, siteName: String? = nil, siteID: UUID? = nil) async {
    let id = capture.id
    guard !starting.contains(id) else { return }
    let config = ServiceConfiguration(defaults: .standard)
    guard !config.writeToken.isEmpty else {
      issues[id] = "Sign in"
      return
    }
    guard let api = config.apiRoot else {
      issues[id] = "API unavailable"
      return
    }
    starting.insert(id)
    defer { starting.remove(id) }
    do {
      let details = await Task.detached { capture.details }.value
      let chosenName = siteName ?? details?.siteName
      let chosenSiteID = siteID ?? details?.siteID
      let uploader: BundleUploader
      if let existing = uploaders[id] {
        uploader = existing
      } else {
        uploader = try await Task.detached(priority: .utility) {
          let directory = capture.uploadDirectory
          let archive = directory.appendingPathComponent("capture.hexcap")
          try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
          if !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("upload.json").path)
          {
            // Remove only an uncommitted archive left by an interrupted archive write.
            if FileManager.default.fileExists(atPath: archive.path) {
              try FileManager.default.removeItem(at: archive)
            }
            try CaptureArchive.write(bundle: capture.directory, to: archive)
          }
          return try BundleUploader(
            directory: directory, archive: archive,
            name: chosenName ?? "Capture", siteID: chosenSiteID, apiBase: api,
            token: {
              let token = APICredentials.token
              guard !token.isEmpty else { throw APICredentials.CredentialError.unavailable }
              return token
            })
        }.value
        uploaders[id] = uploader
      }
      issues[id] = nil
      try await uploader.resume()
      states[id] = await uploader.snapshot
    } catch UploadError.missingLocation { issues[id] = "Location needed" } catch {
      issues[id] = "Retry upload"
    }
  }

  func refresh(_ capture: LocalCapture) async {
    if let uploader = uploaders[capture.id] {
      states[capture.id] = await uploader.snapshot
    } else {
      states[capture.id] = try? UploadState.load(
        from: capture.uploadDirectory.appendingPathComponent("upload.json"))
    }
    if states[capture.id]?.lastError != nil { issues[capture.id] = "Retry upload" }
  }

  func resumeQueued() async {
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Captures")
    let captures = (try? await Task.detached { try LocalCapture.list(at: root) }.value) ?? []
    for capture in captures { await enqueue(capture) }
  }

  func reconnect(identifier: String, completion: @escaping @MainActor @Sendable () -> Void) async {
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Captures")
    let captures = (try? await Task.detached { try LocalCapture.list(at: root) }.value) ?? []
    for capture in captures {
      guard let state = capture.upload,
        identifier == "hex.bundle." + state.bundleID.uuidString.lowercased(),
        let base = state.apiBase.flatMap(URL.init(string:))
      else { continue }
      do {
        let uploader: BundleUploader
        if let existing = uploaders[capture.id] {
          uploader = existing
        } else {
          uploader = try await Task.detached {
            try BundleUploader(
              directory: capture.uploadDirectory, apiBase: base,
              token: { APICredentials.token })
          }.value
        }
        uploaders[capture.id] = uploader
        await uploader.onBackgroundEventsFinished(completion)
        try await uploader.resume()
        return
      } catch { break }
    }
    completion()
  }

  static func progress(_ state: UploadState?) -> Double {
    guard let state else { return 0 }
    if state.jobID != nil { return 1 }
    guard let size = state.partSize, state.totalBytes > 0 else { return 0 }
    let bytes = state.completed.reduce(UInt64(0)) { sum, part in
      let offset = UInt64(max(0, part.partNumber - 1)) * size
      return sum + min(size, state.totalBytes - min(offset, state.totalBytes))
    }
    return min(1, Double(bytes) / Double(state.totalBytes))
  }
}
