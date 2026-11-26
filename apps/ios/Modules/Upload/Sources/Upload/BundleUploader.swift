import Networking
import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private struct Identifier: Decodable { var id: String }
private struct RemoteFile: Decodable {
  var id: String
  var status: String
  var bytes: UInt64?
}
private struct PartURL: Decodable {
  var partNumber: Int
  var url: URL
}
private struct Window: Decodable {
  var partSize: UInt64
  var partsTotal: Int
  var parts: [PartURL]
}
struct Begin: Encodable {
  var name: String
  let kind = "video"
  var siteId: UUID?
  var metadata: [String: Double]

  init(name: String, siteId: UUID?, placement: BundlePlacement?) {
    self.name = name
    self.siteId = siteId
    metadata =
      placement.map {
        [
          "lat": $0.lat, "lon": $0.lon, "uncertaintyM": $0.uncertaintyM,
          "locationAccuracyM": $0.uncertaintyM,
        ]
      } ?? [:]
  }
}
private struct Register: Encodable {
  let filename = "capture.hexcap"
  let contentType = "application/x-hex-capture"
  var bytes: UInt64
}
private struct Parts: Encodable {
  var firstPartNumber: Int
  let count = 1
}
private struct Complete: Encodable { var parts: [CompletedPart] }

/// One durable archive per instance. Recreate with the same directory/base URL and
/// call resume after relaunch; supply credentials from the app's Keychain each time.
/// API control requests resume on launch/background wake; iOS owns in-flight part PUTs.
public actor BundleUploader {
  private let directory: URL
  private let apiBase: URL
  private let token: @Sendable () async throws -> String
  private let transport: any APITransport
  private var state: UploadState
  private var background: URLSession?
  private var pumping = false
  private var pumpAgain = false
  private var eventsFinished: (@MainActor @Sendable () -> Void)?

  public init(
    directory: URL, archive: URL? = nil, name: String = "Capture",
    placement: BundlePlacementSource = .bundleGPS, siteID: UUID? = nil, apiBase: URL,
    token: @escaping @Sendable () async throws -> String,
    transport: any APITransport = URLSessionAPITransport()
  ) throws {
    guard
      apiBase.scheme == "https"
        || (apiBase.scheme == "http" && ["localhost", "127.0.0.1"].contains(apiBase.host ?? ""))
    else { throw UploadError.invalidURL }
    self.directory = directory
    self.apiBase = apiBase
    self.token = token
    self.transport = transport
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let stateURL = directory.appendingPathComponent("upload.json")
    let owned = directory.appendingPathComponent("capture.hexcap")
    if FileManager.default.fileExists(atPath: stateURL.path) {
      state = try UploadState.load(from: stateURL)
      guard state.apiBase == apiBase.absoluteString else { throw UploadError.invalidState }
    } else {
      guard let archive else { throw UploadError.invalidState }
      // The immutable private copy outlives recording folders and background tasks.
      if archive.resolvingSymlinksInPath().standardizedFileURL
        != owned.resolvingSymlinksInPath().standardizedFileURL
      {
        if FileManager.default.fileExists(atPath: owned.path) {
          try FileManager.default.removeItem(at: owned)
        }
        try FileManager.default.copyItem(at: archive, to: owned)
      }
      let size = try FileManager.default.attributesOfItem(atPath: owned.path)[.size] as! NSNumber
      state = try UploadState(name: name, totalBytes: size.uint64Value)
      state.apiBase = apiBase.absoluteString
      state.placement = try placement.resolve(archive: owned, siteID: siteID)
      state.siteID = siteID
      try state.save(to: stateURL)
    }
  }

  public var snapshot: UploadState { state }
  public var sessionIdentifier: String { "hex.bundle." + state.bundleID.uuidString.lowercased() }

  /// Forward UIApplication's background-session completion handler here before resume.
  /// Retain this uploader until events have drained. No App or UI dependency is needed.
  public func onBackgroundEventsFinished(_ completion: @escaping @MainActor @Sendable () -> Void) {
    eventsFinished = completion
  }

  private func save() throws { try state.save(to: directory.appendingPathComponent("upload.json")) }

  private func session() -> URLSession {
    if let background { return background }
    let config: URLSessionConfiguration
    #if os(iOS)
      config = .background(withIdentifier: sessionIdentifier)
      config.sessionSendsLaunchEvents = true
      config.isDiscretionary = false
    #else
      // Linux tests compile the protocol/state; only iOS has durable background PUTs.
      config = .default
    #endif
    config.httpMaximumConnectionsPerHost = 1
    let delegate = TransferDelegate(
      completed: { [weak self] part, status, etag, failed in
        await self?.completed(part: part, status: status, etag: etag, failed: failed)
      }, drained: { [weak self] in await self?.drained() })
    let queue = OperationQueue()
    queue.maxConcurrentOperationCount = 1
    let result = URLSession(configuration: config, delegate: delegate, delegateQueue: queue)
    background = result
    return result
  }

  private func request<Response: Decodable, Body: Encodable>(
    _ path: String, method: String, body: Body, as: Response.Type, refine: Bool = false
  ) async throws -> Response {
    var request = URLRequest(url: apiBase.appendingPathComponent(path))
    request.timeoutInterval = 90
    request.httpMethod = method
    if refine { request.setValue("1", forHTTPHeaderField: "X-Hex-Refine") }
    request.setValue("Bearer " + (try await token()), forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(body)
    let (data, response) = try await transport.data(for: request)
    let status = response.statusCode
    guard (200..<300).contains(status) else { throw UploadError.http(status) }
    return try JSONDecoder().decode(Response.self, from: data)
  }

  public func resume() async throws {
    _ = session()  // Reconnect even when the final job response was already persisted.
    if pumping {
      pumpAgain = true
      return
    }
    pumping = true
    defer { pumping = false }
    repeat {
      pumpAgain = false
      do { try await pump() } catch {
        // Error descriptions can contain signed URLs. Persist only a category.
        state.lastError = "Upload needs retry"
        try save()
        throw error
      }
    } while pumpAgain
  }

  /// An explicit, confirmed user action. The server recovers the same job on retries.
  public func refineInCloud() async throws {
    guard state.uploadComplete, let captureID = state.captureID else {
      throw UploadError.invalidState
    }
    let job = try await request(
      "captures/" + captureID + "/process-bundle", method: "POST",
      body: [String: String](), as: Identifier.self, refine: true)
    state.jobID = job.id
    try save()
  }

  private func pump() async throws {
    guard !state.uploadComplete else { return }
    let session = session()
    let tasks = await withCheckedContinuation { continuation in
      session.getAllTasks { continuation.resume(returning: $0) }
    }
    if !tasks.isEmpty { return }
    let archive = directory.appendingPathComponent("capture.hexcap")
    let size = try FileManager.default.attributesOfItem(atPath: archive.path)[.size] as! NSNumber
    guard size.uint64Value == state.totalBytes else { throw UploadError.changedArchive }
    if state.captureID == nil {
      let capture = try await request(
        "captures/bundles/" + state.bundleID.uuidString,
        method: "PUT",
        body: Begin(name: state.name, siteId: state.siteID, placement: state.placement),
        as: Identifier.self)
      state.captureID = capture.id
      try save()
    }
    let capturePath = "captures/" + state.captureID!
    let file = try await request(
      capturePath + "/bundle-upload", method: "PUT",
      body: Register(bytes: state.totalBytes), as: RemoteFile.self)
    guard file.bytes == state.totalBytes else { throw UploadError.changedArchive }
    if let fileID = state.fileID, fileID != file.id { throw UploadError.invalidState }
    state.fileID = file.id
    try save()
    let filePath = capturePath + "/files/" + file.id
    if file.status == "complete" {
      state.storedInCloud = true
      state.lastError = nil
      try save()
      return
    }
    if state.partSize != nil, try state.nextPart() == nil {
      _ = try await request(
        filePath + "/complete", method: "POST",
        body: Complete(parts: state.completed), as: RemoteFile.self)
      pumpAgain = true
      return
    }
    let next = state.partSize == nil ? 1 : try state.nextPart()!
    let window = try await request(
      filePath + "/parts", method: "POST",
      body: Parts(firstPartNumber: next), as: Window.self)
    let plan = try ChunkPlan(totalBytes: state.totalBytes, partSize: window.partSize)
    guard plan.count == window.partsTotal, let part = window.parts.first,
      part.partNumber == next,
      part.url.scheme == "https" || apiBase.host == "localhost" || apiBase.host == "127.0.0.1"
    else { throw UploadError.invalidPlan }
    if let old = state.partSize, old != window.partSize { throw UploadError.invalidPlan }
    state.partSize = window.partSize
    state.lastError = nil
    try save()
    let chunk = try plan.chunk(next)
    let partFile = directory.appendingPathComponent("part.body")
    try Self.materialize(chunk, archive: archive, destination: partFile)
    var put = URLRequest(url: part.url)
    put.httpMethod = "PUT"
    // Never send the API bearer token to the signed object-storage URL.
    let task = session.uploadTask(with: put, fromFile: partFile)
    task.taskDescription = String(next)
    task.resume()
  }

  public static func materialize(_ chunk: UploadChunk, archive: URL, destination: URL) throws {
    let input = try FileHandle(forReadingFrom: archive)
    defer { try? input.close() }
    try input.seek(toOffset: chunk.offset)
    guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
      throw UploadError.invalidState
    }
    let output = try FileHandle(forWritingTo: destination)
    defer { try? output.close() }
    try output.truncate(atOffset: 0)
    var remaining = chunk.bytes
    while remaining > 0 {
      let data = try input.read(upToCount: Int(min(remaining, 1024 * 1024))) ?? Data()
      guard !data.isEmpty else { throw UploadError.changedArchive }
      try output.write(contentsOf: data)
      remaining -= UInt64(data.count)
    }
    try output.synchronize()
  }

  private func completed(part: Int?, status: Int, etag: String?, failed: Bool) async {
    do {
      guard !failed, (200..<300).contains(status), let part, let etag else {
        throw UploadError.http(status)
      }
      try state.acknowledge(part: part, etag: etag)
      try save()
      try await resume()
    } catch {
      state.lastError = "Upload needs retry"
      try? save()
    }
  }

  func drained() async {
    let completion = eventsFinished
    eventsFinished = nil
    // UIKit requires its background-session completion handler on the main thread.
    await completion?()
  }
}

/// URLSession calls delegates on one serial queue; chained tasks preserve event order
/// while persisting each ETag before allowing the OS completion handler to run.
private final class TransferDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
  let completed: @Sendable (Int?, Int, String?, Bool) async -> Void
  let drained: @Sendable () async -> Void
  private var pending: Task<Void, Never>?
  init(
    completed: @escaping @Sendable (Int?, Int, String?, Bool) async -> Void,
    drained: @escaping @Sendable () async -> Void
  ) {
    self.completed = completed
    self.drained = drained
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    let part = task.taskDescription.flatMap(Int.init)
    let response = task.response as? HTTPURLResponse
    let status = response?.statusCode ?? 0
    let etag = response?.value(forHTTPHeaderField: "ETag")
    let failed = error != nil
    let previous = pending
    pending = Task {
      await previous?.value
      await completed(part, status, etag, failed)
    }
  }
  func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
    let previous = pending
    pending = Task {
      await previous?.value
      await drained()
    }
  }
}
