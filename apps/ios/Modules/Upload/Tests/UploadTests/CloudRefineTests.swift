import Foundation
import Networking
import Testing

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

@testable import Upload

private actor StoredTransport: APITransport {
  var paths: [String] = []
  func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    let url = request.url!
    paths.append(url.path)
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-key")
    let json = url.path.hasSuffix("bundle-upload")
      ? "{\"id\":\"file\",\"status\":\"complete\",\"bytes\":3}"
      : "{\"id\":\"job\"}"
    return (Data(json.utf8), HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!)
  }
}

@Test func uploadStopsAtStorageAndRefineIsExplicitAndDurable() async throws {
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  try Data([1, 2, 3]).write(to: directory.appendingPathComponent("capture.hexcap"))
  let base = URL(string: "https://api.example.test/api/v1")!
  var state = try UploadState(name: "Scan", totalBytes: 3)
  state.apiBase = base.absoluteString
  state.captureID = "capture"
  try state.save(to: directory.appendingPathComponent("upload.json"))
  let transport = StoredTransport()
  let uploader = try BundleUploader(
    directory: directory, apiBase: base, token: { "test-key" }, transport: transport)
  await #expect(throws: UploadError.self) { try await uploader.refineInCloud() }
  try await uploader.resume()
  let stored = await uploader.snapshot
  #expect(stored.uploadComplete && stored.jobID == nil)
  #expect(await transport.paths == ["/api/v1/captures/capture/bundle-upload"])
  let restored = try BundleUploader(
    directory: directory, apiBase: base, token: { "test-key" }, transport: transport)
  try await restored.resume()
  #expect(await transport.paths.count == 1)
  try await restored.refineInCloud()
  #expect(await transport.paths.last == "/api/v1/captures/capture/process-bundle")
  #expect(try UploadState.load(from: directory.appendingPathComponent("upload.json")).jobID == "job")
}
