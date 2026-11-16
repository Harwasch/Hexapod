import Foundation
import Testing

@testable import Upload

@Test func chunkBoundaries() throws {
  let size: UInt64 = 8 * 1024 * 1024
  for total in [UInt64(1), size, size + 1, size * 3] {
    let plan = try ChunkPlan(totalBytes: total, partSize: size)
    let chunks = try (1...plan.count).map { try plan.chunk($0) }
    #expect(chunks.reduce(UInt64(0)) { $0 + $1.bytes } == total)
    #expect(chunks.last!.offset + chunks.last!.bytes == total)
    #expect(chunks.first!.offset == 0)
  }
  #expect(throws: UploadError.self) { try ChunkPlan(totalBytes: 0, partSize: size) }
  #expect(throws: UploadError.self) { try ChunkPlan(totalBytes: size, partSize: 1) }
  #expect(throws: UploadError.self) { try ChunkPlan(totalBytes: UInt64.max, partSize: size) }
}

@Test func restartAndAcknowledgement() throws {
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: url) }
  var state = try UploadState(name: "Fixture", totalBytes: 16 * 1024 * 1024 + 1)
  state.partSize = 8 * 1024 * 1024
  try state.acknowledge(part: 2, etag: "two")
  try state.save(to: url)
  state = try UploadState.load(from: url)
  #expect(try state.nextPart() == 1)
  try state.acknowledge(part: 1, etag: "lost-response-retry")
  try state.acknowledge(part: 1, etag: "replacement")
  #expect(state.completed.count == 2)
  #expect(try state.nextPart() == 3)
  try state.acknowledge(part: 3, etag: "last")
  #expect(try state.nextPart() == nil)
  #expect(throws: UploadError.self) { try state.acknowledge(part: 4, etag: "bad") }
  #expect(throws: UploadError.self) { try state.acknowledge(part: 3, etag: "") }
}

@Test func chunkBodyContainsOnlyRequestedBytes() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let archive = root.appendingPathComponent("archive")
  let part = root.appendingPathComponent("part")
  try Data([0, 1, 2, 3, 4, 5]).write(to: archive)
  try BundleUploader.materialize(
    UploadChunk(partNumber: 1, offset: 2, bytes: 3), archive: archive, destination: part)
  #expect(try Data(contentsOf: part) == Data([2, 3, 4]))
  #expect(throws: UploadError.self) {
    try BundleUploader.materialize(
      UploadChunk(partNumber: 2, offset: 5, bytes: 2), archive: archive, destination: part)
  }
}

@Test func relaunchRestoresIdentityAndRejectsDifferentServer() async throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let archive = root.appendingPathComponent("source.hexcap")
  try Data([1, 2, 3]).write(to: archive)
  let folder = root.appendingPathComponent("transfer")
  let base = URL(string: "https://api.example.test/api/v1")!
  let uploader = try BundleUploader(
    directory: folder, archive: archive, apiBase: base, token: { "unused" })
  let before = await uploader.snapshot
  let resumed = try BundleUploader(directory: folder, apiBase: base, token: { "unused" })
  #expect(await resumed.snapshot == before)
  #expect(throws: UploadError.self) {
    try BundleUploader(
      directory: folder, apiBase: URL(string: "https://other.example.test/api/v1")!,
      token: { "unused" })
  }
  #expect(try Data(contentsOf: folder.appendingPathComponent("capture.hexcap")) == Data([1, 2, 3]))
}

@Test func placementRejectsInventedCoordinates() throws {
  #expect(throws: UploadError.self) {
    try BundlePlacement(latitude: 91, longitude: 0, uncertaintyM: 5)
  }
  #expect(throws: UploadError.self) {
    try BundlePlacement(latitude: 0, longitude: .nan, uncertaintyM: 5)
  }
  let placement = try BundlePlacement(latitude: 37, longitude: -122, uncertaintyM: 12)
  #expect(placement.lat == 37 && placement.lon == -122)
}

@Test func archiveAlreadyInTransferDirectoryIsPreserved() throws {
  let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: folder) }
  let archive = folder.appendingPathComponent("capture.hexcap")
  try Data([9, 8, 7]).write(to: archive)
  _ = try BundleUploader(
    directory: folder, archive: archive,
    apiBase: URL(string: "https://api.example.test/api/v1")!, token: { "unused" })
  #expect(try Data(contentsOf: archive) == Data([9, 8, 7]))
}
