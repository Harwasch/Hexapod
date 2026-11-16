import Foundation

public enum UploadError: Error {
  case invalidPlan, invalidState, changedArchive
  case http(Int)
  case missingETag, invalidURL
}

/// Optional existing catalog placement. ARKit's local origin is never treated as GPS.
public struct BundlePlacement: Codable, Equatable, Sendable {
  public let lat: Double
  public let lon: Double
  public let uncertaintyM: Double

  public init(latitude: Double, longitude: Double, uncertaintyM: Double) throws {
    guard latitude.isFinite, longitude.isFinite, uncertaintyM.isFinite,
      (-90...90).contains(latitude), (-180...180).contains(longitude), uncertaintyM > 0
    else { throw UploadError.invalidState }
    lat = latitude
    lon = longitude
    self.uncertaintyM = uncertaintyM
  }
}

public struct UploadChunk: Codable, Equatable, Sendable {
  public let partNumber: Int
  public let offset: UInt64
  public let bytes: UInt64
}

public struct ChunkPlan: Sendable {
  public let totalBytes: UInt64
  public let partSize: UInt64
  public let count: Int

  public init(totalBytes: UInt64, partSize: UInt64) throws {
    guard totalBytes > 0, partSize >= 5 * 1024 * 1024 else { throw UploadError.invalidPlan }
    let count = 1 + (totalBytes - 1) / partSize
    guard count <= 10_000 else { throw UploadError.invalidPlan }
    self.totalBytes = totalBytes
    self.partSize = partSize
    self.count = Int(count)
  }

  public func chunk(_ part: Int) throws -> UploadChunk {
    guard (1...count).contains(part) else { throw UploadError.invalidPlan }
    let offset = UInt64(part - 1) * partSize
    return UploadChunk(partNumber: part, offset: offset, bytes: min(partSize, totalBytes - offset))
  }
}

public struct CompletedPart: Codable, Equatable, Sendable {
  public var partNumber: Int
  public var etag: String
}

public struct UploadState: Codable, Equatable, Sendable {
  public var version = 1
  public var bundleID: UUID
  public var name: String
  public var totalBytes: UInt64
  public var placement: BundlePlacement?
  public var siteID: UUID?
  public var apiBase: String?
  public var captureID: String?
  public var fileID: String?
  public var partSize: UInt64?
  public var completed: [CompletedPart] = []
  public var jobID: String?
  public var lastError: String?

  public init(bundleID: UUID = UUID(), name: String, totalBytes: UInt64) throws {
    guard totalBytes > 0, !name.isEmpty else { throw UploadError.invalidState }
    self.bundleID = bundleID
    self.name = name
    self.totalBytes = totalBytes
  }

  public mutating func acknowledge(part: Int, etag: String) throws {
    guard let partSize, !etag.isEmpty, etag.count <= 128 else { throw UploadError.invalidState }
    _ = try ChunkPlan(totalBytes: totalBytes, partSize: partSize).chunk(part)
    completed.removeAll { $0.partNumber == part }
    completed.append(CompletedPart(partNumber: part, etag: etag))
    completed.sort { $0.partNumber < $1.partNumber }
  }

  public func nextPart() throws -> Int? {
    guard let partSize else { throw UploadError.invalidState }
    let plan = try ChunkPlan(totalBytes: totalBytes, partSize: partSize)
    let done = Set(completed.map(\.partNumber))
    guard done.count == completed.count,
      completed.allSatisfy({ (1...plan.count).contains($0.partNumber) && !$0.etag.isEmpty })
    else { throw UploadError.invalidState }
    return (1...plan.count).first { !done.contains($0) }
  }

  public func save(to url: URL) throws {
    try JSONEncoder().encode(self).write(to: url, options: .atomic)
  }

  public static func load(from url: URL) throws -> UploadState {
    let state = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    guard state.version == 1, state.totalBytes > 0 else { throw UploadError.invalidState }
    if state.partSize != nil { _ = try state.nextPart() }
    return state
  }
}
