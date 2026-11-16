import Foundation

public enum CaptureError: Error {
  case invalidFrame, invalidFile, closed, incomplete, unsupported, recordingFailed
}

public struct DepthReference: Codable, Equatable, Sendable {
  public var file: String
  public var confidence: String
  public var width: Int
  public var height: Int

  public init(file: String, confidence: String, width: Int, height: Int) {
    self.file = file
    self.confidence = confidence
    self.width = width
    self.height = height
  }
}

public struct CaptureFrame: Codable, Equatable, Sendable {
  public var index: Int
  /// Seconds since the first recorded ARFrame; also the HEVC presentation timestamp.
  public var timestamp: Double
  /// Row-major, camera-to-world, metres, ARKit axes.
  public var transform: [Double]
  /// Row-major, in unrotated capturedImage pixel coordinates.
  public var intrinsics: [Double]
  public var imageWidth: Int
  public var imageHeight: Int
  public var trackingState: String
  public var depth: DepthReference?
  public var smoothedDepth: DepthReference?

  public init(
    index: Int, timestamp: Double, transform: [Double], intrinsics: [Double],
    imageWidth: Int, imageHeight: Int, trackingState: String,
    depth: DepthReference? = nil, smoothedDepth: DepthReference? = nil
  ) {
    self.index = index
    self.timestamp = timestamp
    self.transform = transform
    self.intrinsics = intrinsics
    self.imageWidth = imageWidth
    self.imageHeight = imageHeight
    self.trackingState = trackingState
    self.depth = depth
    self.smoothedDepth = smoothedDepth
  }
}

public struct BundleFile: Codable, Equatable, Sendable {
  public var path: String
  public var bytes: UInt64
}

public struct CaptureManifest: Codable, Equatable, Sendable {
  public var version: Int = 1
  public var format: String = "hex-capture"
  public var device: String
  /// UTC ISO 8601, independent of ARKit's monotonic clock.
  public var startedAt: String
  public var frameCount: Int
  public var depthFrameCount: Int
  public var smoothedDepthFrameCount: Int
  public var video: String = "rgb.mov"
  public var frameLog: String = "frames.jsonl"
  public var locationLog: String?
  public var location: CaptureLocation?
  /// Every payload file, excluding manifest.json itself.
  public var files: [BundleFile]
}

/// Serial use only. A manifest is published only after video finalization succeeds.
public final class CaptureBundleWriter {
  public let directory: URL
  private let device: String
  private let startedAt: String
  private let log: FileHandle
  private var locationLog: FileHandle?
  private var location: CaptureLocation?
  private var count = 0
  private var depthCount = 0
  private var smoothedCount = 0
  private var lastTimestamp: Double = -1
  private var closed = false

  public init(directory: URL, device: String, startedAt: Date = Date()) throws {
    guard !FileManager.default.fileExists(atPath: directory.path) else {
      throw CaptureError.invalidFile
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    self.directory = directory
    self.device = device
    self.startedAt = ISO8601DateFormatter().string(from: startedAt)
    let url = directory.appendingPathComponent("frames.jsonl")
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
      throw CaptureError.invalidFile
    }
    log = try FileHandle(forWritingTo: url)
  }

  deinit {
    try? log.close()
    try? locationLog?.close()
  }

  public func appendLocation(_ fix: CaptureLocation) throws {
    guard !closed else { throw CaptureError.closed }
    guard fix.isValid else { throw CaptureError.invalidFrame }
    if locationLog == nil {
      let url = directory.appendingPathComponent("location.jsonl")
      guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
        throw CaptureError.invalidFile
      }
      locationLog = try FileHandle(forWritingTo: url)
    }
    var data = try JSONEncoder().encode(fix)
    data.append(10)
    try locationLog!.write(contentsOf: data)
    if fix.improves(location) { location = fix }
  }

  public func append(_ frame: CaptureFrame) throws {
    guard !closed else { throw CaptureError.closed }
    guard frame.index == count, frame.timestamp.isFinite, frame.timestamp > lastTimestamp,
      count > 0 || frame.timestamp == 0, frame.transform.count == 16,
      frame.intrinsics.count == 9, frame.transform.allSatisfy(\.isFinite),
      frame.intrinsics.allSatisfy(\.isFinite), frame.imageWidth > 0, frame.imageHeight > 0
    else { throw CaptureError.invalidFrame }
    for reference in [frame.depth, frame.smoothedDepth].compactMap({ $0 }) {
      guard reference.width > 0, reference.height > 0 else { throw CaptureError.invalidFrame }
      try validateFile(
        reference.file, bytes: UInt64(reference.width) * UInt64(reference.height) * 2)
      try validateFile(
        reference.confidence, bytes: UInt64(reference.width) * UInt64(reference.height))
    }
    var data = try JSONEncoder().encode(frame)
    data.append(10)
    try log.write(contentsOf: data)
    count += 1
    lastTimestamp = frame.timestamp
    if frame.depth != nil { depthCount += 1 }
    if frame.smoothedDepth != nil { smoothedCount += 1 }
  }

  private func validateFile(_ name: String, bytes: UInt64) throws {
    guard Self.safeName(name),
      let size = try FileManager.default.attributesOfItem(
        atPath: directory.appendingPathComponent(name).path)[.size] as? NSNumber,
      size.uint64Value == bytes
    else { throw CaptureError.invalidFile }
  }

  public static func safeName(_ name: String) -> Bool {
    !name.isEmpty && name.utf8.count < 100 && name != "." && name != ".."
      && name.unicodeScalars.allSatisfy {
        CharacterSet(
          charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        ).contains($0)
      }
  }

  public func finish() throws -> CaptureManifest {
    guard !closed, count > 0 else { throw CaptureError.incomplete }
    try locationLog?.synchronize()
    try locationLog?.close()
    try log.synchronize()
    try log.close()
    closed = true
    let urls = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey])
    let files = try urls.map { url -> BundleFile in
      let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
      guard Self.safeName(url.lastPathComponent), values.isRegularFile == true,
        values.isSymbolicLink != true
      else { throw CaptureError.invalidFile }
      let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! NSNumber
      return BundleFile(path: url.lastPathComponent, bytes: size.uint64Value)
    }.sorted { $0.path < $1.path }
    guard files.contains(where: { $0.path == "rgb.mov" && $0.bytes > 0 }) else {
      throw CaptureError.incomplete
    }
    let manifest = CaptureManifest(
      device: device, startedAt: startedAt, frameCount: count,
      depthFrameCount: depthCount, smoothedDepthFrameCount: smoothedCount,
      locationLog: locationLog == nil ? nil : "location.jsonl", location: location, files: files)
    try JSONEncoder().encode(manifest).write(
      to: directory.appendingPathComponent("manifest.json"), options: .atomic)
    return manifest
  }
}
