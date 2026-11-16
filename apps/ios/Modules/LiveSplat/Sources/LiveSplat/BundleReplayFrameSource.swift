import Foundation

public enum BundleReplayError: Error, Equatable {
  case invalidRecord(Int)
  case invalidPayload(String)
  case unsafePath(String)
}

/// Replays a fixture's inner bundle/ folder. Depth/confidence are loaded one frame
/// at a time; poses retain the row-major ARKit convention. Limited tracking emits
/// a loss event without decoding or seeding that frame. No ARKit dependency.
@MainActor public final class BundleReplayFrameSource: SplatFrameSource {
  private struct Record: Decodable {
    struct Depth: Decodable {
      let file: String
      let confidence: String
      let width: Int
      let height: Int
    }
    let index: Int
    let timestamp: Double
    let transform: [Double]
    let intrinsics: [Double]
    let imageWidth: Int
    let imageHeight: Int
    let trackingState: String
    let depth: Depth?
  }

  private let directory: URL
  private let records: [Record]
  private let decodeRGB: Bool
  private var cursor = 0
  public var frameCount: Int { records.count }
  public var consumedFrameCount: Int { cursor }
  public static let flatColor = SplatVector(0.5, 0.5, 0.5)

  /// On Apple, decodeRGB uses AVFoundation; false forces the same flat colour as
  /// Linux. A requested Apple decode failure throws rather than hiding bad video.
  public init(bundleDirectory: URL, decodeRGB: Bool = true) throws {
    directory = bundleDirectory.standardizedFileURL.resolvingSymlinksInPath()
    self.decodeRGB = decodeRGB
    let log = try String(contentsOf: Self.file("frames.jsonl", in: directory), encoding: .utf8)
    let decoder = JSONDecoder()
    records = try log.split(whereSeparator: \.isNewline).enumerated().map { line, json in
      let record = try decoder.decode(Record.self, from: Data(json.utf8))
      guard record.index == line, record.timestamp.isFinite,
        record.transform.count == 16, record.transform.allSatisfy(\.isFinite),
        Array(record.transform[12..<16]) == [0, 0, 0, 1],
        record.intrinsics.count == 9, record.intrinsics.allSatisfy(\.isFinite),
        record.intrinsics[1] == 0, record.intrinsics[3] == 0,
        Array(record.intrinsics[6..<9]) == [0, 0, 1],
        record.imageWidth > 0, record.imageHeight > 0,
        record.imageWidth <= 8192, record.imageHeight <= 8192
      else { throw BundleReplayError.invalidRecord(line) }
      return record
    }
    for i in records.indices.dropFirst() {
      guard records[i].timestamp > records[i - 1].timestamp else {
        throw BundleReplayError.invalidRecord(i)
      }
    }
  }

  public func reset() { cursor = 0 }

  public func next() async throws -> SplatFrameEvent? {
    guard cursor < records.count else { return nil }
    let record = records[cursor]
    guard record.trackingState == "normal", let depth = record.depth else {
      cursor += 1
      return .trackingUnavailable
    }
    guard depth.width > 0, depth.height > 0, depth.width <= 512, depth.height <= 512 else {
      throw BundleReplayError.invalidRecord(record.index)
    }
    let t = record.transform
    let k = record.intrinsics
    let camera = try SplatCamera(
      cameraToWorld: SplatMatrix([t[0], t[1], t[2], t[4], t[5], t[6], t[8], t[9], t[10]]),
      position: SplatVector(t[3], t[7], t[11]),
      fx: k[0], fy: k[4], cx: k[2], cy: k[5],
      width: record.imageWidth, height: record.imageHeight)
    let count = depth.width * depth.height
    let raw = try Data(contentsOf: Self.file(depth.file, in: directory))
    let confidence = try Data(contentsOf: Self.file(depth.confidence, in: directory))
    guard raw.count == 2 * count else { throw BundleReplayError.invalidPayload(depth.file) }
    guard confidence.count == count, confidence.allSatisfy({ $0 <= 2 }) else {
      throw BundleReplayError.invalidPayload(depth.confidence)
    }
    let depths = stride(from: 0, to: raw.count, by: 2).map {
      Float(Float16(bitPattern: UInt16(raw[$0]) | UInt16(raw[$0 + 1]) << 8))
    }
    var colors = Array(repeating: Self.flatColor, count: count)
    #if canImport(AVFoundation) && canImport(CoreGraphics)
      if decodeRGB {
        colors = try await ReplayRGB.colors(
          video: Self.file("rgb.mov", in: directory),
          seconds: record.timestamp - records[0].timestamp,
          imageWidth: record.imageWidth, imageHeight: record.imageHeight,
          width: depth.width, height: depth.height)
      }
    #endif
    let frame = try SplatFrame(
      timestamp: record.timestamp, camera: camera, width: depth.width, height: depth.height,
      depths: depths, confidence: Array(confidence), colors: colors)
    cursor += 1
    return .frame(frame)
  }

  private static func file(_ name: String, in directory: URL) throws -> URL {
    guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\\")
    else { throw BundleReplayError.unsafePath(name) }
    let url = directory.appendingPathComponent(name).resolvingSymlinksInPath()
    guard url.deletingLastPathComponent().path == directory.path else {
      throw BundleReplayError.unsafePath(name)
    }
    return url
  }
}
