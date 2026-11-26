import Foundation

/// Dataset policy, in metres, degrees and pixels. Targets are soft: never invent
/// views or points to fill a small scan. Pose bounds match capture_bundle.py.
public struct PrepConfig: Codable, Sendable, Equatable {
  public var keyframeTranslation = 0.04
  public var keyframeRotation = 4.0
  public var relaxedTranslation = 0.03
  public var relaxedRotation = 3.0
  public var minimumKeyframes = 120
  public var maximumKeyframes = 200
  public var blurMedianFraction = 0.25
  public var blurMaxSide = 160
  public var holdoutEvery = 8
  public var trainingMaxSide = 960
  public var voxelSize = 0.015
  public var minimumSeedPoints = 100_000
  public var maximumSeedPoints = 300_000
  public var depthEdgeMetres = 0.05
  public var depthEdgeFraction = 0.05
  public var minimumDepth = 0.1
  public var maximumDepth = 20.0
  public var rotationErrorMax = 1e-3
  public var homogeneousErrorMax = 1e-6
  public var svdEpsilon = 1e-15
  public var svdSweeps = 32
  public var jpegQuality = 0.9

  public init() {}

  public func validate() throws {
    guard (0.03...0.05).contains(keyframeTranslation), (3...5).contains(keyframeRotation),
      relaxedTranslation > 0, relaxedRotation > 0,
      minimumKeyframes > 1, maximumKeyframes >= minimumKeyframes,
      (0...1).contains(blurMedianFraction), blurMaxSide >= 3,
      holdoutEvery >= 2, trainingMaxSide >= 3,
      voxelSize.isFinite, voxelSize > 0, minimumSeedPoints > 0,
      maximumSeedPoints >= minimumSeedPoints,
      depthEdgeMetres > 0, depthEdgeFraction > 0,
      minimumDepth > 0, maximumDepth > minimumDepth,
      rotationErrorMax == 1e-3, homogeneousErrorMax == 1e-6,
      svdEpsilon > 0, svdSweeps > 0, (0...1).contains(jpegQuality)
    else { throw PrepError.invalid("Invalid preparation policy") }
  }
}

public enum PrepError: Error, Sendable, CustomStringConvertible {
  case invalid(String)
  public var description: String {
    switch self {
    case .invalid(let message): message
    }
  }
}

public struct CaptureFrame: Codable, Sendable {
  public struct Depth: Codable, Sendable {
    public let file: String
    public let confidence: String
    public let width: Int
    public let height: Int
  }
  public let index: Int
  public let timestamp: Double
  public let transform: [Double]
  public let intrinsics: [Double]
  public let imageWidth: Int
  public let imageHeight: Int
  public let trackingState: String
  public let depth: Depth?
  public let smoothedDepth: Depth?

  public static func read(from bundle: URL) throws -> [Self] {
    struct Manifest: Decodable {
      let version: Int
      let format: String
      let frameCount: Int
      let video: String
      let frameLog: String
    }
    let manifest = try JSONDecoder().decode(
      Manifest.self,
      from: Data(contentsOf: bundle.appendingPathComponent("manifest.json")))
    guard manifest.version == 1, manifest.format == "hex-capture",
      manifest.video == "rgb.mov", manifest.frameLog == "frames.jsonl"
    else {
      throw PrepError.invalid("Unsupported capture bundle")
    }
    let text = try String(
      contentsOf: bundle.appendingPathComponent("frames.jsonl"), encoding: .utf8)
    let frames = try text.split(separator: "\n").map {
      try JSONDecoder().decode(Self.self, from: Data($0.utf8))
    }
    guard !frames.isEmpty, frames.count == manifest.frameCount else {
      throw PrepError.invalid("Empty frame log")
    }
    for (position, frame) in frames.enumerated() {
      guard frame.index == position, frame.timestamp.isFinite,
        position == 0 || frame.timestamp > frames[position - 1].timestamp,
        frame.imageWidth > 0, frame.imageHeight > 0,
        frame.intrinsics.count == 9, frame.intrinsics.allSatisfy(\.isFinite),
        frame.intrinsics[0] > 0, frame.intrinsics[4] > 0,
        frame.transform.count == 16, frame.transform.allSatisfy(\.isFinite)
      else { throw PrepError.invalid("Invalid frame \(position)") }
    }
    return frames
  }

  public func scaledIntrinsics(width: Int, height: Int) -> [Double] {
    let sx = Double(width) / Double(imageWidth)
    let sy = Double(height) / Double(imageHeight)
    return [
      intrinsics[0] * sx, 0, intrinsics[2] * sx,
      0, intrinsics[4] * sy, intrinsics[5] * sy, 0, 0, 1,
    ]
  }
}
