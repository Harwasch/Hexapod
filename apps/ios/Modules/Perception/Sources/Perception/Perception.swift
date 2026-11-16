import EntitySchema
import Foundation

public enum PerceptionError: Error, Sendable {
  case invalidGeometry, insufficientDepth, invalidModel, unsupportedModel, noSubject
}

public struct ImagePoint: Equatable, Sendable {
  /// Normalized in the unrotated camera image, top-left origin.
  public var x: Double
  public var y: Double
  public init(x: Double, y: Double) {
    self.x = x
    self.y = y
  }
}

public struct ImageBox: Equatable, Sendable {
  public var min: ImagePoint
  public var max: ImagePoint
  public init(min: ImagePoint, max: ImagePoint) {
    self.min = min
    self.max = max
  }
  public var area: Double { Swift.max(0, max.x - min.x) * Swift.max(0, max.y - min.y) }
  public func iou(_ other: Self) -> Double {
    let intersection =
      Swift.max(0, Swift.min(max.x, other.max.x) - Swift.max(min.x, other.min.x))
      * Swift.max(0, Swift.min(max.y, other.max.y) - Swift.max(min.y, other.min.y))
    let union = area + other.area - intersection
    return union > 0 ? intersection / union : 0
  }
  public func contains(_ point: ImagePoint) -> Bool {
    point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
  }
}

public enum SubjectPrompt: Equatable, Sendable {
  case point(ImagePoint)
  case box(ImageBox)
}

/// Packed RGBA8, in the same unrotated image coordinates as intrinsics and masks.
public struct CameraImage: Sendable {
  public var width: Int
  public var height: Int
  public var rgba: [UInt8]
  public init(width: Int, height: Int, rgba: [UInt8]) {
    self.width = width
    self.height = height
    self.rgba = rgba
  }
}

public struct SubjectMask: Equatable, Sendable {
  public var width: Int
  public var height: Int
  public var pixels: [UInt8]
  public init(width: Int, height: Int, pixels: [UInt8]) {
    self.width = width
    self.height = height
    self.pixels = pixels
  }
  public var isValid: Bool {
    width > 0 && height > 0 && pixels.count / width == height && pixels.count % width == 0
  }
  public var bounds: ImageBox? {
    guard isValid else { return nil }
    var x0 = width
    var y0 = height
    var x1 = -1
    var y1 = -1
    for y in 0..<height {
      for x in 0..<width where pixels[y * width + x] > 0 {
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
      }
    }
    guard x1 >= x0 else { return nil }
    return ImageBox(
      min: .init(x: Double(x0) / Double(width), y: Double(y0) / Double(height)),
      max: .init(x: Double(x1 + 1) / Double(width), y: Double(y1 + 1) / Double(height)))
  }
}

public protocol Segmenter: Sendable {
  func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask
}

public struct FallbackSegmenter: Segmenter {
  private let primary: (any Segmenter)?
  private let fallback: any Segmenter
  public init(primary: (any Segmenter)?, fallback: any Segmenter) {
    self.primary = primary
    self.fallback = fallback
  }
  public func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
    if let primary {
      do { return try await primary.segment(image, prompt: prompt) } catch is CancellationError {
        throw CancellationError()
      } catch {}
    }
    try Task.checkCancellation()
    return try await fallback.segment(image, prompt: prompt)
  }
}

public struct DepthImage: Sendable {
  public var width: Int
  public var height: Int
  public var metres: [Float]
  public init(width: Int, height: Int, metres: [Float]) {
    self.width = width
    self.height = height
    self.metres = metres
  }
}

/// Rigid camera-to-world transform, column-major (ARKit convention).
public struct CameraGeometry: Sendable {
  public var fx: Double, fy: Double, cx: Double, cy: Double
  public var imageWidth: Int, imageHeight: Int
  public var transform: [Double]
  public init(
    fx: Double, fy: Double, cx: Double, cy: Double,
    imageWidth: Int, imageHeight: Int, transform: [Double]
  ) {
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.imageWidth = imageWidth
    self.imageHeight = imageHeight
    self.transform = transform
  }
  public var isValid: Bool {
    transform.count == 16 && transform.allSatisfy(\.isFinite) && fx.isFinite && fy.isFinite
      && cx.isFinite && cy.isFinite && fx > 0 && fy > 0 && imageWidth > 0 && imageHeight > 0
  }
  public func worldPoint(pixel: ImagePoint, depth: Double) -> Vector3? {
    guard isValid, depth.isFinite, depth > 0 else { return nil }
    // Image Y points down, AR camera Y up, forward is -Z.
    let x = (pixel.x * Double(imageWidth) - cx) / fx * depth
    let y = -(pixel.y * Double(imageHeight) - cy) / fy * depth
    let z = -depth
    let t = transform
    return Vector3(
      x: t[0] * x + t[4] * y + t[8] * z + t[12],
      y: t[1] * x + t[5] * y + t[9] * z + t[13],
      z: t[2] * x + t[6] * y + t[10] * z + t[14])
  }
  public func project(_ point: Vector3) -> ImagePoint? {
    guard isValid else { return nil }
    let t = transform
    let dx = point.x - t[12]
    let dy = point.y - t[13]
    let dz = point.z - t[14]
    let x = t[0] * dx + t[1] * dy + t[2] * dz
    let y = t[4] * dx + t[5] * dy + t[6] * dz
    let z = t[8] * dx + t[9] * dy + t[10] * dz
    guard z < -0.001 else { return nil }
    return ImagePoint(
      x: (fx * x / -z + cx) / Double(imageWidth),
      y: (cy - fy * y / -z) / Double(imageHeight))
  }
  public func projectedBounds(_ entity: Entity) -> ImageBox? {
    var points: [ImagePoint] = []
    for x in [-0.5, 0.5] {
      for y in [-0.5, 0.5] {
        for z in [-0.5, 0.5] {
          let p = entity.anchor.world
          let e = entity.extent
          if let point = project(.init(x: p.x + x * e.x, y: p.y + y * e.y, z: p.z + z * e.z)) {
            points.append(point)
          }
        }
      }
    }
    guard let first = points.first else { return nil }
    return ImageBox(
      min: .init(x: points.map(\.x).min() ?? first.x, y: points.map(\.y).min() ?? first.y),
      max: .init(x: points.map(\.x).max() ?? first.x, y: points.map(\.y).max() ?? first.y))
  }
}

public enum MaskLifter {
  /// Independent depth samples, not replicated mask pixels, determine confidence.
  /// Depth MAD and local support reject isolated returns before measuring the visible bounds.
  public static func lift(
    mask: SubjectMask, depth: DepthImage, camera: CameraGeometry,
    label: String, model: String = "depth-mask"
  ) throws -> Entity {
    guard mask.isValid, depth.width > 0, depth.height > 0,
      depth.metres.count / depth.width == depth.height,
      depth.metres.count % depth.width == 0, camera.isValid
    else {
      throw PerceptionError.invalidGeometry
    }
    var samples: [(ImagePoint, Double)] = []
    var maskedDepth = [Double](repeating: .nan, count: depth.metres.count)
    for y in 0..<depth.height {
      for x in 0..<depth.width {
        let u = (Double(x) + 0.5) / Double(depth.width)
        let v = (Double(y) + 0.5) / Double(depth.height)
        let mx = min(mask.width - 1, Int(u * Double(mask.width)))
        let my = min(mask.height - 1, Int(v * Double(mask.height)))
        let d = Double(depth.metres[y * depth.width + x])
        if mask.pixels[my * mask.width + mx] > 0 && d.isFinite && d >= 0.1 && d <= 20 {
          samples.append((.init(x: u, y: v), d))
          maskedDepth[y * depth.width + x] = d
        }
      }
    }
    guard samples.count >= 6 else { throw PerceptionError.insufficientDepth }
    let median = percentile(samples.map { $0.1 }, 0.5)
    let mad = percentile(samples.map { abs($0.1 - median) }, 0.5)
    // Sensor floor is an explicit policy, not an arithmetic test tolerance.
    let band = max(0.08, 3 * 1.4826 * mad)
    let points = samples.filter { sample in
      if abs(sample.1 - median) <= band { return true }
      // A curved crown or a narrow trunk can be far from the global median.
      // Keep a locally supported surface; reject isolated returns, not whole limbs.
      let x = min(depth.width - 1, Int(sample.0.x * Double(depth.width)))
      let y = min(depth.height - 1, Int(sample.0.y * Double(depth.height)))
      let footprint = sample.1 * Double(camera.imageWidth) / (camera.fx * Double(depth.width))
      var neighbours = 0
      for dy in -1...1 {
        for dx in -1...1 where dx != 0 || dy != 0 {
          let nx = x + dx
          let ny = y + dy
          if nx >= 0, nx < depth.width, ny >= 0, ny < depth.height,
            abs(maskedDepth[ny * depth.width + nx] - sample.1) <= max(0.08, 2 * footprint)
          {
            neighbours += 1
          }
        }
      }
      return neighbours >= 2
    }.compactMap {
      camera.worldPoint(pixel: $0.0, depth: $0.1)
    }
    guard points.count >= 6 else { throw PerceptionError.insufficientDepth }
    let xs = points.map(\.x)
    let ys = points.map(\.y)
    let zs = points.map(\.z)
    return Entity(
      label: label,
      anchor: Anchor(
        world: .init(
          x: percentile(xs, 0.5),
          y: percentile(ys, 0.5), z: percentile(zs, 0.5))),
      extent: .init(
        x: (xs.max()! - xs.min()!),
        y: (ys.max()! - ys.min()!),
        z: (zs.max()! - zs.min()!)),
      masks: [], confidence: Double(points.count) / Double(depth.width * depth.height),
      attrs: ["measurement": .string("LiDAR supported visible extent")],
      source: EntitySource(module: "Perception", model: model))
  }
  private static func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    let position = Double(values.count - 1) * p
    let lower = Int(position)
    let upper = min(lower + 1, values.count - 1)
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
  }
}

public struct Detection: Sendable {
  public var entity: Entity
  public var bounds: ImageBox
  public init(entity: Entity, bounds: ImageBox) {
    self.entity = entity
    self.bounds = bounds
  }
}

public struct EntityTrack: Sendable {
  public var id: UUID
  public var entity: Entity
  public var bounds: ImageBox
  public var missedFrames: Int
}

/// Greedy global cost association, one detection per track, with world-distance gating.
/// Tracks age only on processed perception frames, not on skipped AR frames.
public struct EntityTracker: Sendable {
  public private(set) var tracks: [EntityTrack] = []
  public var maximumDistance: Double = 0.5
  public var maximumMissedFrames = 8
  public init() {}
  @discardableResult public mutating func update(_ detections: [Detection], camera: CameraGeometry)
    -> [EntityTrack]
  {
    var pairs: [(Int, Int, Double)] = []
    for i in tracks.indices {
      tracks[i].missedFrames += 1
      guard let projected = camera.projectedBounds(tracks[i].entity) else { continue }
      tracks[i].bounds = projected
      for j in detections.indices {
        let a = tracks[i].entity
        let b = detections[j].entity
        guard a.label.lowercased() == b.label.lowercased() else { continue }
        let p = a.anchor.world
        let q = b.anchor.world
        let d = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2))
        let iou = projected.iou(detections[j].bounds)
        if d <= maximumDistance && (iou >= 0.05 || d <= maximumDistance * 0.5) {
          pairs.append((i, j, d / maximumDistance + 1 - iou))
        }
      }
    }
    pairs.sort { $0.2 < $1.2 }
    var usedTracks: Set<Int> = []
    var usedDetections: Set<Int> = []
    for (i, j, _) in pairs where !usedTracks.contains(i) && !usedDetections.contains(j) {
      tracks[i].entity = detections[j].entity
      tracks[i].bounds = detections[j].bounds
      tracks[i].missedFrames = 0
      usedTracks.insert(i)
      usedDetections.insert(j)
    }
    tracks.removeAll { $0.missedFrames > maximumMissedFrames }
    for j in detections.indices where !usedDetections.contains(j) {
      tracks.append(
        EntityTrack(
          id: UUID(), entity: detections[j].entity,
          bounds: detections[j].bounds, missedFrames: 0))
    }
    return tracks
  }
}

public protocol InstanceSegmenting: Segmenter {
  func instances(_ image: CameraImage) async throws -> [SubjectMask]
}

public struct DeviceLabel: Sendable {
  public var label: String
  public var confidence: Double
  public init(label: String, confidence: Double) {
    self.label = label
    self.confidence = confidence
  }
}

public protocol ImageIdentifying: Sendable {
  func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel?
}

/// Masks outside the subject are made opaque black before on-device classification.
public enum SubjectImage {
  public static func isolated(_ image: CameraImage, mask: SubjectMask) throws -> CameraImage {
    guard image.width > 0, image.height > 0, image.rgba.count == image.width * image.height * 4,
      mask.isValid
    else { throw PerceptionError.invalidGeometry }
    var output = image
    for y in 0..<image.height {
      for x in 0..<image.width {
        let mx = x * mask.width / image.width
        let my = y * mask.height / image.height
        if mask.pixels[my * mask.width + mx] == 0 {
          let i = (y * image.width + x) * 4
          output.rgba[i] = 0
          output.rgba[i + 1] = 0
          output.rgba[i + 2] = 0
          output.rgba[i + 3] = 255
        }
      }
    }
    return output
  }
}
