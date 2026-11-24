import Foundation

/// Swift mirror of Shaders/LiveSplat.metal. Four-way buckets bound lookup/eviction work.
/// Positions use world metres; quantization levels are part of the key.
public struct SurfelKey: Equatable, Sendable {
  public let x: Int32, y: Int32, z: Int32, level: Int32
  public init(position: SIMD3<Float>, depth: Float) {
    level = depth < 3 ? 0 : 1
    let size: Float = level == 0 ? 0.02 : 0.04
    x = Int32(floor(position.x / size))
    y = Int32(floor(position.y / size))
    z = Int32(floor(position.z / size))
  }
  public func bucket(count: Int) -> Int {
    let hash =
      UInt32(bitPattern: x) &* 73_856_093
      ^ UInt32(bitPattern: y) &* 19_349_663
      ^ UInt32(bitPattern: z) &* 83_492_791 ^ UInt32(bitPattern: level)
    return Int(hash % UInt32(count))
  }
}
public struct Surfel: Sendable {
  public var key: SurfelKey
  public var position: SIMD3<Float>
  public var color: SIMD3<Float>
  public var radius: Float
  public var observations: UInt32
  public var epoch: UInt32
}
public struct SurfelHash: Sendable {
  public static let capacity = 262_144
  public static let sampleBudget = 12_288
  public private(set) var cells: [Surfel?]
  public private(set) var count = 0
  public init(capacity: Int = Self.capacity) {
    precondition(capacity >= 4 && capacity % 4 == 0)
    cells = Array(repeating: nil, count: capacity)
  }
  /// Radius overlaps both voxel spacing and depth sample spacing, including the sampling stride.
  public static func radius(depth: Float, focalPixels: Float, sampleStride: Float = 2) -> Float {
    max(0.85 * (depth < 3 ? 0.02 : 0.04), min(0.12, 0.85 * depth * sampleStride / focalPixels))
  }
  @discardableResult public mutating func fuse(
    position: SIMD3<Float>, color: SIMD3<Float>,
    depth: Float, confidence: UInt8, radius: Float, epoch: UInt32,
    allowMediumConfidence: Bool = false
  ) -> Bool {
    guard confidence == 2 || (allowMediumConfidence && confidence == 1), depth.isFinite,
      (0.1...8).contains(depth),
      position.x.isFinite, position.y.isFinite, position.z.isFinite,
      max(abs(position.x), max(abs(position.y), abs(position.z))) < 1_000_000,
      radius.isFinite, radius > 0
    else { return false }
    let key = SurfelKey(position: position, depth: depth)
    let start = key.bucket(count: cells.count / 4) * 4
    var victim = start
    for i in start..<start + 4 {
      guard var cell = cells[i] else {
        victim = i
        break
      }
      if cell.key == key {
        let weight: Float = 1 / Float(min(32, cell.observations + 1))
        cell.position += (position - cell.position) * weight
        cell.color += (color - cell.color) * weight
        cell.radius = max(cell.radius, radius)
        cell.observations = min(32, cell.observations + 1)
        cell.epoch = epoch
        cells[i] = cell
        return true
      }
      if let old = cells[victim],
        cell.epoch < old.epoch
          || (cell.epoch == old.epoch && cell.observations < old.observations)
      {
        victim = i
      }
    }
    if cells[victim] == nil { count += 1 }
    cells[victim] = Surfel(
      key: key, position: position, color: color, radius: radius, observations: 1, epoch: epoch)
    return true
  }
}

/// Feedback target, never a permanent fallback. Three expensive frames reduce quality;
/// 120 inexpensive frames recover one level. Thermal/power state is a reversible floor.
public struct PreviewQuality: Sendable {
  public private(set) var level = 0
  private var slow = 0, fast = 0
  public init() {}
  public mutating func record(seconds: Double, constrained: Bool) {
    if !seconds.isFinite || seconds <= 0 || seconds > 0.008 {
      slow += 1
      fast = 0
      if slow >= 3 {
        level = min(2, level + 1)
        slow = 0
      }
    } else {
      slow = 0
      fast += 1
      if fast >= 120 {
        level = max(constrained ? 1 : 0, level - 1)
        fast = 0
      }
    }
    if constrained { level = max(1, level) }
  }
  public var fusionInterval: Double { [0.1, 0.2, 0.4][level] }
  public var visibleStride: Int { 1 << level }
  public var longestEdge: Int { [960, 720, 480][level] }
}

/// The GPU samples one rotating pixel per stride cell, never more than the dispatch cap.
public struct SurfelSampling: Sendable {
  public let stride: Int, count: Int
  private let width: Int, height: Int
  public init(width: Int, height: Int) {
    precondition(width > 0 && height > 0 && width <= 512 && height <= 512)
    self.width = width
    self.height = height
    stride = max(2, Int(ceil(sqrt(Double(width * height) / Double(SurfelHash.sampleBudget)))))
    count = min(
      SurfelHash.sampleBudget, ((width + stride - 1) / stride) * ((height + stride - 1) / stride))
  }
  public func pixel(sample: Int, epoch: UInt32) -> SIMD2<Int>? {
    guard (0..<count).contains(sample) else { return nil }
    let columns = (width + stride - 1) / stride
    let phase = Int(epoch) % (stride * stride)
    let x = sample % columns * stride + phase % stride
    let y = sample / columns * stride + phase / stride
    return x < width && y < height ? SIMD2(x, y) : nil
  }
}
