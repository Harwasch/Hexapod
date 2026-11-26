import Foundation

public struct SeedPoint: Codable, Sendable, Equatable {
  public let x: Double
  public let y: Double
  public let z: Double
  public let rgb: [UInt8]
  public init(x: Double, y: Double, z: Double, rgb: [UInt8]) {
    self.x = x
    self.y = y
    self.z = z
    self.rgb = rgb
  }
}

public enum SeedPoints {
  public static func decodeDepth(_ data: Data, count: Int) throws -> [Double] {
    guard count > 0, data.count == count * 2 else { throw PrepError.invalid("Depth byte count") }
    let bytes = [UInt8](data)
    return (0..<count).map {
      Double(Float16(bitPattern: UInt16(bytes[2 * $0]) | UInt16(bytes[2 * $0 + 1]) << 8))
    }
  }

  /// Integer depth sample centres match the capture/server convention. RGB is
  /// tightly packed RGB8 at its stated resolution, sampled by normalized pixel.
  public static func backProject(
    depth: [Double], confidence: [UInt8], width: Int, height: Int,
    intrinsics: [Double], pose: [Double], rgb: [UInt8], rgbWidth: Int, rgbHeight: Int,
    config: PrepConfig = .init()
  ) throws -> [SeedPoint] {
    guard width >= 3, height >= 3, depth.count == width * height,
      confidence.count == depth.count, intrinsics.count == 9, pose.count == 16,
      rgbWidth > 0, rgbHeight > 0, rgb.count == rgbWidth * rgbHeight * 3,
      intrinsics.allSatisfy(\.isFinite), intrinsics[0] > 0, intrinsics[4] > 0,
      pose.allSatisfy(\.isFinite)
    else { throw PrepError.invalid("Seed raster/calibration mismatch") }
    var points: [SeedPoint] = []
    for v in 1..<(height - 1) {
      for u in 1..<(width - 1) {
        let i = v * width + u
        let z = depth[i]
        guard confidence[i] == 2, z.isFinite,
          z >= config.minimumDepth, z <= config.maximumDepth
        else { continue }
        let tolerance = max(config.depthEdgeMetres, z * config.depthEdgeFraction)
        guard
          [i - 1, i + 1, i - width, i + width].allSatisfy({
            depth[$0].isFinite && depth[$0] > 0 && abs(depth[$0] - z) <= tolerance
          })
        else { continue }
        let camera = [
          (Double(u) - intrinsics[2]) * z / intrinsics[0],
          -(Double(v) - intrinsics[5]) * z / intrinsics[4], -z,
        ]
        var world = [Double](repeating: 0, count: 3)
        for r in 0..<3 {
          world[r] = pose[r * 4 + 3]
          for c in 0..<3 { world[r] += pose[r * 4 + c] * camera[c] }
        }
        let x = min(rgbWidth - 1, u * rgbWidth / width)
        let y = min(rgbHeight - 1, v * rgbHeight / height)
        let c = (y * rgbWidth + x) * 3
        points.append(SeedPoint(x: world[0], y: world[1], z: world[2], rgb: Array(rgb[c..<c + 3])))
      }
    }
    return points
  }

  public static func voxelDownsample(_ points: [SeedPoint], size: Double) -> [SeedPoint] {
    var grid = VoxelSeeds(size: size, maximum: Int.max / 2)
    grid.add(points)
    return grid.points
  }
}

/// Bounded streaming voxel grid. First observation wins deterministically.
/// Coarsens when needed; sparse captures keep fewer points rather than padding.
public struct VoxelSeeds: Sendable {
  private struct Key: Hashable, Comparable, Sendable {
    let x: Int64, y: Int64, z: Int64
    static func < (a: Self, b: Self) -> Bool {
      if a.x != b.x { return a.x < b.x }
      if a.y != b.y { return a.y < b.y }
      return a.z < b.z
    }
  }
  private var cells: [Key: SeedPoint] = [:]
  public private(set) var size: Double
  private let maximum: Int
  public init(size: Double, maximum: Int) {
    self.size = size
    self.maximum = maximum
  }
  public var points: [SeedPoint] { cells.keys.sorted().compactMap { cells[$0] } }
  public mutating func add(_ points: [SeedPoint]) {
    guard size.isFinite, size > 0, maximum > 0 else { return }
    for point in points {
      insert(point)
      if cells.count > maximum * 2 { coarsen() }
    }
    while cells.count > maximum { coarsen() }
  }
  private mutating func insert(_ p: SeedPoint) {
    let values = [p.x / size, p.y / size, p.z / size].map { floor($0) }
    guard p.rgb.count == 3, values.allSatisfy({ $0.isFinite && abs($0) < Double(Int64.max) }) else {
      return
    }
    let key = Key(x: Int64(values[0]), y: Int64(values[1]), z: Int64(values[2]))
    if cells[key] == nil { cells[key] = p }
  }
  private mutating func coarsen() {
    let old = points
    cells.removeAll(keepingCapacity: true)
    size *= 2
    for point in old { insert(point) }
  }
}
