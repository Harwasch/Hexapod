import Foundation

/// Sorting depends on the view's depth direction. Translation adds the same
/// depth offset to every point and cannot change order. Three degrees amortizes
/// slow orbit sorting without accumulating drift: compare to the last sorted view.
public enum SplatSortPolicy {
  public static func direction(viewRows: [Float]) -> SIMD3<Float>? {
    guard viewRows.count == 16 else { return nil }
    let direction = SIMD3(viewRows[8], viewRows[9], viewRows[10])
    let squared = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z
    guard squared.isFinite, squared > 0 else { return nil }
    return direction / sqrt(squared)
  }
  public static func shouldSort(previous: SIMD3<Float>?, viewRows: [Float]) -> Bool {
    guard let current = direction(viewRows: viewRows) else { return false }
    guard let previous else { return true }
    let dot = previous.x * current.x + previous.y * current.y + previous.z * current.z
    let threshold: Float = cos(3 * Float.pi / 180)
    return dot < threshold
  }
}

/// The viewer retains only positions for CPU sorting after uploading attributes.
/// At 16 bytes per point this replaces the 80-byte CPU Gaussian copy.
public struct SavedSplatSortData: Sendable {
  private let positions: [SIMD3<Float>]
  public let center: SIMD3<Float>
  public let radius: Float
  public init(scene: SavedSplat) {
    var positions: [SIMD3<Float>] = []
    positions.reserveCapacity(scene.gaussians.count)
    for gaussian in scene.gaussians {
      let point = gaussian.position
      positions.append(SIMD3(point.x, point.y, point.z))
    }
    self.positions = positions
    center = scene.center
    radius = scene.radius
  }
  public func sortedIndices(viewRows: [Float]) -> Data {
    guard viewRows.count == 16 else { return Data() }
    var order: [(UInt32, Float)] = []
    order.reserveCapacity(positions.count)
    for i in positions.indices {
      let p = positions[i]
      let z = p.x*viewRows[8] + p.y*viewRows[9] + p.z*viewRows[10] + viewRows[11]
      order.append((UInt32(i), z))
    }
    order.sort { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 < $1.1 }
    var result: [UInt32] = []
    result.reserveCapacity(order.count)
    for item in order { result.append(item.0) }
    return result.withUnsafeBytes { Data($0) }
  }
}
