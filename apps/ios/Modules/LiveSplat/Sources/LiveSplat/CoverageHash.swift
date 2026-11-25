import Foundation

public struct CoverageKey: Equatable, Sendable {
  public let xyz: SIMD3<Int32>
  public init(position: SIMD3<Float>) {
    xyz = SIMD3(
      Int32(floor(position.x / CoverageHash.cellSize)),
      Int32(floor(position.y / CoverageHash.cellSize)),
      Int32(floor(position.z / CoverageHash.cellSize)))
  }
  public func bucket(count: Int) -> Int {
    Int(
      (UInt32(bitPattern: xyz.x) &* 73_856_093
        ^ UInt32(bitPattern: xyz.y) &* 19_349_663
        ^ UInt32(bitPattern: xyz.z) &* 83_492_791) % UInt32(count))
  }
}

/// CPU oracle only. Live frames use CoverageMetalPass; no model readback.
/// Four-way buckets evict the least observed slot. All storage is allocated once.
public struct CoverageHash: Sendable {
  public static let capacity = 32_768
  public static let cellSize: Float = 0.1
  public static let maxDrawnHairs = 60_000
  public private(set) var cells: [CoverageGPUCell]
  public private(set) var keys: [CoverageKey?]
  public private(set) var positions: [SIMD3<Float>]
  private var firstDirections: [SIMD2<Int16>]
  private var firstMasks: [UInt16]
  public private(set) var count = 0
  public let parameters: CoverageCreditParameters
  public init(capacity: Int = Self.capacity, parameters: CoverageCreditParameters = .init()) {
    precondition(capacity >= 4 && capacity % 4 == 0)
    self.parameters = parameters
    cells = Array(repeating: CoverageGPUCell(), count: capacity)
    keys = Array(repeating: nil, count: capacity)
    positions = Array(repeating: .zero, count: capacity)
    firstDirections = Array(repeating: .zero, count: capacity * 14)
    firstMasks = Array(repeating: 0, count: capacity)
  }
  /// Fixed payload counts, not a process RSS measurement or Swift allocator overhead.
  public var allocatedSlots: Int {
    cells.count + keys.count + positions.count + firstDirections.count + firstMasks.count
  }
  @discardableResult public mutating func observe(
    position: SIMD3<Float>, normal: SIMD3<Float>, camera: SIMD3<Float>,
    focalPixels: Float, epoch: UInt16
  ) -> Int? {
    let delta = camera - position
    var distance = sqrt(CoverageCreditReference.dot(delta, delta))
    guard position.x.isFinite, position.y.isFinite, position.z.isFinite,
      max(abs(position.x), max(abs(position.y), abs(position.z))) < 1_000_000,
      normal.x.isFinite, normal.y.isFinite, normal.z.isFinite,
      CoverageCreditReference.dot(normal, normal) > 1e-12,
      distance.isFinite, distance > 0, focalPixels.isFinite, focalPixels > 0
    else { return nil }
    let key = CoverageKey(position: position)
    let start = key.bucket(count: cells.count / 4) * 4
    var slot = start
    for i in start..<start + 4 {
      if keys[i] == key {
        slot = i
        break
      }
      if keys[i] == nil {
        slot = i
        break
      }
      if cells[i].viewCount < cells[slot].viewCount { slot = i }
    }
    let found = keys[slot] == key
    if found && cells[slot].epoch == epoch { return slot }
    var view = delta / distance
    var n = CoverageCreditReference.unit(normal)
    if found {
      let previous = CoverageCreditReference.decodeNormal(cells[slot].normal)
      if CoverageCreditReference.dot(n, previous) < 0 { n = -n }
      n = CoverageCreditReference.unit(previous + (n - previous) * parameters.normalEMA)
      positions[slot] += (position - positions[slot]) * parameters.normalEMA
    } else {
      if CoverageCreditReference.dot(n, view) < 0 { n = -n }
      if keys[slot] == nil { count += 1 }
      keys[slot] = key
      positions[slot] = position
      cells[slot] = CoverageGPUCell()
      firstMasks[slot] = 0
    }
    var cell = cells[slot]
    cell.normal = CoverageCreditReference.encodeNormal(n)
    // Credit uses the same quantized mean normal as the GPU and the renderer.
    n = CoverageCreditReference.decodeNormal(cell.normal)
    let meanDelta = camera - positions[slot]
    distance = sqrt(CoverageCreditReference.dot(meanDelta, meanDelta))
    view = meanDelta / distance
    let incidence = CoverageCreditReference.dot(n, view)
    cell.viewCount = cell.viewCount == .max ? .max : cell.viewCount + 1
    cell.epoch = epoch
    if incidence > 0 {
      let mm = distance * 1000 / focalPixels / incidence
      cell.bestMmPerPx = cell.bestMmPerPx == 0 ? Float16(mm) : min(cell.bestMmPerPx, Float16(mm))
      for h in 0..<parameters.hairs {
        let cosine = CoverageCreditReference.dot(
          view, CoverageCreditReference.hair(h, count: parameters.hairs, normal: n))
        let credit = CoverageCreditReference.credit(
          cosine: cosine, mmPerPx: mm, parameters: parameters)
        guard credit > 0 else { continue }
        let bit = UInt16(1) << h
        if let separation = parameters.parallaxMinDeg {
          if firstMasks[slot] & bit == 0 {
            firstDirections[slot * 14 + h] = CoverageCreditReference.encodeNormal(view)
            firstMasks[slot] |= bit
          } else if CoverageCreditReference.dot(
            view, CoverageCreditReference.decodeNormal(firstDirections[slot * 14 + h]))
            <= cos(separation * .pi / 180)
          {
            cell.flags |= bit
          }
        }
        cell[h] = CoverageCreditReference.creditedByte(
          cell[h], credit: credit, viewsToFill: parameters.viewsToFill)
      }
    }
    cells[slot] = cell
    return slot
  }
}
