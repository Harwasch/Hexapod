import Foundation

/// Bounded 10 cm direction grid. The CPU implementation is a test oracle only;
/// live frames use the identical four-way hash in the existing Metal command buffer.
public struct CoverageHash: Sendable {
  public static let capacity = 32_768
  public static let cellSize: Float = 0.1
  public static let sectorCount = 24
  public private(set) var cells: [Cell?]
  public struct Cell: Sendable {
    public var key: SIMD3<Int32>
    public var azimuth: UInt32
    public var elevation: UInt32
    public var observations: UInt32
    public var epoch: UInt32
  }
  public init(capacity: Int = Self.capacity) {
    precondition(capacity >= 4 && capacity % 4 == 0)
    cells = Array(repeating: nil, count: capacity)
  }
  public static func direction(_ offset: SIMD3<Float>) -> (azimuth: UInt32, elevation: UInt32) {
    let length = sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z)
    guard length.isFinite, length > 0 else { return (0, 0) }
    let angle = (atan2(offset.z, offset.x) + 2 * .pi).truncatingRemainder(dividingBy: 2 * .pi)
    let az = min(sectorCount - 1, Int(angle * Float(sectorCount) / (2 * .pi)))
    let elevation = asin(min(1, max(-1, offset.y / length))) * 180 / .pi
    let band = elevation < -10 ? 0 : elevation < 20 ? 1 : elevation < 45 ? 2 : 3
    return (1 << az, 1 << band)
  }
  public static func bucket(_ key: SIMD3<Int32>, count: Int) -> Int {
    Int(
      (UInt32(bitPattern: key.x) &* 73_856_093) ^ (UInt32(bitPattern: key.y) &* 19_349_663)
        ^ (UInt32(bitPattern: key.z) &* 83_492_791)) % count
  }
  public mutating func observe(position: SIMD3<Float>, camera: SIMD3<Float>, epoch: UInt32) {
    guard position.x.isFinite, position.y.isFinite, position.z.isFinite,
      max(abs(position.x), abs(position.y), abs(position.z)) < 1_000_000
    else { return }
    let key = SIMD3<Int32>(
      Int32(floor(position.x / Self.cellSize)),
      Int32(floor(position.y / Self.cellSize)), Int32(floor(position.z / Self.cellSize)))
    let start = Self.bucket(key, count: cells.count / 4) * 4
    var slot = start
    var found = false
    for i in start..<start + 4 {
      guard let cell = cells[i] else {
        slot = i
        break
      }
      if cell.key == key {
        slot = i
        found = true
        break
      }
      if cell.epoch < (cells[slot]?.epoch ?? .max) { slot = i }
    }
    if found && cells[slot]!.epoch == epoch { return }
    let center =
      SIMD3<Float>(Float(key.x) + 0.5, Float(key.y) + 0.5, Float(key.z) + 0.5) * Self.cellSize
    let masks = Self.direction(camera - center)
    var cell =
      found ? cells[slot]! : Cell(key: key, azimuth: 0, elevation: 0, observations: 0, epoch: 0)
    cell.azimuth |= masks.azimuth
    cell.elevation |= masks.elevation
    cell.observations = min(32, cell.observations + 1)
    cell.epoch = epoch
    cells[slot] = cell
  }
}
