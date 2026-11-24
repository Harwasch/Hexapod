import Foundation

/// Portable reference for the live surface Gaussian. The two tangent axes share
/// a scale; the normal axis is thinner. No photometric refinement is scheduled:
/// its benefit and a safe device budget have not been measured.
public enum GaussianSeed {
  public static let drawCapacity = 32_768
  public static let attributeBytes = SurfelHash.capacity * 16
  public static let sortBytes = drawCapacity * 8

  public static func make(
    position: SplatVector, normal: SplatVector, depth: Double,
    focalPixels: Double, stride: Double = 2, confidence: UInt8, color: SplatVector
  ) -> Gaussian? {
    guard depth.isFinite, (0.1...8).contains(depth), focalPixels.isFinite,
      focalPixels > 0, Float(focalPixels).isFinite, Float(focalPixels) > 0,
      stride.isFinite, stride > 0, Float(stride).isFinite, confidence == 1 || confidence == 2,
      normal.isFinite, normal.squaredLength.isFinite, normal.squaredLength > 1e-12
    else { return nil }
    let n = normal * (1 / sqrt(normal.squaredLength))
    // Rotate local +Z onto the surface normal (antipodal case has a fixed axis).
    let rotation = n.z < -0.999999 ? [0.0, 1, 0, 0] : [1 + n.z, -n.y, n.x, 0]
    let sigma =
      Double(
        SurfelHash.radius(
          depth: Float(depth), focalPixels: Float(focalPixels), sampleStride: Float(stride))) * 0.5
    return try? Gaussian(
      position: position, scale: SplatVector(sigma, sigma, max(0.001, sigma * 0.2)),
      rotation: rotation, opacity: confidence == 2 ? 0.85 : 0.45, color: color)
  }
}

/// CPU mirror of the GPU's descending bitonic network; ties use cell index so
/// translucent overlap is deterministic. Invalid/behind-camera entries sort last.
public enum GaussianSort {
  public struct Entry: Sendable, Equatable {
    public let depth: Float
    public let index: UInt32
    public init(depth: Float, index: UInt32) {
      self.depth = depth.isFinite && depth > 0.05 && depth <= 100 ? depth : -.infinity
      self.index = index
    }
  }
  public static func precedes(_ a: Entry, _ b: Entry) -> Bool {
    a.depth > b.depth || (a.depth == b.depth && a.index < b.index)
  }
  public static func sort(_ entries: inout [Entry]) {
    precondition(!entries.isEmpty && entries.count.nonzeroBitCount == 1)
    var width = 2
    while width <= entries.count {
      var distance = width / 2
      while distance > 0 {
        for i in entries.indices {
          let j = i ^ distance
          if j > i && (precedes(entries[j], entries[i]) == ((i & width) == 0)) {
            entries.swapAt(i, j)
          }
        }
        distance /= 2
      }
      width *= 2
    }
  }
}

extension GaussianSeed {
  /// Same adjacent-depth normal estimate and discontinuity gate as the shader.
  public static func surface(frame: SplatFrame, x: Int, y: Int, stride: Double = 2) -> Gaussian? {
    guard (0..<frame.width).contains(x), (0..<frame.height).contains(y) else { return nil }
    let index = y * frame.width + x
    let depth = Double(frame.depths[index])
    func point(_ px: Int, _ py: Int) -> SplatVector {
      frame.camera.unproject(
        u: Double(px) * Double(frame.camera.width) / Double(frame.width),
        v: Double(py) * Double(frame.camera.height) / Double(frame.height),
        depth: Double(frame.depths[py * frame.width + px]))
    }
    let p = point(x, y)
    var normal = frame.camera.position - p
    if x + 1 < frame.width, y + 1 < frame.height {
      let ix = index + 1
      let iy = index + frame.width
      if frame.confidence[ix] == 2, frame.confidence[iy] == 2,
        (0.1...8).contains(frame.depths[ix]), (0.1...8).contains(frame.depths[iy]),
        abs(Double(frame.depths[ix]) - depth) < depth * 0.05,
        abs(Double(frame.depths[iy]) - depth) < depth * 0.05
      {
        let a = point(x + 1, y) - p
        let b = point(x, y + 1) - p
        let n = SplatVector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
        if n.squaredLength > 1e-12 { normal = n }
      }
    }
    return make(
      position: p, normal: normal, depth: depth,
      focalPixels: frame.camera.fx * Double(frame.width) / Double(frame.camera.width),
      stride: stride, confidence: frame.confidence[index], color: frame.colors[index])
  }
}

/// Offline mirror only. Live ingress never allocates CPU geometry. The same
/// four-way SurfelHash chooses voxel matches and bounded eviction; attributes
/// occupy a fixed parallel array and are overwritten with the winning seed.
public struct GaussianVoxelHash: Sendable {
  public private(set) var voxels: SurfelHash
  public private(set) var seeds: [Gaussian?]
  public init(capacity: Int = SurfelHash.capacity) {
    let bounded = max(4, min(SurfelHash.capacity, capacity) / 4 * 4)
    voxels = SurfelHash(capacity: bounded)
    seeds = Array(repeating: nil, count: bounded)
  }
  @discardableResult public mutating func fuse(
    _ seed: Gaussian, depth: Float, confidence: UInt8, epoch: UInt32
  ) -> Bool {
    let p = seed.position
    let c = seed.color
    let position = SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z))
    guard
      voxels.fuse(
        position: position, color: SIMD3(Float(c.x), Float(c.y), Float(c.z)), depth: depth,
        confidence: confidence, radius: Float(exp(seed.parameters[3]) * 2), epoch: epoch,
        allowMediumConfidence: true)
    else { return false }
    let key = SurfelKey(position: position, depth: depth)
    let start = key.bucket(count: seeds.count / 4) * 4
    for i in start..<start + 4 where voxels.cells[i]?.key == key {
      seeds[i] = seed
      return true
    }
    return false
  }
  public func gaussian(at index: Int) -> Gaussian? {
    guard let cell = voxels.cells[index], var g = seeds[index] else { return nil }
    g.parameters[0] = Double(cell.position.x)
    g.parameters[1] = Double(cell.position.y)
    g.parameters[2] = Double(cell.position.z)
    let sigma = Double(cell.radius) * 0.5
    g.parameters[3] = log(sigma)
    g.parameters[4] = log(sigma)
    g.parameters[5] = log(max(0.001, sigma * 0.2))
    g.parameters[11] = Double(cell.color.x)
    g.parameters[12] = Double(cell.color.y)
    g.parameters[13] = Double(cell.color.z)
    return g
  }
}
