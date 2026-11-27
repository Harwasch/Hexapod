import Foundation

/// Standard 3DGS PLY reader. Retains covariance and opacity; display uses SH DC.
/// Higher SH coefficients remain in the original PLY used for sharing.
public struct SavedGaussian: Sendable {
  public let position: SIMD4<Float>
  public let color: SIMD4<Float>
  public let axisX: SIMD4<Float>
  public let axisY: SIMD4<Float>
  public let axisZ: SIMD4<Float>
}
public struct SavedSplat: Sendable {
  public let gaussians: [SavedGaussian]
  public let center: SIMD3<Float>
  public let radius: Float

  public init(data: Data) throws {
    let marker = Data("end_header\n".utf8)
    guard let range = data.range(of: marker), range.upperBound < 65_536,
      let header = String(data: data[..<range.upperBound], encoding: .utf8),
      header.hasPrefix("ply\nformat binary_little_endian 1.0\n")
    else { throw LiveSplatError.invalidGaussian }
    var names: [String] = []
    var count = 0
    var vertices = false
    for line in header.split(separator: "\n") {
      let parts = line.split(separator: " ")
      if parts.first == "element" {
        guard parts.count == 3 else { throw LiveSplatError.invalidGaussian }
        vertices = parts[1] == "vertex"
        if vertices { count = Int(parts[2]) ?? 0 }
      }
      if parts.first == "property", vertices {
        guard parts.count == 3, parts[1] == "float" else { throw LiveSplatError.invalidGaussian }
        names.append(String(parts[2]))
      }
    }
    let required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
      "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    var offsets: [Int] = []
    for name in required {
      guard let index = names.firstIndex(of: name) else { throw LiveSplatError.invalidGaussian }
      offsets.append(index * 4)
    }
    guard count > 0, count <= 2_000_000, names.count <= 128,
      Set(names).count == names.count else { throw LiveSplatError.invalidGaussian }
    let stride = names.count * 4
    let start = range.upperBound
    guard data.count - start >= count * stride else { throw LiveSplatError.invalidGaussian }
    var output: [SavedGaussian] = []
    output.reserveCapacity(count)
    var low = SIMD3<Float>(repeating: .infinity)
    var high = SIMD3<Float>(repeating: -.infinity)
    try data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
      for i in 0..<count {
        let base = start + i * stride
        var v: [Float] = []
        for offset in offsets {
          let bits = bytes.loadUnaligned(fromByteOffset: base + offset, as: UInt32.self)
          let value = Float(bitPattern: UInt32(littleEndian: bits))
          guard value.isFinite else { throw LiveSplatError.invalidGaussian }
          v.append(value)
        }
        let p = SIMD4(v[0], v[1], v[2], Float(1))
        for axis in 0..<3 { low[axis] = min(low[axis], p[axis]); high[axis] = max(high[axis], p[axis]) }
        var q = SIMD4(v[10], v[11], v[12], v[13]) // wxyz
        let length = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        guard length.isFinite, length > 0 else { throw LiveSplatError.invalidGaussian }
        q /= length
        let w = q.x; let x = q.y; let y = q.z; let z = q.w
        let sx = exp(v[7]); let sy = exp(v[8]); let sz = exp(v[9])
        guard sx.isFinite, sy.isFinite, sz.isFinite else { throw LiveSplatError.invalidGaussian }
        let a = SIMD4(1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y), Float(0)) * sx
        let b = SIMD4(2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x), Float(0)) * sy
        let c = SIMD4(2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y), Float(0)) * sz
        let alpha: Float = 1 / (1 + exp(-v[6]))
        let sh: Float = 0.2820948
        let color = SIMD4(max(0, min(1, 0.5 + sh*v[3])), max(0, min(1, 0.5 + sh*v[4])),
          max(0, min(1, 0.5 + sh*v[5])), alpha)
        output.append(SavedGaussian(position: p, color: color, axisX: a, axisY: b, axisZ: c))
      }
    }
    gaussians = output
    center = (low + high) / 2
    let extent = high - low
    let sceneRadius = sqrt(extent.x*extent.x + extent.y*extent.y + extent.z*extent.z) / 2
    guard sceneRadius.isFinite, center.x.isFinite, center.y.isFinite, center.z.isFinite else {
      throw LiveSplatError.invalidGaussian
    }
    radius = max(0.15, sceneRadius)
  }

  /// Far-to-near, stable ties. Caller runs this off MainActor, one request at a time.
  public func sortedIndices(viewRows: [Float]) -> Data {
    guard viewRows.count == 16 else { return Data() }
    var order: [(UInt32, Float)] = []
    order.reserveCapacity(gaussians.count)
    for i in gaussians.indices {
      let p = gaussians[i].position
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

/// Row-major OpenGL camera-to-world, compatible with msplat and ARKit.
public struct SplatOrbit: Sendable {
  public var yaw: Float = 0
  public var pitch: Float = 0.15
  public var distance: Float = 2
  public var center = SIMD3<Float>(repeating: 0)
  public init() {}
  public mutating func drag(x: Float, y: Float) {
    yaw -= x * 0.006
    pitch = max(-1.3, min(1.3, pitch + y * 0.006))
  }
  public mutating func zoom(factor: Float) {
    guard factor.isFinite, factor > 0 else { return }
    distance = max(0.1, min(100, distance / factor))
  }
  public var pose: [Float] {
    let back = SIMD3(sin(yaw)*cos(pitch), sin(pitch), cos(yaw)*cos(pitch))
    let right = SIMD3(cos(yaw), Float(0), -sin(yaw))
    let up = SIMD3(-sin(yaw)*sin(pitch), cos(pitch), -cos(yaw)*sin(pitch))
    let p = center + back * distance
    return [right.x, up.x, back.x, p.x, right.y, up.y, back.y, p.y,
      right.z, up.z, back.z, p.z, 0, 0, 0, 1]
  }
}
