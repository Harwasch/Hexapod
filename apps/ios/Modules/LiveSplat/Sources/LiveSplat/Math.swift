import Foundation

public struct SplatVector: Sendable, Equatable {
  public var x: Double
  public var y: Double
  public var z: Double
  public init(_ x: Double, _ y: Double, _ z: Double) {
    self.x = x
    self.y = y
    self.z = z
  }
  public static let zero = Self(0, 0, 0)
  public static func + (a: Self, b: Self) -> Self { Self(a.x + b.x, a.y + b.y, a.z + b.z) }
  public static func - (a: Self, b: Self) -> Self { Self(a.x - b.x, a.y - b.y, a.z - b.z) }
  public static func * (a: Self, b: Double) -> Self { Self(a.x * b, a.y * b, a.z * b) }
  public var squaredLength: Double { x * x + y * y + z * z }
  public var isFinite: Bool { x.isFinite && y.isFinite && z.isFinite }
  public var values: [Double] { [x, y, z] }
}

/// Row-major storage, independent of Apple's column-major SIMD memory layout.
public struct SplatMatrix: Sendable, Equatable {
  public let values: [Double]
  public init(_ values: [Double]) {
    precondition(values.count == 9)
    self.values = values
  }
  public static let identity = Self([1, 0, 0, 0, 1, 0, 0, 0, 1])
  public subscript(_ r: Int, _ c: Int) -> Double { values[3 * r + c] }
  public var transposed: Self { Self((0..<9).map { self[$0 % 3, $0 / 3] }) }
  public static func * (a: Self, b: Self) -> Self {
    Self((0..<9).map { i in (0..<3).reduce(0) { $0 + a[i / 3, $1] * b[$1, i % 3] } })
  }
  public static func * (a: Self, b: SplatVector) -> SplatVector {
    let v = b.values
    let out = (0..<3).map { r in (0..<3).reduce(0.0) { $0 + a[r, $1] * v[$1] } }
    return SplatVector(out[0], out[1], out[2])
  }
}

public enum LiveSplatError: Error {
  case invalidFrame, invalidGaussian, exportWhileRunning, bundleFinalized, unavailable
}

public struct SplatCamera: Sendable {
  public let cameraToWorld: SplatMatrix
  public let position: SplatVector
  public let fx: Double, fy: Double, cx: Double, cy: Double
  public let width: Int, height: Int
  public init(
    cameraToWorld: SplatMatrix = .identity, position: SplatVector = .zero,
    fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int
  ) throws {
    let r = cameraToWorld
    let gram = r.transposed * r
    let determinant =
      r[0, 0] * (r[1, 1] * r[2, 2] - r[1, 2] * r[2, 1])
      - r[0, 1] * (r[1, 0] * r[2, 2] - r[1, 2] * r[2, 0]) + r[0, 2]
      * (r[1, 0] * r[2, 1] - r[1, 1] * r[2, 0])
    // 1e-5 is the capture format's float32 rigid-pose acceptance bound.
    guard r.values.allSatisfy(\.isFinite), position.isFinite,
      zip(gram.values, SplatMatrix.identity.values).allSatisfy({ abs($0 - $1) < 1e-5 }),
      abs(determinant - 1) < 1e-5, [fx, fy, cx, cy].allSatisfy(\.isFinite),
      fx > 0, fy > 0, width > 0, height > 0, width <= 8192, height <= 8192
    else {
      throw LiveSplatError.invalidFrame
    }
    self.cameraToWorld = r
    self.position = position
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.width = width
    self.height = height
  }
  /// World -> pinhole camera: diag(1,-1,-1) R_cw^T, applied after inversion.
  public var worldToOptical: SplatMatrix {
    SplatMatrix([1, 0, 0, 0, -1, 0, 0, 0, -1]) * cameraToWorld.transposed
  }
  public func project(_ world: SplatVector) -> SplatVector? {
    let p = worldToOptical * (world - position)
    guard p.isFinite, p.z > 0.01 else { return nil }
    return SplatVector(fx * p.x / p.z + cx, fy * p.y / p.z + cy, p.z)
  }
  public func unproject(u: Double, v: Double, depth: Double) -> SplatVector {
    position + cameraToWorld * SplatVector((u - cx) * depth / fx, -(v - cy) * depth / fy, -depth)
  }
}

/// Parameters: world xyz, log standard deviations xyz, Hamilton quaternion wxyz,
/// opacity logit, RGB (SH degree zero represented as RGB until PLY export).
public struct Gaussian: Sendable, Equatable {
  public internal(set) var parameters: [Double]
  public init(
    position: SplatVector, scale: SplatVector, rotation: [Double] = [1, 0, 0, 0],
    opacity: Double = 0.8, color: SplatVector
  ) throws {
    guard scale.values.allSatisfy({ $0 > 0 && $0.isFinite }), rotation.count == 4,
      rotation.allSatisfy(\.isFinite), rotation.reduce(0, { $0 + $1 * $1 }).isFinite,
      rotation.reduce(0, { $0 + $1 * $1 }) > 1e-12,
      position.isFinite, position.values.allSatisfy({ Float($0).isFinite }),
      color.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      opacity > 0, opacity < 1
    else { throw LiveSplatError.invalidGaussian }
    parameters =
      position.values + scale.values.map(log) + rotation + [log(opacity / (1 - opacity))]
      + color.values
    constrain()
  }
  public var position: SplatVector { SplatVector(parameters[0], parameters[1], parameters[2]) }
  public var color: SplatVector { SplatVector(parameters[11], parameters[12], parameters[13]) }
  public var opacity: Double { 1 / (1 + exp(-parameters[10])) }
  public var isValid: Bool {
    parameters.count == 14 && parameters.allSatisfy({ $0.isFinite && Float($0).isFinite })
      && parameters[6...9].reduce(0, { $0 + $1 * $1 }) > 1e-12
  }
  mutating func constrain() {
    for i in 3..<6 { parameters[i] = min(log(0.5), max(log(0.001), parameters[i])) }
    let n = sqrt(parameters[6...9].reduce(0, { $0 + $1 * $1 }))
    for i in 6..<10 { parameters[i] /= n }
    parameters[10] = min(8, max(-8, parameters[10]))
    for i in 11..<14 { parameters[i] = min(1, max(0, parameters[i])) }
  }
  public var covariance: SplatMatrix {
    SplatMatrix(SplatMath.covariance(parameters))
  }
}

public struct ProjectedGaussian: Sendable {
  public let u: Double, v: Double, depth: Double
  public let xx: Double, xy: Double, yy: Double
  public var radius: Double {
    3 * sqrt(0.5 * (xx + yy + sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)))
  }
}

/// Gaussian projection/export compatibility for offline fixtures; no optimizer.
public enum SplatMath {
  static func covariance(_ p: [Double]) -> [Double] {
    let n = sqrt(p[6...9].reduce(0) { $0 + $1 * $1 })
    let w = p[6] / n
    let x = p[7] / n
    let y = p[8] / n
    let z = p[9] / n
    let r = [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]
    let scales = (3..<6).map { exp(2 * p[$0]) }
    var result = Array(repeating: 0.0, count: 9)
    for i in 0..<9 {
      for k in 0..<3 { result[i] += r[(i / 3) * 3 + k] * scales[k] * r[(i % 3) * 3 + k] }
    }
    return result
  }
  public static func project(_ g: Gaussian, camera: SplatCamera) -> ProjectedGaussian? {
    guard g.isValid, let projected = camera.project(g.position) else { return nil }
    let p = camera.worldToOptical * (g.position - camera.position)
    let z = p.z
    let j = [
      camera.fx / z, 0, -camera.fx * p.x / (z * z), 0, camera.fy / z, -camera.fy * p.y / (z * z),
    ]
    let a = (0..<6).map { i in
      (0..<3).reduce(0.0) { $0 + j[(i / 3) * 3 + $1] * camera.worldToOptical[$1, i % 3] }
    }
    let cov = covariance(g.parameters)
    func c(_ r: Int, _ s: Int) -> Double {
      var sum = r == s ? 0.3 : 0
      for k in 0..<3 { for l in 0..<3 { sum += a[r * 3 + k] * cov[k * 3 + l] * a[s * 3 + l] } }
      return sum
    }
    return ProjectedGaussian(
      u: projected.x, v: projected.y, depth: z, xx: c(0, 0), xy: c(0, 1), yy: c(1, 1))
  }
}
