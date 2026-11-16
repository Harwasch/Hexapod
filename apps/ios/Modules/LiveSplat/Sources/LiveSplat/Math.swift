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
    SplatMatrix(SplatMath.covariance(parameters.map { Dual($0) }).map(\.v))
  }
}

/// Scalar forward-mode differentiation. Shaders/LiveSplat.metal mirrors these
/// formulas, parameter ordering, near plane, blur variance, alpha cap and clamps.
/// Linux tests pin the CPU formulas; they do NOT verify Metal compilation/parity.
struct Dual {
  var v: Double
  var d: Double = 0
  init(_ v: Double, _ d: Double = 0) {
    self.v = v
    self.d = d
  }
  static func + (a: Self, b: Self) -> Self { Self(a.v + b.v, a.d + b.d) }
  static func - (a: Self, b: Self) -> Self { Self(a.v - b.v, a.d - b.d) }
  static prefix func - (a: Self) -> Self { Self(-a.v, -a.d) }
  static func * (a: Self, b: Self) -> Self { Self(a.v * b.v, a.d * b.v + a.v * b.d) }
  static func / (a: Self, b: Self) -> Self {
    Self(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v))
  }
  func exponential() -> Self { Self(exp(v), exp(v) * d) }
  func root() -> Self { Self(sqrt(v), d / (2 * sqrt(v))) }
}

public struct ProjectedGaussian: Sendable {
  public let u: Double, v: Double, depth: Double
  public let xx: Double, xy: Double, yy: Double
  public var radius: Double {
    3 * sqrt(0.5 * (xx + yy + sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)))
  }
}

public struct PhotometricSample: Sendable {
  public let u: Double, v: Double
  public let target: SplatVector
  public let background: SplatVector
  public init(u: Double, v: Double, target: SplatVector, background: SplatVector = .zero) {
    self.u = u
    self.v = v
    self.target = target
    self.background = background
  }
}

public enum SplatMath {
  static func covariance(_ p: [Dual]) -> [Dual] {
    let n = p[6...9].reduce(Dual(0)) { $0 + $1 * $1 }.root()
    let w = p[6] / n
    let x = p[7] / n
    let y = p[8] / n
    let z = p[9] / n
    let one = Dual(1)
    let two = Dual(2)
    let r = [
      one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y),
      two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x),
      two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y),
    ]
    let s = (3..<6).map { (two * p[$0]).exponential() }
    var result = Array(repeating: Dual(0), count: 9)
    for i in 0..<9 {
      for k in 0..<3 {
        let left = r[(i / 3) * 3 + k] * s[k]
        result[i] = result[i] + left * r[(i % 3) * 3 + k]
      }
    }
    return result
  }
  // Returns [u,v,z,Cxx,Cxy,Cyy], with C2 = J W C3 W^T J^T + 0.3 I.
  static func projected(_ p: [Dual], _ camera: SplatCamera) -> [Dual]? {
    let w = camera.worldToOptical
    let relative = (0..<3).map { p[$0] - Dual(camera.position.values[$0]) }
    let t = (0..<3).map { r in (0..<3).reduce(Dual(0)) { $0 + Dual(w[r, $1]) * relative[$1] } }
    guard t[2].v > 0.01 else { return nil }
    let fx = Dual(camera.fx)
    let fy = Dual(camera.fy)
    let z = t[2]
    let j = [fx / z, Dual(0), -fx * t[0] / (z * z), Dual(0), fy / z, -fy * t[1] / (z * z)]
    let a = (0..<6).map { i in
      (0..<3).reduce(Dual(0)) { $0 + j[(i / 3) * 3 + $1] * Dual(w[$1, i % 3]) }
    }
    let cov = covariance(p)
    func c(_ r: Int, _ s: Int) -> Dual {
      var sum = Dual(r == s ? 0.3 : 0)
      for k in 0..<3 { for l in 0..<3 { sum = sum + a[r * 3 + k] * cov[k * 3 + l] * a[s * 3 + l] } }
      return sum
    }
    let xx = c(0, 0)
    let xy = c(0, 1)
    let yy = c(1, 1)
    let determinant = xx.v * yy.v - xy.v * xy.v
    guard determinant.isFinite, determinant > 0 else { return nil }
    return [fx * t[0] / z + Dual(camera.cx), fy * t[1] / z + Dual(camera.cy), z, xx, xy, yy]
  }
  public static func project(_ g: Gaussian, camera: SplatCamera) -> ProjectedGaussian? {
    guard g.isValid, let c = projected(g.parameters.map { Dual($0) }, camera),
      c.allSatisfy({ $0.v.isFinite })
    else { return nil }
    return ProjectedGaussian(
      u: c[0].v, v: c[1].v, depth: c[2].v, xx: c[3].v, xy: c[4].v, yy: c[5].v)
  }
  static func loss(_ p: [Dual], camera: SplatCamera, samples: [PhotometricSample]) -> Dual {
    guard let c = projected(p, camera), !samples.isEmpty else { return Dual(0) }
    let det = c[3] * c[5] - c[4] * c[4]
    var loss = Dual(0)
    for sample in samples {
      let dx = Dual(sample.u) - c[0]
      let dy = Dual(sample.v) - c[1]
      let q = (c[5] * dx * dx - Dual(2) * c[4] * dx * dy + c[3] * dy * dy) / det
      let raw = (-Dual(0.5) * q).exponential() / (Dual(1) + (-p[10]).exponential())
      let alpha = raw.v < 0.99 ? raw : Dual(0.99)
      for k in 0..<3 {
        let prediction = alpha * p[11 + k] + (Dual(1) - alpha) * Dual(sample.background.values[k])
        let e = prediction - Dual(sample.target.values[k])
        loss = loss + Dual(0.5) * e * e
      }
    }
    return loss / Dual(Double(samples.count))
  }
  public static func loss(_ g: Gaussian, camera: SplatCamera, samples: [PhotometricSample])
    -> Double
  {
    guard g.isValid else { return .infinity }
    return loss(g.parameters.map { Dual($0) }, camera: camera, samples: samples).v
  }
  public static func gradient(_ g: Gaussian, camera: SplatCamera, samples: [PhotometricSample])
    -> [Double]
  {
    guard g.isValid else { return Array(repeating: 0, count: 14) }
    return (0..<14).map { index in
      loss(
        g.parameters.enumerated().map { Dual($0.element, $0.offset == index ? 1 : 0) },
        camera: camera, samples: samples
      ).d
    }
  }
  public static func step(_ g: Gaussian, gradient: [Double], learningRate: Double = 0.001)
    -> Gaussian
  {
    guard g.isValid, gradient.count == 14, gradient.allSatisfy(\.isFinite),
      learningRate.isFinite, learningRate > 0
    else { return g }
    var result = g
    // Per-component clipping bounds an individual update; this is not a convergence guarantee.
    for i in 0..<14 { result.parameters[i] -= learningRate * min(1, max(-1, gradient[i])) }
    guard result.isValid else { return g }
    result.constrain()
    return result
  }
  /// Front-to-back, premultiplied alpha; caller provides increasing camera depth.
  public static func composite(_ layers: [(color: SplatVector, alpha: Double)]) -> (
    SplatVector, Double
  ) {
    var color = SplatVector.zero
    var transmittance = 1.0
    for layer in layers {
      let a = min(0.99, max(0, layer.alpha))
      color = color + layer.color * (transmittance * a)
      transmittance *= 1 - a
    }
    return (color, 1 - transmittance)
  }
}

/// CPU reference of one shader tile's bounded insertion sort. This is a numerical
/// oracle for device parity tests, not the runtime fallback's rendering algorithm.
public struct SplatTile: Sendable {
  public let indices: [Int]
  public let overflow: Int
  public static func bin(_ projections: [ProjectedGaussian?], column: Int, row: Int) -> Self {
    let loX = Double(column * 16)
    let loY = Double(row * 16)
    var indices: [Int] = []
    var dropped = 0
    for (i, optional) in projections.enumerated() {
      guard let p = optional, p.depth > 0,
        p.u + p.radius >= loX, p.u - p.radius <= loX + 16,
        p.v + p.radius >= loY, p.v - p.radius <= loY + 16
      else { continue }
      let at = indices.firstIndex { projections[$0]!.depth > p.depth } ?? indices.count
      if indices.count == 64 { dropped += 1 }
      if at < 64 {
        indices.insert(i, at: at)
        if indices.count > 64 { indices.removeLast() }
      }
    }
    return Self(indices: indices, overflow: dropped)
  }
}
