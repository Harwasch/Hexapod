import Foundation

/// Prototype defaults, not calibrated training-quality thresholds. Credit bytes encode
/// 0...viewsToFill as 0...255; each keyframe rounds to the nearest byte.
public struct CoverageCreditParameters: Codable, Equatable, Sendable {
  /// 1...14 active directions, default 14 (the packed ABI has 14 credit bytes).
  public var hairs: Int = 14
  /// Angular support radius in degrees, default 25.
  public var coneDeg: Float = 25
  /// Cone falloff exponent, default 1 (linear in cosine).
  public var sharpness: Float = 1
  /// Ideal contributing keyframes required, default 8.
  public var viewsToFill: Float = 8
  /// nil disables resolution weighting; otherwise w_res = min(1, target / mmPerPx).
  public var targetMmPerPx: Float? = nil
  /// nil disables the two-view gate. Separation is measured from the first contributor.
  public var parallaxMinDeg: Float? = nil
  /// Mean-normal and position update weight, default 0.2.
  public var normalEMA: Float = 0.2
  /// Maximum hair length, in world metres.
  public var baseLength: Float = 0.08
  /// Visible fraction retained until optional parallax succeeds, default 0.15.
  public var parallaxStubFraction: Float = 0.15
  public init() {}
  public static func load(_ data: Data) throws -> Self {
    let value = try JSONDecoder().decode(Self.self, from: data)
    guard (1...14).contains(value.hairs), value.coneDeg.isFinite,
      value.coneDeg > 0, value.coneDeg < 90,
      value.sharpness.isFinite, value.sharpness > 0,
      value.viewsToFill.isFinite, value.viewsToFill >= 1, value.viewsToFill <= 255,
      value.normalEMA.isFinite, value.normalEMA > 0, value.normalEMA <= 1,
      value.baseLength.isFinite, value.baseLength > 0, value.baseLength <= 0.2,
      value.parallaxStubFraction.isFinite, value.parallaxStubFraction > 0,
      value.parallaxStubFraction <= 1,
      value.targetMmPerPx.map({ $0.isFinite && $0 > 0 }) ?? true,
      value.parallaxMinDeg.map({ $0.isFinite && $0 > 0 && $0 < 180 }) ?? true
    else { throw CoverageCreditError.invalidParameters }
    return value
  }
  public static func bundled() throws -> Self {
    guard let url = Bundle.module.url(forResource: "coverage_credit", withExtension: "json") else {
      throw CoverageCreditError.missingResource
    }
    return try load(Data(contentsOf: url))
  }
  public init(from decoder: any Decoder) throws {
    self.init()
    let c = try decoder.container(keyedBy: CodingKeys.self)
    hairs = try c.decodeIfPresent(Int.self, forKey: .hairs) ?? hairs
    coneDeg = try c.decodeIfPresent(Float.self, forKey: .coneDeg) ?? coneDeg
    sharpness = try c.decodeIfPresent(Float.self, forKey: .sharpness) ?? sharpness
    viewsToFill = try c.decodeIfPresent(Float.self, forKey: .viewsToFill) ?? viewsToFill
    targetMmPerPx = try c.decodeIfPresent(Float.self, forKey: .targetMmPerPx)
    parallaxMinDeg = try c.decodeIfPresent(Float.self, forKey: .parallaxMinDeg)
    normalEMA = try c.decodeIfPresent(Float.self, forKey: .normalEMA) ?? normalEMA
    baseLength = try c.decodeIfPresent(Float.self, forKey: .baseLength) ?? baseLength
    parallaxStubFraction =
      try c.decodeIfPresent(Float.self, forKey: .parallaxStubFraction) ?? parallaxStubFraction
  }
}
public enum CoverageCreditError: Error { case invalidParameters, missingResource }

public enum CoverageCreditReference {
  static func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x * b.x + a.y * b.y + a.z * b.z
  }
  static func unit(_ v: SIMD3<Float>) -> SIMD3<Float> {
    let d = dot(v, v)
    return d.isFinite && d > 1e-12 ? v / sqrt(d) : SIMD3(0, 0, 1)
  }
  static func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
  }
  public static func encodeNormal(_ normal: SIMD3<Float>) -> SIMD2<Int16> {
    let n = unit(normal)
    var p = SIMD2(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z))
    if n.z < 0 {
      p = SIMD2(
        (1 - abs(p.y)) * (p.x >= 0 ? 1 : -1),
        (1 - abs(p.x)) * (p.y >= 0 ? 1 : -1))
    }
    return SIMD2(Int16((p.x * 32767).rounded()), Int16((p.y * 32767).rounded()))
  }
  public static func decodeNormal(_ encoded: SIMD2<Int16>) -> SIMD3<Float> {
    let p = SIMD2<Float>(Float(encoded.x), Float(encoded.y)) / 32767
    var n = SIMD3(p.x, p.y, 1 - abs(p.x) - abs(p.y))
    let t = max(0, -n.z)
    n.x += n.x >= 0 ? -t : t
    n.y += n.y >= 0 ? -t : t
    return unit(n)
  }
  /// Outward cell-to-camera directions in a deterministic normal-relative hemisphere.
  public static func hair(_ index: Int, count: Int, normal: SIMD3<Float>) -> SIMD3<Float> {
    let n = unit(normal)
    let tangent = unit(cross(abs(n.z) < 0.999 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0), n))
    let bitangent = cross(n, tangent)
    let z = (Float(index) + 0.5) / Float(count)
    let r = sqrt(max(0, 1 - z * z))
    let phi = Float(index) * 2.39996323
    return tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * z
  }
  /// Exactly zero on/outside the cone, including back-facing/invalid resolution.
  public static func credit(cosine: Float, mmPerPx: Float, parameters p: CoverageCreditParameters)
    -> Float
  {
    let edge = cos(p.coneDeg * .pi / 180)
    guard cosine.isFinite, cosine > edge, mmPerPx.isFinite, mmPerPx > 0 else { return 0 }
    let resolution = p.targetMmPerPx.map { min(1, $0 / mmPerPx) } ?? 1
    return resolution * pow(min(1, (cosine - edge) / (1 - edge)), p.sharpness)
  }
  public static func creditedByte(_ old: UInt8, credit: Float, viewsToFill: Float) -> UInt8 {
    UInt8(min(255, max(0, (Float(old) + credit * 255 / viewsToFill).rounded())))
  }
  public static func filled(_ credit: UInt8, parallax: Bool, parameters: CoverageCreditParameters)
    -> Bool
  {
    credit == 255 && (parameters.parallaxMinDeg == nil || parallax)
  }
  public static func remaining(
    _ credit: UInt8, parallax: Bool, parameters: CoverageCreditParameters
  )
    -> Float
  {
    max(
      1 - Float(credit) / 255,
      parameters.parallaxMinDeg != nil && !parallax ? parameters.parallaxStubFraction : 0)
  }
}

/// Stable 32-byte Swift/Metal ABI: normal 0, credit 4, views 18, epoch 20,
/// parallax flags 22, best mm/px 24, six reserved bytes 26. Keys/positions and
/// optional first-contributor directions live in separate fixed-capacity buffers.
public struct CoverageGPUCell: Sendable {
  public var normal = SIMD2<Int16>(0, 0)
  private var credits:
    (
      UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
      UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  public var viewCount: UInt16 = 0
  public var epoch: UInt16 = 0
  public var flags: UInt16 = 0
  public var bestMmPerPx: Float16 = 0
  private var padding: (UInt16, UInt16, UInt16) = (0, 0, 0)
  public init() {}
  public subscript(hair: Int) -> UInt8 {
    get {
      precondition((0..<14).contains(hair))
      return withUnsafeBytes(of: credits) { $0[hair] }
    }
    set {
      precondition((0..<14).contains(hair))
      withUnsafeMutableBytes(of: &credits) { $0[hair] = newValue }
    }
  }
}
