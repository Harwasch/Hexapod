import Foundation

public enum Pose {
  /// SVD polar projection U diag(1,1,det(UVᵀ)) Vᵀ, as in
  /// tools/pipeline/capture_bundle.py::world_to_camera, without COLMAP's axis flip.
  /// A symmetric Jacobi eigensolve of AᵀA supplies V and singular values;
  /// A V Σ⁻¹ supplies U. Positive input determinant is required by the server.
  public static func corrected(_ input: [Double], config: PrepConfig = .init()) throws -> [Double] {
    guard input.count == 16, input.allSatisfy(\.isFinite) else {
      throw PrepError.invalid("Pose shape/nonfinite")
    }
    let a = (0..<3).flatMap { r in (0..<3).map { input[r * 4 + $0] } }
    let gram = multiply(transpose(a), a)
    let error = (0..<9).map { abs(gram[$0] - ($0 / 3 == $0 % 3 ? 1 : 0)) }.max()!
    let rowError = (0..<4).map { abs(input[12 + $0] - ($0 == 3 ? 1 : 0)) }.max()!
    guard error <= config.rotationErrorMax, determinant(a) > 0,
      rowError <= config.homogeneousErrorMax
    else {
      throw PrepError.invalid(
        "Pose orthonormality=\(error), det=\(determinant(a)), row=\(rowError)")
    }
    var b = gram
    var v = [1.0, 0, 0, 0, 1, 0, 0, 0, 1]
    for _ in 0..<config.svdSweeps {
      guard
        let pair = [(0, 1), (0, 2), (1, 2)].max(by: {
          abs(b[$0.0 * 3 + $0.1]) < abs(b[$1.0 * 3 + $1.1])
        })
      else { break }
      let (p, q) = pair
      if abs(b[p * 3 + q]) <= config.svdEpsilon { break }
      let angle = 0.5 * atan2(2 * b[p * 3 + q], b[q * 3 + q] - b[p * 3 + p])
      let c = cos(angle)
      let s = sin(angle)
      var j = [1.0, 0, 0, 0, 1, 0, 0, 0, 1]
      j[p * 3 + p] = c
      j[q * 3 + q] = c
      j[p * 3 + q] = s
      j[q * 3 + p] = -s
      b = multiply(multiply(transpose(j), b), j)
      v = multiply(v, j)
    }
    var u = multiply(a, v)
    for c in 0..<3 {
      let sigma = sqrt(max(0, b[c * 3 + c]))
      guard sigma > 0 else { throw PrepError.invalid("Singular pose") }
      for r in 0..<3 { u[r * 3 + c] /= sigma }
    }
    // The determinant policy makes the SVD reflection correction +1.
    let rotation = multiply(u, transpose(v))
    var output = input
    for r in 0..<3 { for c in 0..<3 { output[r * 4 + c] = rotation[r * 3 + c] } }
    output.replaceSubrange(12..<16, with: [0, 0, 0, 1])
    return output
  }

  public static func distance(_ a: [Double], _ b: [Double]) -> (metres: Double, degrees: Double) {
    let metres = sqrt([3, 7, 11].reduce(0) { $0 + pow(a[$1] - b[$1], 2) })
    let trace = (0..<3).reduce(0.0) { total, c in
      total + (0..<3).reduce(0.0) { $0 + a[$1 * 4 + c] * b[$1 * 4 + c] }
    }
    return (metres, acos(min(1, max(-1, (trace - 1) / 2))) * 180 / .pi)
  }

  private static func transpose(_ a: [Double]) -> [Double] {
    (0..<9).map { a[($0 % 3) * 3 + $0 / 3] }
  }
  private static func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
    var result = [Double](repeating: 0, count: 9)
    for r in 0..<3 {
      for c in 0..<3 {
        for k in 0..<3 { result[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c] }
      }
    }
    return result
  }
  private static func determinant(_ a: [Double]) -> Double {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
      + a[2] * (a[3] * a[7] - a[4] * a[6])
  }
}
