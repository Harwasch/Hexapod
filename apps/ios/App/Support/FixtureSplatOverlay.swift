#if os(iOS) && canImport(SwiftUI)
  import LiveSplat
  import SwiftUI

  /// One CPU reference frame, independent of RealityKit, Metal and any display link.
  /// Coverage uses the initial cool tint at time zero; no pulse or shimmer advances.
  @MainActor struct FixtureSplatOverlay: View {
    let renderer: DefaultLiveSplatRenderer
    var body: some View {
      let dots = snapshot()
      Canvas { context, size in
        for dot in dots {
          let x: CGFloat = dot.x * size.width
          let y: CGFloat = dot.y * size.height
          let bounds = CGRect(x: x - 2, y: y - 2, width: 4, height: 4)
          let colour = Color(red: dot.red, green: dot.green, blue: dot.blue).opacity(dot.opacity)
          context.fill(Path(ellipseIn: bounds), with: .color(colour))
        }
      }
    }
    private struct Dot: Sendable {
      let x: CGFloat, y: CGFloat
      let red: Double, green: Double, blue: Double, opacity: Double
    }
    private func snapshot() -> [Dot] {
      guard let frame = renderer.latestFrame else { return [] }
      let transform = renderer.displayTransform
      var dots: [Dot] = []
      for splat in renderer.splats {
        guard let point = frame.camera.project(splat.position) else { continue }
        let u: Double = point.x / Double(frame.camera.width)
        let v: Double = point.y / Double(frame.camera.height)
        let x: Double = transform[0] * u + transform[2] * v + transform[4]
        let y: Double = transform[1] * u + transform[3] * v + transform[5]
        guard (0...1).contains(x), (0...1).contains(y) else { continue }
        let colour = SIMD3<Float>(Float(splat.color.x), Float(splat.color.y), Float(splat.color.z))
        let progress: Float = renderer.preference.coverageColoursEnabled ? 0 : 1
        let appearance = CoverageAppearance.appearance(colour: colour, progress: progress, time: 0)
        dots.append(
          Dot(
            x: CGFloat(x), y: CGFloat(y), red: Double(appearance.colour.x),
            green: Double(appearance.colour.y), blue: Double(appearance.colour.z),
            opacity: Double(appearance.opacity)))
      }
      return dots
    }
  }
#endif
