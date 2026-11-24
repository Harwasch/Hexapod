#if canImport(SwiftUI) && canImport(RealityKit) && os(iOS)
  import SwiftUI

  /// Transparent gesture surface over the GPU viewport. No second display loop.
  /// Touch alignment, enlargement and drag feel require verification on the phone.
  @MainActor public struct ScanInsetHost: View {
    private let renderer: DefaultLiveSplatRenderer
    @ObservedObject private var preference: LiveSplatPreference
    @State private var orbit = ScanOrbit()
    @State private var previous = CGSize.zero
    public init(renderer: DefaultLiveSplatRenderer) {
      self.renderer = renderer
      preference = renderer.preference
    }
    public var body: some View {
      GeometryReader { geometry in
        if preference.insetEnabled {
          let rect = orbit.rect
          RoundedRectangle(cornerRadius: 12)
            .stroke(.white.opacity(0.65), lineWidth: 1)
            .background(Color.black.opacity(0.001))
            .contentShape(Rectangle())
            .frame(
              width: geometry.size.width * CGFloat(rect.z),
              height: geometry.size.height * CGFloat(rect.w)
            )
            .position(
              x: geometry.size.width * CGFloat(rect.x + rect.z / 2),
              y: geometry.size.height * CGFloat(rect.y + rect.w / 2)
            )
            .onTapGesture {
              orbit.enlarged.toggle()
              renderer.processor?.setOrbit(orbit)
            }
            .gesture(
              DragGesture().onChanged { value in
                orbit.rotate(
                  horizontal: Float(value.translation.width - previous.width) * 0.01,
                  vertical: Float(value.translation.height - previous.height) * 0.01)
                previous = value.translation
                renderer.processor?.setOrbit(orbit)
              }.onEnded { _ in previous = .zero }
            )
            .accessibilityLabel("Scan inset")
            .accessibilityHint("Tap to enlarge. Drag to rotate.")
        }
      }
      .onAppear { renderer.processor?.setOrbit(orbit) }
    }
  }
#endif
