#if canImport(SwiftUI)
  import SwiftUI
  import EntitySchema

  /// Shared card for camera and map answers. Only display attributes present in evidence.
  @MainActor public struct EntityCard: View {
    let entity: Entity
    let compact: Bool
    let close: () -> Void
    let pin: (() -> Void)?
    let ask: (() -> Void)?
    public init(
      entity: Entity, compact: Bool = false, pin: (() -> Void)? = nil, ask: (() -> Void)? = nil,
      close: @escaping () -> Void
    ) {
      self.entity = entity
      self.compact = compact
      self.pin = pin
      self.ask = ask
      self.close = close
    }
    private let panel = Color(red: 20 / 255, green: 32 / 255, blue: 35 / 255)
    private let amber = Color(red: 1, green: 181 / 255, blue: 71 / 255)
    public var body: some View {
      VStack(alignment: .leading, spacing: compact ? 8 : 16) {
        HStack(alignment: .top) {
          VStack(alignment: .leading, spacing: 5) {
            Text(entity.label).font(.system(size: compact ? 18 : 22, weight: .semibold))
            if let latin = EntityCardFacts(entity).latinName {
              Text(latin).font(.system(size: 14)).italic().foregroundStyle(.secondary)
            }
          }
          Spacer()
          Button(action: close) { Image(systemName: "xmark").frame(width: 44, height: 44) }
            .accessibilityLabel("Close entity")
        }
        HStack(spacing: 8) {
          if EntityCardFacts(entity).invasive {
            Text("Invasive").padding(.horizontal, 12).padding(.vertical, 5)
              .background(Color(red: 1, green: 107 / 255, blue: 90 / 255), in: Capsule())
              .foregroundStyle(.black)
          }
          Text("\(Int(entity.confidence * 100))%")
            .padding(.horizontal, 12).padding(.vertical, 5).background(
              .white.opacity(0.08), in: Capsule())
          Text("\(entity.extent.y, specifier: "%.1f") m")
            .padding(.horizontal, 12).padding(.vertical, 5).background(
              .white.opacity(0.08), in: Capsule())
        }.font(.system(size: 13, weight: .medium, design: .monospaced))
        HStack(spacing: 10) {
          Button {
            pin?()
          } label: {
            Label("Pin", systemImage: "mappin.and.ellipse").frame(maxWidth: .infinity).frame(
              height: compact ? 40 : 52
            )
            .background(amber, in: RoundedRectangle(cornerRadius: 14)).foregroundStyle(.black)
          }.disabled(pin == nil)
          Button {
            ask?()
          } label: {
            Label("Ask", systemImage: "mic").frame(maxWidth: .infinity).frame(
              height: compact ? 40 : 52
            )
            .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
          }.disabled(ask == nil)
        }.font(.system(size: 16, weight: .semibold))
      }
      .padding(compact ? 12 : 20).background(panel, in: RoundedRectangle(cornerRadius: 24))
      .foregroundStyle(Color(red: 228 / 255, green: 236 / 255, blue: 233 / 255))
      .buttonStyle(.plain).accessibilityIdentifier("entity.card")
    }
  }
#endif
