#if canImport(SwiftUI)
  import SwiftUI
  import EntitySchema

  /// Shared card for camera and map answers.
  @MainActor public struct EntityCard: View {
    let entity: Entity
    let close: () -> Void
    public init(entity: Entity, close: @escaping () -> Void) {
      self.entity = entity
      self.close = close
    }
    public var body: some View {
      HStack {
        VStack(alignment: .leading) {
          Text(entity.label).lineLimit(1)
          Text("\(entity.extent.y, specifier: "%.1f") m").font(.caption)
        }
        Spacer()
        Button(action: close) { Image(systemName: "xmark") }.accessibilityLabel("Close")
      }
      .padding(16)
      .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
  }
#endif
