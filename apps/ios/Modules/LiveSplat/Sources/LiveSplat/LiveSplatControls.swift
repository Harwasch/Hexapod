#if canImport(SwiftUI)
  import SwiftUI

  /// Icon-only scan control. Share the renderer's preference instance.
  @MainActor public struct LiveSplatToggle: View {
    @ObservedObject private var preference: LiveSplatPreference
    public init(preference: LiveSplatPreference = .shared) { self.preference = preference }
    public var body: some View {
      Button {
        preference.previewEnabled.toggle()
      } label: {
        Image(systemName: preference.previewEnabled ? "sparkles" : "circle.dotted")
      }
      .accessibilityLabel("Scan preview")
      .accessibilityValue(preference.previewEnabled ? "On" : "Off")
    }
  }

  @MainActor public struct LiveSplatSettingsRow: View {
    @ObservedObject private var preference: LiveSplatPreference
    public init(preference: LiveSplatPreference = .shared) { self.preference = preference }
    public var body: some View {
      Toggle(
        "Scan preview",
        isOn: Binding(
          get: { preference.previewEnabled },
          set: { preference.previewEnabled = $0 }))
    }
  }
#endif
