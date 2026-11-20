#if canImport(SwiftUI)
  import SwiftUI

  /// Icon-only scan control. Share the renderer's preference instance.
  @MainActor public struct LiveSplatToggle: View {
    @ObservedObject private var preference: LiveSplatPreference
    public init(preference: LiveSplatPreference = .shared) { self.preference = preference }
    public var body: some View {
      Button {
        preference.trainingEnabled.toggle()
      } label: {
        Image(systemName: preference.trainingEnabled ? "sparkles" : "circle.dotted")
      }
      .accessibilityLabel("Live splat")
      .accessibilityValue(preference.trainingEnabled ? "On" : "Off")
    }
  }

  @MainActor public struct LiveSplatSettingsRow: View {
    @ObservedObject private var preference: LiveSplatPreference
    public init(preference: LiveSplatPreference = .shared) { self.preference = preference }
    public var body: some View {
      Toggle(
        "Live splat",
        isOn: Binding(
          get: { preference.trainingEnabled },
          set: { preference.trainingEnabled = $0 }))
    }
  }
#endif
