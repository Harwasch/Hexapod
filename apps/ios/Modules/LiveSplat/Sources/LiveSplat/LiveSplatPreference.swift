import Foundation

#if canImport(Combine)
  import Combine
#endif

public enum ScanPreviewStyle: String, CaseIterable, Sendable {
  case discs = "Discs"
  case splat = "Splat"
}

/// A missing value differs from an explicit false. Access on the main actor.
@MainActor public protocol LiveSplatPreferenceStore: AnyObject {
  func bool(forKey key: String) -> Bool?
  func set(_ value: Bool, forKey key: String)
}

@MainActor public final class UserDefaultsLiveSplatStore: LiveSplatPreferenceStore {
  private let defaults: UserDefaults
  public init(defaults: UserDefaults = .standard) { self.defaults = defaults }
  public func bool(forKey key: String) -> Bool? {
    defaults.object(forKey: key) == nil ? nil : defaults.bool(forKey: key)
  }
  public func set(_ value: Bool, forKey key: String) { defaults.set(value, forKey: key) }
}

@MainActor protocol LiveSplatPreferenceObserver: AnyObject {
  func preferenceChanged()
}

/// Share one instance between controls and renderers; write through previewEnabled.
/// Runtime fallback never writes this preference.
@MainActor public final class LiveSplatPreference {
  public static let styleKey = "hex.liveSplat.gaussianPreview"
  public static let coverageKey = "hex.liveSplat.coverageHairs"
  public static let insetKey = "hex.liveSplat.inset"
  public static let key = "hex.liveSplat.training"
  public static let shared = LiveSplatPreference()
  private let store: any LiveSplatPreferenceStore
  private struct Observer {
    weak var value: (any LiveSplatPreferenceObserver)?
  }
  private var observers: [Observer] = []
  #if canImport(Combine)
    public let objectWillChange = ObservableObjectPublisher()
  #endif
  public init(store: any LiveSplatPreferenceStore = UserDefaultsLiveSplatStore()) {
    self.store = store
  }
  public var previewEnabled: Bool {
    get { store.bool(forKey: Self.key) ?? true }
    set { update(newValue, key: Self.key, current: previewEnabled) }
  }
  public var previewStyle: ScanPreviewStyle {
    get { (store.bool(forKey: Self.styleKey) ?? false) ? .splat : .discs }
    set { update(newValue == .splat, key: Self.styleKey, current: previewStyle == .splat) }
  }
  public var insetEnabled: Bool {
    get { store.bool(forKey: Self.insetKey) ?? false }
    set { update(newValue, key: Self.insetKey, current: insetEnabled) }
  }
  public var coverageEnabled: Bool {
    get { store.bool(forKey: Self.coverageKey) ?? true }
    set { update(newValue, key: Self.coverageKey, current: coverageEnabled) }
  }
  private func update(_ value: Bool, key: String, current: Bool) {
    guard value != current else { return }
    #if canImport(Combine)
      objectWillChange.send()
    #endif
    store.set(value, forKey: key)
    observers.removeAll { $0.value == nil }
    for observer in observers { observer.value?.preferenceChanged() }
  }
  func observe(_ observer: any LiveSplatPreferenceObserver) {
    observers.removeAll { $0.value == nil }
    observers.append(Observer(value: observer))
  }
}

#if canImport(Combine)
  // SwiftUI reads this publisher on the main actor; Combine's requirement predates isolation.
  extension LiveSplatPreference: @preconcurrency ObservableObject {}
#endif
