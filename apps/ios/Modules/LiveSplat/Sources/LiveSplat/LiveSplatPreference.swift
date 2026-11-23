import Foundation

#if canImport(Combine)
  import Combine
#endif

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
    set {
      guard newValue != previewEnabled else { return }
      #if canImport(Combine)
        objectWillChange.send()
      #endif
      store.set(newValue, forKey: Self.key)
      observers.removeAll { $0.value == nil }
      for observer in observers { observer.value?.preferenceChanged() }
    }
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
