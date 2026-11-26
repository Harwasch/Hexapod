import Perception

/// Overlay failure cannot bypass creation (or re-enabling) of the perception driver.
@MainActor enum CameraAttachment {
  static func attach<Overlay, Driver>(
    makeOverlay: () throws -> Overlay,
    makeDriver: (Overlay?) -> Driver
  ) -> (overlay: Overlay?, driver: Driver, overlayError: String?) {
    let overlay: Overlay?
    let errorText: String?
    do {
      overlay = try makeOverlay()
      errorText = nil
    } catch {
      overlay = nil
      errorText = String(describing: error)
    }
    return (overlay, makeDriver(overlay), errorText)
  }
}

struct CameraSelectionFeedback {
  private(set) var lastRefusal: SelectionRefusal?
  private(set) var banner: String?
  mutating func receive(_ result: TapAdmissionResult) {
    switch result {
    case .accepted: banner = nil
    case .waiting: banner = "Hold still…"
    case .refused(let reason):
      lastRefusal = reason
      banner = "Can't select here"
    }
  }
}
