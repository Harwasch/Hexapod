import Foundation

public enum GlobeImplementation: String, CaseIterable, Sendable {
  case cesium = "Cesium"
  case appleMaps = "Apple Maps 3D"
  public static let defaultValue = Self.cesium
}

extension GlobeSnapshot {
  /// Wire selection may identify an epoch; App scan routes must identify a capture.
  public func captureRoute(for selection: UUID?) -> GlobeRoute {
    if let epoch = property.epochs.first(where: { ($0.captureId ?? $0.id) == selection }) {
      return GlobeRoute(siteID: property.id, scanID: epoch.captureId)
    }
    return GlobeRoute(siteID: property.id, scanID: selection)
  }

  public func bridgeSnapshot(fixture: Bool = false) -> BridgeSnapshot {
    BridgeSnapshot(
      pins: pins.compactMap { pin in
        guard let geo = pin.coordinate else { return nil }
        return BridgePin(
          id: pin.id.uuidString.lowercased(), siteId: pin.siteId.uuidString.lowercased(),
          scanId: pin.scanId.uuidString.lowercased(), label: pin.entity.label,
          observedAt: pin.observedAt, uncertaintyM: pin.uncertaintyM,
          geo: BridgeGeo(lat: geo.lat, lon: geo.lon, alt: geo.alt))
      },
      scans: property.epochs.compactMap { scan in
        guard let point = scan.footprint.polygons.first?.first?.first, point.count >= 2 else {
          return nil
        }
        let asset = property.assets?.first { asset in
          asset.representation == "gaussian-splat" && asset.source.type == "3d-tiles-url"
            && (scan.currentAssetId.map { asset.id == $0 } ?? (asset.epochId == scan.id))
        }
        return BridgeScan(
          id: (scan.captureId ?? scan.id).uuidString.lowercased(),
          geo: BridgeGeo(lat: point[1], lon: point[0], alt: 10), polygons: scan.footprint.polygons,
          tilesetURL: fixture ? "./fixtures/globe-splat/tileset.json" : asset?.source.url)
      }, boundary: property.boundary.polygons)
  }
}

/// Only events about the currently displayed snapshot may open native UI.
public struct GlobeBridgeDispatcher: Sendable {
  public enum Action: Equatable, Sendable {
    case ready
    case pin(UUID)
    case scan(UUID)
    case failure(String)
  }
  public init() {}
  public func dispatch(_ message: BridgeMessage, snapshot: GlobeSnapshot?) throws -> Action {
    switch message {
    case .ready: return .ready
    case .failure(let message): return .failure(message)
    case .pinTapped(let id):
      guard let uuid = UUID(uuidString: id),
        snapshot?.pins.contains(where: { $0.id == uuid }) == true
      else { throw GlobeError.invalidResponse }
      return .pin(uuid)
    case .scanSelected(let id):
      guard let uuid = UUID(uuidString: id),
        snapshot?.property.epochs.contains(where: { ($0.captureId ?? $0.id) == uuid }) == true
      else { throw GlobeError.invalidResponse }
      return .scan(uuid)
    default: throw GlobeError.invalidResponse
    }
  }
}

/// Queue only the latest native state until the local page is ready, and replay after reload.
public struct GlobeBridgeState: Sendable {
  public private(set) var ready = false
  private var snapshot: BridgeSnapshot?
  private var scanID: String?
  public init() {}
  public mutating func invalidate() { ready = false }
  public mutating func update(snapshot: BridgeSnapshot, scanID: String?) -> [BridgeMessage] {
    let selected = snapshot.scans.first(where: { $0.id == scanID })?.id ?? snapshot.scans.first?.id
    let changed = self.snapshot != snapshot
    let selectionChanged = self.scanID != selected
    self.snapshot = snapshot
    self.scanID = selected
    guard ready else { return [] }
    return (changed ? [.snapshot(snapshot: snapshot)] : [])
      + (changed || selectionChanged ? focus() : [])
  }
  public mutating func becomeReady() -> [BridgeMessage] {
    ready = true
    guard let snapshot else { return [] }
    return [.snapshot(snapshot: snapshot)] + focus()
  }
  private func focus() -> [BridgeMessage] {
    guard let scanID else { return [] }
    return [.selectScan(scanId: scanID), .flyToScan(scanId: scanID)]
  }
}
