import Foundation

/// One CoreLocation observation. Timestamp is Unix UTC seconds, separate from ARKit time.
public struct CaptureLocation: Codable, Equatable, Sendable {
  public var timestamp: Double
  public var lat: Double
  public var lon: Double
  public var altitudeM: Double
  public var horizontalAccuracyM: Double
  /// A negative accuracy preserves CoreLocation's indication that altitude is invalid.
  public var verticalAccuracyM: Double

  public init(
    timestamp: Double, latitude: Double, longitude: Double, altitudeM: Double,
    horizontalAccuracyM: Double, verticalAccuracyM: Double
  ) {
    self.timestamp = timestamp
    lat = latitude
    lon = longitude
    self.altitudeM = altitudeM
    self.horizontalAccuracyM = horizontalAccuracyM
    self.verticalAccuracyM = verticalAccuracyM
  }

  public var isValid: Bool {
    [timestamp, lat, lon, altitudeM, horizontalAccuracyM, verticalAccuracyM].allSatisfy(\.isFinite)
      && (-90...90).contains(lat) && (-180...180).contains(lon)
  }

  /// Lowest positive horizontal uncertainty; ties keep the first delivered fix.
  public func improves(_ previous: CaptureLocation?) -> Bool {
    isValid && horizontalAccuracyM > 0
      && (previous == nil || horizontalAccuracyM < previous!.horizontalAccuracyM)
  }
}

#if canImport(CoreLocation)
  @preconcurrency import CoreLocation

  /// CoreLocation needs a run loop; all manager/delegate access stays on the main actor.
  @MainActor
  final class CaptureLocationRecorder: NSObject, @preconcurrency CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private let receive: @Sendable ([CaptureLocation]) -> Void

    init(receive: @escaping @Sendable ([CaptureLocation]) -> Void) {
      self.receive = receive
      super.init()
      manager.delegate = self
      manager.desiredAccuracy = kCLLocationAccuracyBest
    }

    func start() {
      if manager.authorizationStatus == .notDetermined {
        manager.requestWhenInUseAuthorization()
      } else if [.authorizedWhenInUse, .authorizedAlways].contains(manager.authorizationStatus) {
        manager.startUpdatingLocation()
      }
    }

    func stop() {
      manager.stopUpdatingLocation()
      manager.delegate = nil
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
      if [.authorizedWhenInUse, .authorizedAlways].contains(manager.authorizationStatus) {
        manager.startUpdatingLocation()
      }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
      receive(
        locations.map {
          CaptureLocation(
            timestamp: $0.timestamp.timeIntervalSince1970,
            latitude: $0.coordinate.latitude, longitude: $0.coordinate.longitude,
            altitudeM: $0.altitude, horizontalAccuracyM: $0.horizontalAccuracy,
            verticalAccuracyM: $0.verticalAccuracy)
        })
    }

    // Permission denial or unavailable GPS leaves the location absent; never invent a fix.
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
  }
#endif
