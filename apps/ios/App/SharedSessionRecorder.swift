#if canImport(ARKit) && os(iOS)
  import Capture
  import CoreLocation
  import Darwin
  import Foundation

  /// MainActor owns controls and location only; ingress never calls back into the UI.
  @MainActor
  final class SharedSessionRecorder: NSObject, RecordingControl,
    @preconcurrency CLLocationManagerDelegate
  {
    nonisolated let ingress = ScanIngress()
    private(set) var isRecording = false
    var captureDetails: (() -> CaptureDetails)?
    var onPhase: ((CapturePhase) -> Void)?
    var onChange: ((Bool, Bool) -> Void)?
    private var writer: BackgroundWriter<ScanPacket>?
    private var worker: ScanRecordingWorker?
    private var directory: URL?
    private var finishing = false
    private var starting: Task<Void, Never>?
    private var generation = UUID()
    private let location = CLLocationManager()

    func toggleRecording() {
      if isRecording {
        stop()
        return
      }
      guard !finishing else { return }
      finishing = true
      onPhase?(.starting)
      let id = UUID()
      generation = id
      starting = Task {
        defer { if generation == id { starting = nil } }
        do {
          let root = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Captures")
          var system = utsname()
          uname(&system)
          let device = withUnsafeBytes(of: &system.machine) {
            String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self)
          }
          let directory = root.appendingPathComponent(UUID().uuidString)
          let worker = try await ScanRecordingWorker.create(directory: directory, device: device)
          // Hold exposure and white balance for the whole scan, before the first frame is
          // admitted, so no frame of it is auto-exposed (CaptureCamera.holdExposure).
          await ingress.camera.holdExposure()
          guard generation == id, !Task.isCancelled else {
            ingress.camera.releaseExposure()
            return
          }
          let writer = BackgroundWriter<ScanPacket>(capacity: 4) { try await worker.write($0) }
          self.worker = worker
          self.writer = writer
          self.directory = directory
          ingress.start(writer)
          finishing = false
          isRecording = true
          location.delegate = self
          location.requestWhenInUseAuthorization()
          location.startUpdatingLocation()
          onChange?(true, false)
          onPhase?(.recording(Date()))
        } catch { if generation == id { fail() } }
      }
    }
    func pollFailure() { if writer?.hasFailed == true { stop() } }
    func stop() {
      ingress.stop()
      if starting != nil {
        generation = UUID()
        starting?.cancel()
        starting = nil
        finishing = false
        onPhase?(.idle)
      }
      guard isRecording else { return }
      isRecording = false
      location.stopUpdatingLocation()
      let settings = ingress.camera.settings
      ingress.camera.releaseExposure()
      onChange?(false, false)
      guard let worker, let writer, let directory else {
        fail()
        return
      }
      finishing = true
      onPhase?(.stopping)
      let details = captureDetails?()
      Task { [weak self] in
        do {
          // Drain retained frames BEFORE closing the codec and publishing the inventory.
          try await writer.close()
          try await worker.finish(details: details, settings: settings)
          self?.reset()
          self?.onPhase?(.saved(directory))
        } catch {
          await worker.cancel()
          self?.fail()
        }
      }
    }
    private func fail() {
      ingress.stop()
      ingress.camera.releaseExposure()
      location.stopUpdatingLocation()
      reset()
      onChange?(false, true)
      onPhase?(.failed)
    }
    private func reset() {
      generation = UUID()
      starting?.cancel()
      starting = nil
      isRecording = false
      finishing = false
      writer = nil
      worker = nil
      directory = nil
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
      guard isRecording else { return }
      for fix in locations {
        ingress.location(
          CaptureLocation(
            timestamp: fix.timestamp.timeIntervalSince1970,
            latitude: fix.coordinate.latitude, longitude: fix.coordinate.longitude,
            altitudeM: fix.altitude,
            horizontalAccuracyM: fix.horizontalAccuracy, verticalAccuracyM: fix.verticalAccuracy))
      }
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
  }
#endif
