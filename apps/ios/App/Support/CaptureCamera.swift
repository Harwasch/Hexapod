#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Capture
  import Foundation

  /// The camera side of a scan: which video format the session runs, the exposure lock held
  /// while recording, each frame's exposure, and the motion-blur cue.
  ///
  /// `observe` and `exposure` run on the ARSession delegate queue; `configure`, `holdExposure` and
  /// `releaseExposure` on the MainActor. Everything shared sits behind one lock.
  final class CaptureCamera: @unchecked Sendable {
    /// Versioned by app build: a later build that maps 4K depth differently tries again.
    static var fourKFallbackKey: String {
      "hex.capture.fourKFallback."
        + ((Bundle.main.infoDictionary?["CFBundleVersion"] as? String) ?? "0")
    }

    private let mutex = NSLock()
    private var format: CaptureVideoFormat?
    private var control = CaptureExposureControl(mode: "auto", whiteBalance: "auto")
    private var device: AVCaptureDevice?
    private var meter = MotionBlurMeter()
    private var probe = FourKProbe()
    /// This session's rejection, recorded into the default-format session that follows it.
    private var sessionFallback: CaptureFormatFallback?
    private var sessionStarted = Date.distantPast
    /// Called once, off the main thread, when the 4K format must be abandoned.
    var onFourKRejected: (@Sendable () -> Void)?

    var motionLevel: MotionBlurMeter.Level { mutex.withLock { meter.level } }

    /// The 4K rejection this build measured on this device, if any (depth aspect only).
    static var persistedFallback: CaptureFormatFallback? {
      UserDefaults.standard.data(forKey: fourKFallbackKey).flatMap {
        try? JSONDecoder().decode(CaptureFormatFallback.self, from: $0)
      }
    }

    /// The format the session will run: ARKit's 4K one unless this build ruled it out here.
    @MainActor static func videoFormat() -> (ARConfiguration.VideoFormat?, String) {
      if persistedFallback == nil,
        let format = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution
      {
        return (format, "4k")
      }
      return (nil, "default")
    }

    @MainActor func configure(_ configuration: ARWorldTrackingConfiguration, kind: String) {
      let format = configuration.videoFormat
      let persisted = Self.persistedFallback
      mutex.withLock {
        self.format = CaptureVideoFormat(
          kind: kind, width: Int(format.imageResolution.width),
          height: Int(format.imageResolution.height), framesPerSecond: format.framesPerSecond,
          fallback: kind == "default" ? (sessionFallback ?? persisted) : nil)
        if kind == "4k" { sessionFallback = nil }
        device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
        meter = MotionBlurMeter()
        probe = FourKProbe()
        sessionStarted = Date()
      }
    }

    /// What finish() writes into the manifest.
    var settings: (CaptureVideoFormat?, CaptureExposureControl) {
      mutex.withLock { (format, control) }
    }

    /// Hold what auto-exposure metered on the shortest shutter the noise ceiling allows
    /// (ExposurePlan), and lock white balance. Waits out the first second of a fresh session so it has metered.
    @MainActor func holdExposure() async {
      let age = mutex.withLock { Date().timeIntervalSince(sessionStarted) }
      if age < ExposurePlan.meteringS {
        try? await Task.sleep(for: .seconds(ExposurePlan.meteringS - age))
      }
      guard let device = mutex.withLock({ device }) else { return }
      do {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        let format = device.activeFormat
        let plan = ExposurePlan.plan(
          meteredDurationS: device.exposureDuration.seconds, meteredISO: Double(device.iso),
          minDurationS: format.minExposureDuration.seconds,
          maxDurationS: format.maxExposureDuration.seconds,
          minISO: Double(format.minISO), maxISO: Double(format.maxISO))
        var mode = "auto"
        if plan.mode == "custom", device.isExposureModeSupported(.custom) {
          device.setExposureModeCustom(
            duration: CMTime(seconds: plan.durationS, preferredTimescale: 1_000_000),
            iso: Float(plan.iso), completionHandler: nil)
          mode = "custom"
        } else if device.isExposureModeSupported(.locked) {
          device.exposureMode = .locked
          mode = "locked"
        }
        var whiteBalance = "auto"
        if device.isWhiteBalanceModeSupported(.locked) {
          device.whiteBalanceMode = .locked
          whiteBalance = "locked"
        }
        let held = mode != "auto"
        let applied = CaptureExposureControl(
          mode: mode, whiteBalance: whiteBalance, durationS: held ? plan.durationS : nil,
          iso: held ? plan.iso : nil, shortfallEV: held ? plan.shortfallEV : nil)
        mutex.withLock { control = applied }
      } catch {
        mutex.withLock { control = CaptureExposureControl(mode: "auto", whiteBalance: "auto") }
      }
    }

    /// Back to continuous auto-exposure and white balance once a scan ends.
    @MainActor func releaseExposure() {
      guard let device = mutex.withLock({ device }), (try? device.lockForConfiguration()) != nil
      else { return }
      if device.isExposureModeSupported(.continuousAutoExposure) {
        device.exposureMode = .continuousAutoExposure
      }
      if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
        device.whiteBalanceMode = .continuousAutoWhiteBalance
      }
      device.unlockForConfiguration()
      mutex.withLock { control = CaptureExposureControl(mode: "auto", whiteBalance: "auto") }
    }

    /// One frame's exposure for the frame log. ISO and gains are the device's current values.
    func exposure(_ frame: ARFrame) -> CaptureExposure {
      let device = mutex.withLock { self.device }
      let gains = device?.deviceWhiteBalanceGains
      return CaptureExposure(
        durationS: frame.camera.exposureDuration > 0 ? frame.camera.exposureDuration : nil,
        offsetEV: Double(frame.camera.exposureOffset),
        iso: device.map { Double($0.iso) },
        whiteBalanceGains: gains.map {
          [Double($0.redGain), Double($0.greenGain), Double($0.blueGain)]
        })
    }

    /// Every delegate frame: update the blur cue, and early in a 4K session confirm that the
    /// depth ARKit pairs with it covers the same view as the image (FourKProbe).
    func observe(_ frame: ARFrame) {
      let k = frame.camera.intrinsics
      let depth = Self.centreDepth(frame)
      let image = frame.camera.imageResolution
      let map = frame.smoothedSceneDepth != nil ? frame.sceneDepth?.depthMap : nil
      let depthSize = map.map { (CVPixelBufferGetWidth($0), CVPixelBufferGetHeight($0)) }
      let rejected = mutex.withLock { () -> (CaptureFormatFallback, Bool)? in
        meter.update(
          time: frame.timestamp,
          transform: (0..<4).flatMap { row in
            (0..<4).map { Double(frame.camera.transform[$0][row]) }
          },
          focalPx: Double(k[0][0]), shutterS: frame.camera.exposureDuration, depthM: depth)
        guard format?.kind == "4k", probe.verdict == .pending else { return nil }
        let verdict = probe.observe(
          time: frame.timestamp, imageWidth: Int(image.width), imageHeight: Int(image.height),
          depthWidth: depthSize?.0, depthHeight: depthSize?.1)
        guard case .rejected(let reason, let persist) = verdict else { return nil }
        // The evidence of what ARKit gave at 4K, for the manifest and for mapping it later.
        let evidence = CaptureFormatFallback(
          reason: reason, imageWidth: Int(image.width), imageHeight: Int(image.height),
          depthWidth: depthSize?.0, depthHeight: depthSize?.1)
        sessionFallback = evidence
        return (evidence, persist)
      }
      guard let (evidence, persist) = rejected else { return }
      if persist, let data = try? JSONEncoder().encode(evidence) {
        UserDefaults.standard.set(data, forKey: Self.fourKFallbackKey)
      }
      onFourKRejected?()
    }

    /// The smoothed depth at the image centre, metres, when ARKit is confident of it.
    static func centreDepth(_ frame: ARFrame) -> Double? {
      guard let data = frame.smoothedSceneDepth, let confidence = data.confidenceMap else {
        return nil
      }
      let map = data.depthMap
      guard CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess else { return nil }
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      guard CVPixelBufferLockBaseAddress(confidence, .readOnly) == kCVReturnSuccess else {
        return nil
      }
      defer { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
      let x = CVPixelBufferGetWidth(map) / 2
      let y = CVPixelBufferGetHeight(map) / 2
      guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        let base = CVPixelBufferGetBaseAddress(map),
        let conf = CVPixelBufferGetBaseAddress(confidence),
        conf.advanced(by: y * CVPixelBufferGetBytesPerRow(confidence) + x)
          .load(as: UInt8.self) >= UInt8(ARConfidenceLevel.medium.rawValue)
      else { return nil }
      let value = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map) + x * 4)
        .load(as: Float32.self)
      return value.isFinite && value > 0 ? Double(value) : nil
    }
  }
#endif
