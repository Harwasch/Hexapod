import Foundation

/// App-facing seam shared by the numerical renderer and the deterministic UI fake.
@MainActor public protocol LiveSplatRenderer: AnyObject {
  var status: LiveSplatStatus { get }
  var coverageFraction: Double { get }
  var splats: [Gaussian] { get }
  func start()
  func stop()
  func ingest(_ frame: SplatFrame)
  func trackingUnavailable()
  func exportPLY(to captureBundleFolder: URL) throws -> URL
}

public enum SplatFrameEvent: Sendable {
  case frame(SplatFrame)
  case trackingUnavailable
}

extension LiveSplatRenderer {
  public func ingest(_ event: SplatFrameEvent) {
    switch event {
    case .frame(let frame): ingest(frame)
    case .trackingUnavailable: trackingUnavailable()
    }
  }
}

/// Pull-driven: no camera, timer, or background task. nil means end of replay.
/// Callers serialize next/reset; async allows optional Apple video decoding.
@MainActor public protocol SplatFrameSource: AnyObject {
  func next() async throws -> SplatFrameEvent?
  func reset()
}

@MainActor public final class FakeFrameSource: SplatFrameSource {
  private let events: [SplatFrameEvent]
  private var cursor = 0
  public init(events: [SplatFrameEvent]) { self.events = events }
  /// Small stationary plane for previews that have no fixture URL or camera.
  public static func plane(frameCount: Int = 16) throws -> FakeFrameSource {
    let camera = try SplatCamera(fx: 32, fy: 32, cx: 16, cy: 12, width: 32, height: 24)
    let events = try (0..<min(120, max(0, frameCount))).map { index in
      SplatFrameEvent.frame(
        try SplatFrame(
          timestamp: Double(index) / 12, camera: camera, width: 32, height: 24,
          depths: Array(repeating: 2, count: 768), confidence: Array(repeating: 2, count: 768),
          colors: Array(repeating: BundleReplayFrameSource.flatColor, count: 768)))
    }
    return FakeFrameSource(events: events)
  }
  public func reset() { cursor = 0 }
  public func next() async throws -> SplatFrameEvent? {
    guard cursor < events.count else { return nil }
    defer { cursor += 1 }
    return events[cursor]
  }
}

/// Simulated coverage: each accepted timestamp adds a fixed number of cells.
/// Does not seed geometry or run Metal; export is an empty, valid PLY for UI tests.
@MainActor public final class FakeLiveSplat: LiveSplatRenderer, LiveSplatPreferenceObserver {
  public private(set) var status: LiveSplatStatus = .stopped
  public private(set) var coverage = CoverageGrid()
  public var coverageFraction: Double { coverage.fraction }
  public let splats: [Gaussian] = []
  public let preference: LiveSplatPreference
  private let cellsPerFrame: Int
  private var lastTimestamp: Double?
  public init(cellsPerFrame: Int = 64, preference: LiveSplatPreference = .shared) {
    self.preference = preference
    self.cellsPerFrame = min(768, max(1, cellsPerFrame))
    preference.observe(self)
  }
  public func start() {
    coverage = CoverageGrid()
    lastTimestamp = nil
    status = .scanPreview
    preferenceChanged()
  }
  func preferenceChanged() {
    guard status != .stopped else { return }
    status = preference.previewEnabled ? .scanPreview : .coverageOnly
  }
  public func stop() { status = .stopped }
  public func ingest(_ frame: SplatFrame) {
    guard status != .stopped, lastTimestamp.map({ frame.timestamp > $0 }) ?? true else { return }
    lastTimestamp = frame.timestamp
    let start = coverage.covered.filter { $0 }.count
    for cell in start..<min(coverage.covered.count, start + cellsPerFrame) { coverage.mark(cell) }
  }
  public func trackingUnavailable() {
    guard status != .stopped else { return }
    coverage = CoverageGrid()
    lastTimestamp = nil
  }
  public func exportPLY(to captureBundleFolder: URL) throws -> URL {
    guard status == .stopped else { throw LiveSplatError.exportWhileRunning }
    guard
      !FileManager.default.fileExists(
        atPath: captureBundleFolder.appendingPathComponent("manifest.json").path)
    else { throw LiveSplatError.bundleFinalized }
    let url = captureBundleFolder.appendingPathComponent("livesplat.ply")
    try SplatPLY.encode(splats).write(to: url, options: .atomic)
    return url
  }
}
