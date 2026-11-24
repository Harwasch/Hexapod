import Foundation

public enum LiveSplatMode: Sendable { case scanPreview, coverageOnly }
public enum LiveSplatStatus: Sendable, Equatable {
  case stopped, coverageOnly, scanPreview
}

public struct LiveSplatConfiguration: Sendable {
  public var mode: LiveSplatMode = .scanPreview
  public var maxSplats: Int = 262_144
  public var maxSeedsPerFrame: Int = 12_288
  public init() {}
  var bounded: Self {
    var c = self
    c.maxSplats = max(4, min(262_144, maxSplats) / 4 * 4)
    c.maxSeedsPerFrame = min(12_288, max(1, maxSeedsPerFrame))
    return c
  }
}

/// Depth-resolution, unrotated image. Depth is optical-axis metres, not ray distance.
/// RGB is sampled from capturedImage at matching normalized pixel coordinates.
public struct SplatFrame: Sendable {
  public let timestamp: Double
  public let camera: SplatCamera
  public let width: Int, height: Int
  public let depths: [Float]
  public let confidence: [UInt8]
  public let colors: [SplatVector]
  public init(
    timestamp: Double, camera: SplatCamera, width: Int, height: Int,
    depths: [Float], confidence: [UInt8], colors: [SplatVector]
  ) throws {
    guard timestamp.isFinite, width > 0, height > 0, width <= 512, height <= 512,
      depths.count == width * height, confidence.count == depths.count,
      colors.count == depths.count,
      colors.allSatisfy({ $0.values.allSatisfy { $0.isFinite && (0...1).contains($0) } })
    else {
      throw LiveSplatError.invalidFrame
    }
    self.timestamp = timestamp
    self.camera = camera
    self.width = width
    self.height = height
    self.depths = depths
    self.confidence = confidence
    self.colors = colors
  }
  public func seed(x: Int, y: Int) -> Gaussian? {
    guard (0..<width).contains(x), (0..<height).contains(y) else { return nil }
    let i = y * width + x
    let d = Double(depths[i])
    guard confidence[i] == 2, d.isFinite, d >= 0.1, d <= 8 else { return nil }
    let u = Double(x) * Double(camera.width) / Double(width)
    let v = Double(y) * Double(camera.height) / Double(height)
    let footprint =
      d
      * max(
        Double(camera.width) / Double(width) / camera.fx,
        Double(camera.height) / Double(height) / camera.fy)
    return try? Gaussian(
      position: camera.unproject(u: u, v: v, depth: d),
      scale: SplatVector(footprint, footprint, footprint) * 0.5, color: colors[i])
  }
}

/// Current-view coverage, not a claim about unseen world surfaces. Only a projected
/// centre with a matching measured depth covers a cell (10 cm or 5% depth gate).
public struct CoverageGrid: Sendable {
  public let columns: Int, rows: Int
  public private(set) var covered: [Bool]
  public init(columns: Int = 32, rows: Int = 24) {
    precondition(columns > 0 && rows > 0 && columns <= 512 && rows <= 512)
    self.columns = columns
    self.rows = rows
    covered = Array(repeating: false, count: columns * rows)
  }
  public var fraction: Double { Double(covered.filter { $0 }.count) / Double(covered.count) }
  /// One uncovered cell in the visible camera crop, nearest its centre. Coverage is
  /// view-local; the app renders a single hint instead of painting a debug grid.
  public func gapHint(displayTransform t: [Double]) -> SplatVector? {
    guard t.count == 6, t.allSatisfy({ $0.isFinite }) else { return nil }
    var nearest: SplatVector?
    var distance = Double.infinity
    for index in covered.indices where !covered[index] {
      let u = (Double(index % columns) + 0.5) / Double(columns)
      let v = (Double(index / columns) + 0.5) / Double(rows)
      let x = t[0] * u + t[2] * v + t[4]
      let y = t[1] * u + t[3] * v + t[5]
      guard (0.1...0.9).contains(x), (0.2...0.75).contains(y) else { continue }
      let d = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
      if d < distance {
        nearest = SplatVector(x, y, 0)
        distance = d
      }
    }
    return nearest
  }
  public func cell(u: Double, v: Double, width: Int, height: Int) -> Int? {
    guard u.isFinite, v.isFinite, width > 0, height > 0,
      u >= 0, v >= 0, u < Double(width), v < Double(height)
    else { return nil }
    return min(rows - 1, Int(v * Double(rows) / Double(height))) * columns
      + min(columns - 1, Int(u * Double(columns) / Double(width)))
  }
  mutating func mark(_ cell: Int) { covered[cell] = true }
  mutating func observe(_ g: Gaussian, in frame: SplatFrame) -> Bool {
    guard g.isValid, g.opacity >= 0.05, let p = frame.camera.project(g.position) else {
      return false
    }
    // Quantize to the nearest depth pixel, then classify its centre. Seeding
    // at integer pixels otherwise lands exactly on coverage-cell boundaries,
    // where projection roundoff can repeatedly seed the adjacent empty cell.
    let px = (p.x * Double(frame.width) / Double(frame.camera.width)).rounded()
    let py = (p.y * Double(frame.height) / Double(frame.camera.height)).rounded()
    guard px >= 0, py >= 0, px < Double(frame.width), py < Double(frame.height) else {
      return false
    }
    let x = Int(px)
    let y = Int(py)
    guard let cell = cell(u: px + 0.5, v: py + 0.5, width: frame.width, height: frame.height) else {
      return false
    }
    let i = y * frame.width + x
    let measured = Double(frame.depths[i])
    guard frame.confidence[i] == 2, measured.isFinite, measured >= 0.1, measured <= 8,
      abs(measured - p.z) <= max(0.1, 0.05 * measured)
    else { return false }
    mark(cell)
    return true
  }
}

/// UI facade and offline reference. Live frames go directly to ScanPreviewProcessor;
/// this replay method is never used by the camera delegate or GPU completion.
@MainActor
public final class DefaultLiveSplatRenderer: LiveSplatRenderer, LiveSplatPreferenceObserver {
  public let preference: LiveSplatPreference
  public let configuration: LiveSplatConfiguration
  public private(set) var status: LiveSplatStatus = .stopped
  public private(set) var splats: [Gaussian] = []
  public private(set) var coverage = CoverageGrid()
  public var coverageFraction: Double { liveCoverage ?? coverage.fraction }
  private var liveCoverage: Double?
  public private(set) var latestFrame: SplatFrame?
  public private(set) var displayTransform: [Double] = [1, 0, 0, 1, 0, 0]
  private(set) var sessionGeneration: UInt64 = 0
  private var reference: SurfelHash?
  private var gaussianReference: GaussianVoxelHash?
  private var referenceStyle: ScanPreviewStyle = .discs
  private var epoch: UInt32 = 0
  #if canImport(RealityKit) && os(iOS)
    public private(set) var processor: ScanPreviewProcessor?
  #endif
  public init(
    configuration: LiveSplatConfiguration = .init(), preference: LiveSplatPreference = .shared
  ) {
    self.preference = preference
    self.configuration = configuration.bounded
    preference.observe(self)
  }
  public func start() {
    sessionGeneration &+= 1
    reference = nil
    gaussianReference = nil
    referenceStyle = preference.previewStyle
    splats = []
    latestFrame = nil
    coverage = CoverageGrid()
    liveCoverage = nil
    epoch = 0
    status = .scanPreview
    preferenceChanged()
    #if canImport(RealityKit) && os(iOS)
      processor?.reset()
    #endif
  }
  public func stop() {
    status = .stopped
    #if canImport(RealityKit) && os(iOS)
      processor?.setActive(false, enabled: preference.previewEnabled)
    #endif
  }
  public func trackingUnavailable() {
    guard status != .stopped else { return }
    latestFrame = nil
    coverage = CoverageGrid()
    liveCoverage = nil
  }
  public func ingest(_ frame: SplatFrame) {
    guard status != .stopped, latestFrame.map({ frame.timestamp > $0.timestamp }) ?? true else {
      return
    }
    if referenceStyle != preference.previewStyle {
      reference = nil
      gaussianReference = nil
      referenceStyle = preference.previewStyle
    }
    if preference.previewStyle == .splat {
      ingestGaussians(frame)
      return
    }
    if reference == nil { reference = SurfelHash(capacity: configuration.maxSplats) }
    epoch &+= 1
    let count = frame.width * frame.height
    for offset in 0..<min(count, configuration.maxSeedsPerFrame) {
      let i = (offset * 997 + Int(epoch) * 131) % count
      guard let seed = frame.seed(x: i % frame.width, y: i / frame.width) else { continue }
      let p = seed.position
      let c = seed.color
      let depth = frame.depths[i]
      let focal = Float(frame.camera.fx * Double(frame.width) / Double(frame.camera.width))
      reference!.fuse(
        position: SIMD3(Float(p.x), Float(p.y), Float(p.z)),
        color: SIMD3(Float(c.x), Float(c.y), Float(c.z)), depth: depth,
        confidence: frame.confidence[i],
        radius: SurfelHash.radius(depth: depth, focalPixels: focal), epoch: epoch)
    }
    splats = reference!.cells.compactMap { cell in
      guard let cell else { return nil }
      let p = cell.position
      let c = cell.color
      let r = Double(cell.radius)
      return try? Gaussian(
        position: SplatVector(Double(p.x), Double(p.y), Double(p.z)),
        scale: SplatVector(r, r, r), color: SplatVector(Double(c.x), Double(c.y), Double(c.z)))
    }
    var grid = CoverageGrid()
    for g in splats { _ = grid.observe(g, in: frame) }
    coverage = grid
    latestFrame = frame
  }
  private func ingestGaussians(_ frame: SplatFrame) {
    if gaussianReference == nil {
      gaussianReference = GaussianVoxelHash(capacity: configuration.maxSplats)
    }
    epoch &+= 1
    let count = frame.width * frame.height
    for offset in 0..<min(count, configuration.maxSeedsPerFrame) {
      let i = (offset * 997 + Int(epoch) * 131) % count
      guard let seed = GaussianSeed.surface(frame: frame, x: i % frame.width, y: i / frame.width)
      else { continue }
      gaussianReference!.fuse(
        seed, depth: frame.depths[i], confidence: frame.confidence[i], epoch: epoch)
    }
    splats = gaussianReference!.seeds.indices.compactMap { gaussianReference!.gaussian(at: $0) }
    var grid = CoverageGrid()
    for g in splats { _ = grid.observe(g, in: frame) }
    coverage = grid
    latestFrame = frame
  }
  func preferenceChanged() {
    #if canImport(RealityKit) && os(iOS)
      processor?.setStyle(splat: preference.previewStyle == .splat, inset: preference.insetEnabled)
    #endif
    guard status != .stopped else { return }
    status =
      preference.previewEnabled && configuration.mode == .scanPreview ? .scanPreview : .coverageOnly
    #if canImport(RealityKit) && os(iOS)
      processor?.setActive(true, enabled: status == .scanPreview)
    #endif
  }
  #if canImport(RealityKit) && os(iOS)
    public func preparePreview() throws -> ScanPreviewProcessor {
      if let processor { return processor }
      let value = try ScanPreviewProcessor()
      processor = value
      value.setStyle(splat: preference.previewStyle == .splat, inset: preference.insetEnabled)
      return value
    }
    public func pollSummary() { liveCoverage = processor?.coverageFraction ?? 0 }
  #endif
  public func exportPLY(to captureBundleFolder: URL) throws -> URL {
    guard status == .stopped else { throw LiveSplatError.exportWhileRunning }
    guard
      !FileManager.default.fileExists(
        atPath: captureBundleFolder.appendingPathComponent("manifest.json").path)
    else {
      throw LiveSplatError.bundleFinalized
    }
    let url = captureBundleFolder.appendingPathComponent("livesplat.ply")
    try SplatPLY.encode(splats).write(to: url, options: .atomic)
    return url
  }
}

public enum SplatPLY {
  public static func encode(_ splats: [Gaussian]) throws -> Data {
    guard splats.allSatisfy(\.isValid) else { throw LiveSplatError.invalidGaussian }
    // Standard 3DGS: log scales, opacity logit, normalized Hamilton wxyz, SH DC.
    let properties = [
      "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    var data = Data(
      ("ply\nformat binary_little_endian 1.0\ncomment ARKit world metres Y up\nelement vertex \(splats.count)\n"
        + properties.map { "property float \($0)\n" }.joined() + "end_header\n").utf8)
    let sh0 = 0.28209479177387814  // 1 / sqrt(4*pi)
    for splat in splats {
      var g = splat
      g.constrain()
      let p = g.parameters
      var values = Array(p[0..<3])
      values += [0, 0, 0]
      values += p[11..<14].map { ($0 - 0.5) / sh0 }
      values += [p[10]]
      values += Array(p[3..<6])
      values += Array(p[6..<10])
      for value in values {
        guard Float(value).isFinite else { throw LiveSplatError.invalidGaussian }
        var bits = Float(value).bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
      }
    }
    return data
  }
}
