import Foundation

public enum LiveSplatMode: Sendable { case automatic, lidarPreview, experimentalGaussian }
public enum LiveSplatStatus: Sendable, Equatable {
  case stopped, lidarPreview, experimentalGaussian
  case fallback(String)
}

public struct LiveSplatConfiguration: Sendable {
  public var mode: LiveSplatMode = .automatic
  public var maxSplats: Int = 8_192
  public var maxSeedsPerFrame: Int = 128
  public var maxAgeFrames: Int = 600
  public var gpuBudgetMilliseconds: Double = 8
  public init() {}
  var bounded: Self {
    var c = self
    c.maxSplats = min(32_768, max(1, maxSplats))
    c.maxSeedsPerFrame = min(256, max(1, maxSeedsPerFrame))
    c.maxAgeFrames = min(3_600, max(1, maxAgeFrames))
    c.gpuBudgetMilliseconds =
      gpuBudgetMilliseconds.isFinite ? min(8, max(1, gpuBudgetMilliseconds)) : 8
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

/// Cannot preempt a Metal command buffer. The budget is a feedback target: a single
/// overrun disables training/rendering and selects preview on subsequent frames.
public struct SplatBudget: Sendable {
  public let milliseconds: Double
  public private(set) var exceeded = false
  public init(milliseconds: Double = 8) { self.milliseconds = milliseconds }
  public mutating func record(seconds: Double) {
    if !seconds.isFinite || seconds <= 0 || seconds * 1_000 > milliseconds { exceeded = true }
  }
  public var optimizationSteps: Int { exceeded ? 0 : 2 }
}

/// Main actor owns the state. Feed frames from the existing ARSession delegate;
/// this module never replaces that delegate or runs/resets the shared session.
@MainActor public final class DefaultLiveSplatRenderer: LiveSplatRenderer {
  public let configuration: LiveSplatConfiguration
  public private(set) var status: LiveSplatStatus = .stopped
  public private(set) var splats: [Gaussian] = []
  public private(set) var coverage = CoverageGrid()
  public var coverageFraction: Double { coverage.fraction }
  public private(set) var latestFrame: SplatFrame?
  public private(set) var recentKeyframes: [SplatFrame] = []
  public private(set) var budget: SplatBudget
  public private(set) var generation: UInt64 = 0
  public private(set) var displayTransform: [Double] = [1, 0, 0, 1, 0, 0]
  func setDisplayTransform(_ value: [Double]) { displayTransform = value }
  private(set) var splatIDs: [UInt64] = []
  private(set) var sessionGeneration: UInt64 = 0
  private var nextID: UInt64 = 0
  public func trackingUnavailable() {
    guard status != .stopped else { return }
    latestFrame = nil
    recentKeyframes = []
    coverage = CoverageGrid()
  }
  private var lastSeen: [Int] = []
  private var sequence = 0
  public init(configuration: LiveSplatConfiguration = .init()) {
    self.configuration = configuration.bounded
    budget = SplatBudget(milliseconds: configuration.bounded.gpuBudgetMilliseconds)
  }
  /// Starts a fresh local AR world. Call stop/start after tracking resets/relocalization.
  public func start() {
    generation &+= 1
    sessionGeneration &+= 1
    nextID = 0
    splatIDs = []
    splats = []
    lastSeen = []
    recentKeyframes = []
    latestFrame = nil
    sequence = 0
    coverage = CoverageGrid()
    budget = SplatBudget(milliseconds: configuration.gpuBudgetMilliseconds)
    status = .lidarPreview
  }
  public func stop() {
    status = .stopped
    generation &+= 1
    sessionGeneration &+= 1
  }
  public func ingest(_ frame: SplatFrame) {
    guard status != .stopped, latestFrame.map({ frame.timestamp > $0.timestamp }) ?? true else {
      return
    }
    generation &+= 1
    sequence += 1
    var grid = CoverageGrid()
    var retained: [Gaussian] = []
    var seen: [Int] = []
    var retainedIDs: [UInt64] = []
    for (i, g) in splats.enumerated() where g.isValid && g.opacity >= 0.05 {
      let visible = grid.observe(g, in: frame)
      let age = visible ? sequence : lastSeen[i]
      if sequence - age <= configuration.maxAgeFrames {
        retained.append(g)
        seen.append(age)
        retainedIDs.append(splatIDs[i])
      }
    }
    splats = retained
    lastSeen = seen
    splatIDs = retainedIDs
    var added = 0
    // Rotate the scan origin so a capped insertion pass does not always favour the top left.
    let count = frame.width * frame.height
    for offset in 0..<count {
      if added >= configuration.maxSeedsPerFrame { break }
      let i = (offset + sequence * 997) % count
      let x = i % frame.width
      let y = i / frame.width
      guard
        let cell = grid.cell(
          u: Double(x) + 0.5, v: Double(y) + 0.5, width: frame.width, height: frame.height),
        !grid.covered[cell], let seed = frame.seed(x: x, y: y)
      else { continue }
      if splats.count >= configuration.maxSplats { break }
      splats.append(seed)
      lastSeen.append(sequence)
      splatIDs.append(nextID)
      nextID &+= 1
      grid.mark(cell)
      added += 1
    }
    coverage = grid
    latestFrame = frame
    if recentKeyframes.last.map({ frame.timestamp - $0.timestamp >= 0.2 }) ?? true {
      recentKeyframes.append(frame)
      if recentKeyframes.count > 3 { recentKeyframes.removeFirst() }
    }
  }
  func activateGaussian() { if status != .stopped { status = .experimentalGaussian } }
  func fallback(_ reason: String) { if status != .stopped { status = .fallback(reason) } }
  func acceptGPU(_ updated: [Gaussian], ids: [UInt64], seconds: Double, session: UInt64) {
    guard session == sessionGeneration, status == .experimentalGaussian else { return }
    // Record even when the camera advanced: stale view snapshots still consumed GPU time.
    budget.record(seconds: seconds)
    if budget.exceeded {
      fallback("GPU budget exceeded")
      return
    }
    guard ids.count == updated.count else { return }
    let indices = Dictionary(
      uniqueKeysWithValues: splatIDs.enumerated().map { ($0.element, $0.offset) })
    for (i, id) in ids.enumerated() {
      if let current = indices[id], updated[i].isValid { splats[current] = updated[i] }
    }
  }
  public func exportPLY(to captureBundleFolder: URL) throws -> URL {
    guard status == .stopped else { throw LiveSplatError.exportWhileRunning }
    guard
      !FileManager.default.fileExists(
        atPath: captureBundleFolder.appendingPathComponent("manifest.json").path)
    else {
      throw LiveSplatError.bundleFinalized
    }
    let url = captureBundleFolder.appendingPathComponent("live_splat.ply")
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
