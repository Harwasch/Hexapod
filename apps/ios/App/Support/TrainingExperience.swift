import Foundation

/// The complete public UI contract, shared with portable tests and UI automation.
enum TrainingID {
  static let start = "train.start"
  static let quick = "train.quick"
  static let best = "train.best"
  static let begin = "train.begin"
  static let pause = "train.pause"
  static let resume = "train.resume"
  static let cancel = "train.cancel"
  static let retry = "train.retry"
  static let progress = "train.progress"
  static let sharpness = "train.sharpness"
  static let thermal = "train.thermal"
  static let close = "result.close"
  static let orbit = "result.orbit"
  static let focus = "result.focus"
  static let share = "result.share"
  static let shareCloud = "result.shareCloud"
  static let cloudRetry = "result.cloudRetry"
  static let phone = "result.phone"
  static let cloud = "result.cloud"
  static let inPlace = "inplace.open"
  static let camera = "inplace.camera"
  static let splat = "inplace.splat"
  static let leave = "inplace.close"
  static let refine = "refine.start"
  static let confirm = "refine.confirm"
  static let decline = "refine.cancel"
  static let refineRetry = "refine.retry"
  static let refineProgress = "refine.progress"
  static let phoneBadge = "captures.badge.phone"
  static let cloudBadge = "captures.badge.cloud"
  static let all = [start, quick, best, begin, pause, resume, cancel, retry, progress, sharpness,
    thermal, close, orbit, focus, share, shareCloud, cloudRetry, phone, cloud, inPlace, camera, splat, leave,
    refine, confirm, decline, refineRetry, refineProgress, phoneBadge, cloudBadge]
}

enum TrainingStage: Equatable, Sendable {
  case choosing, preparing, training, paused, cooling, settling, result, failed, cancelled
  var busy: Bool {
    switch self {
    case .preparing, .training, .paused, .cooling, .settling: true
    default: false
    }
  }
}

struct TrainingExperience: Sendable {
  private(set) var stage = TrainingStage.choosing
  mutating func begin() { if !stage.busy { stage = .preparing } }
  mutating func prepared() { if stage == .preparing { stage = .training } }
  mutating func pause() { if stage == .training { stage = .paused } }
  mutating func cool() { if stage.busy { stage = .cooling } }
  mutating func resume() {
    if stage == .paused || stage == .cooling { stage = .training }
  }
  mutating func finish() { if stage == .training { stage = .settling } }
  mutating func showResult() { stage = .result }
  mutating func fail() { stage = .failed }
  mutating func cancel() { stage = .cancelled }
}

enum TrainingPresentation {
  /// Display heuristic, not accuracy: held-out PSNR 15 dB → 0, 35 dB → 100,
  /// linear between, clamped outside. Missing/nonfinite measurements stay unknown.
  /// This is the only PSNR-to-Sharpness mapping in the product.
  static func sharpness(psnr: Float?) -> Int? {
    guard let psnr, psnr.isFinite else { return nil }
    let fraction: Float = (psnr - 15) / 20
    return Int((max(0, min(1, fraction)) * 100).rounded())
  }
  static func fraction(step: Int, budget: Int?, elapsed: Double, remaining: Double) -> Double {
    if let budget, budget > 0 { return min(1, max(0, Double(step) / Double(budget))) }
    guard elapsed.isFinite, remaining.isFinite, elapsed >= 0, remaining >= 0 else { return 0 }
    let total = elapsed + remaining
    guard total.isFinite, total > 0 else { return 0 }
    return elapsed / total
  }
  static func eta(_ seconds: Double?) -> String {
    guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
    let minutes = Int(ceil(min(seconds, 86_400) / 60))
    return minutes < 1 ? "<1 min" : "~\(minutes) min"
  }
}

struct SplatSelection: Sendable {
  enum Source: Sendable { case phone, cloud }
  private(set) var source = Source.phone
  private(set) var cloudAvailable = false
  mutating func cloudReady() {
    guard !cloudAvailable else { return }
    cloudAvailable = true
    source = .cloud
  }
  mutating func select(_ value: Source) {
    if value == .phone || cloudAvailable { source = value }
  }
}

struct CaptureQuality: Equatable, Sendable {
  let phone: Bool
  let cloud: Bool
  var symbols: [String] {
    var result: [String] = []
    if phone { result.append("iphone") }
    if cloud { result.append("cloud.fill") }
    return result
  }
}

/// Stored outside the immutable upload bundle. The manifest is the atomic commit
/// point, so a crash while copying a new PLY leaves the previous result usable.
struct PhoneSplatRecord: Codable, Sendable {
  let file: String
  let sharpness: Int?
  let checkpoint: String?
  let previewFile: String?
  let simulated: Bool?
  static func directory(for capture: LocalCapture) -> URL {
    capture.directory.deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("PhoneResults/\(capture.id)")
  }
  static func load(for capture: LocalCapture) -> Self? {
    let directory = directory(for: capture)
    guard let data = try? Data(contentsOf: directory.appendingPathComponent("result.json")),
      let record = try? JSONDecoder().decode(Self.self, from: data),
      record.file == URL(fileURLWithPath: record.file).lastPathComponent,
      FileManager.default.fileExists(atPath: directory.appendingPathComponent(record.file).path)
    else { return nil }
    return record
  }
  func url(for capture: LocalCapture) -> URL { Self.directory(for: capture).appendingPathComponent(file) }
  func previewURL(for capture: LocalCapture) -> URL? {
    guard let previewFile, previewFile == URL(fileURLWithPath: previewFile).lastPathComponent else { return nil }
    return Self.directory(for: capture).appendingPathComponent(previewFile)
  }
  static func save(ply: URL, sharpness: Int?, checkpoint: URL?, for capture: LocalCapture,
    previewPNG: Data? = nil, simulated: Bool = false) throws -> Self {
    let directory = directory(for: capture)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = "phone-\(UUID().uuidString).ply"
    let destination = directory.appendingPathComponent(file)
    let previous = load(for: capture)
    try FileManager.default.copyItem(at: ply, to: destination)
    var previewFile: String?
    if previewPNG != nil { previewFile = "preview-\(UUID().uuidString).png" }
    let record = Self(file: file, sharpness: sharpness, checkpoint: checkpoint?.path,
      previewFile: previewFile, simulated: simulated)
    do {
      if let previewPNG, let url = record.previewURL(for: capture) { try previewPNG.write(to: url, options: .atomic) }
      try JSONEncoder().encode(record).write(to: directory.appendingPathComponent("result.json"), options: .atomic)
    } catch {
      try? FileManager.default.removeItem(at: destination)
      if let url = record.previewURL(for: capture) { try? FileManager.default.removeItem(at: url) }
      throw error
    }
    if let previous {
      try? FileManager.default.removeItem(at: previous.url(for: capture))
      if let url = previous.previewURL(for: capture) { try? FileManager.default.removeItem(at: url) }
    }
    return record
  }
}

/// A confirmed request is durable before bytes move. No tokens or signed URLs.
/// Server refinement remains idempotent through the existing Upload endpoint.
enum CloudRefineIntent {
  static func url(for capture: LocalCapture) -> URL {
    capture.uploadDirectory.appendingPathComponent("refine-intent.json")
  }
  static func pending(for capture: LocalCapture) -> Bool {
    FileManager.default.fileExists(atPath: url(for: capture).path)
  }
  static func confirm(for capture: LocalCapture) throws {
    try FileManager.default.createDirectory(at: capture.uploadDirectory, withIntermediateDirectories: true)
    try Data("{\"confirmed\":true}".utf8).write(to: url(for: capture), options: .atomic)
  }
  static func complete(for capture: LocalCapture) throws {
    if pending(for: capture) { try FileManager.default.removeItem(at: url(for: capture)) }
  }
}

struct CloudSplatRecord: Codable, Sendable {
  let siteID: UUID
  let scanID: UUID
  private static func url(for capture: LocalCapture) -> URL {
    PhoneSplatRecord.directory(for: capture).appendingPathComponent("cloud.json")
  }
  static func load(for capture: LocalCapture) -> Self? {
    guard let data = try? Data(contentsOf: url(for: capture)) else { return nil }
    return try? JSONDecoder().decode(Self.self, from: data)
  }
  func save(for capture: LocalCapture) throws {
    try FileManager.default.createDirectory(at: PhoneSplatRecord.directory(for: capture),
      withIntermediateDirectories: true)
    try JSONEncoder().encode(self).write(to: Self.url(for: capture), options: .atomic)
  }
  @discardableResult func saveIfChanged(for capture: LocalCapture) throws -> Bool {
    if let previous = Self.load(for: capture), previous.siteID == siteID, previous.scanID == scanID {
      return false
    }
    try save(for: capture)
    return true
  }
  /// The public scan viewer addresses a site with a fragment. The native globe
  /// route additionally selects scanID for exact capture selection.
  func shareURL(webBaseURL: URL?) -> URL? {
    guard let webBaseURL else { return nil }
    var components = URLComponents(url: webBaseURL.appendingPathComponent("view.html"),
      resolvingAgainstBaseURL: false)
    components?.fragment = siteID.uuidString.lowercased()
    return components?.url
  }
}

/// Pausing a retained session preserves its world; a new session or explicit reset
/// never inherits the old capture's coordinates. ObjectIdentifier is value-only.
struct SplatWorldOrigin: Sendable {
  private var capture: URL?
  private var session: ObjectIdentifier?
  mutating func record(capture: URL, session: ObjectIdentifier) {
    self.capture = capture
    self.session = session
  }
  mutating func invalidate() { capture = nil; session = nil }
  func permits(capture: URL, session: ObjectIdentifier) -> Bool {
    self.capture == capture && self.session == session
  }
}
