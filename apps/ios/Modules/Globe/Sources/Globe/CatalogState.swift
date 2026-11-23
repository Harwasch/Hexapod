import Foundation

/// A single property opens directly; multiple properties need a deliberate choice.
public func globeInitialRoute(sites: [GlobeSite]) -> GlobeRoute? {
  sites.count == 1 ? GlobeRoute(siteID: sites[0].id) : nil
}

/// Public catalog read, including the site assigned by reconstruction after upload.
/// An upload's queued job alone is never evidence that its scan is ready.
public struct CatalogCaptureState: Decodable, Sendable {
  public let status: String
  public let siteId: UUID?
  public let jobs: [Job]?

  public struct Job: Decodable, Sendable {
    let status: String
    let createdAt: String
    let error: String?
    let steps: [Step]
  }
  public struct Step: Decodable, Sendable {
    let stageId: String
    let status: String
    let ordinal: Int
    let metrics: Metrics
  }
  public struct Metrics: Decodable, Sendable {
    let progress: Progress?
    let preparing: Bool
    enum CodingKeys: String, CodingKey { case progress, preparing }
    public init(from decoder: any Decoder) throws {
      let values = try decoder.container(keyedBy: CodingKeys.self)
      progress = try? values.decode(Progress.self, forKey: .progress)
      preparing = (try? values.decode(Bool.self, forKey: .preparing)) ?? false
    }
  }
  public struct Progress: Decodable, Sendable {
    let done: Double
    let total: Double
    var percent: Int? {
      guard done.isFinite, total.isFinite, total > 0, done >= 0, done <= total else { return nil }
      return Int(done / total * 100)
    }
  }
  private var latestJob: Job? { jobs?.max { $0.createdAt < $1.createdAt } }
  public var fullError: String? { latestJob?.error }
  private var activeStep: Step? {
    guard let job = latestJob, job.status == "in-progress" else { return nil }
    return job.steps.filter { $0.status == "in-progress" }.min { $0.ordinal < $1.ordinal }
  }
  public var label: String {
    if latestJob?.status == "cancelled" { return "Failed · Processing cancelled" }
    if status == "error" || latestJob?.status == "error" {
      let step = latestJob?.steps.first { $0.status == "error" }?.stageId
        ?? ["normalize", "pose", "train"].first { fullError?.contains("stage '\($0)'") == true }
      let reason: String?
      switch step {
      case "normalize": reason = "Could not prepare capture"
      case "pose": reason = "Camera alignment failed"
      case "train": reason = "Training failed"
      default: reason = fullError == nil ? nil : "Processing failed"
      }
      return reason.map { "Failed · \($0)" } ?? "Failed"
    }
    switch status {
    case "awaiting-files", "not-started", "in-progress":
      guard let step = activeStep, step.stageId == "train", !step.metrics.preparing else { return "Processing" }
      return step.metrics.progress?.percent.map { "Training \($0)%" } ?? "Training"
    case "complete": return "Ready"
    default: return "Connecting…"
    }
  }
  public static func fetch(
    captureID: UUID, baseURL: URL,
    transport: any GlobeTransport = URLSessionGlobeTransport()
  ) async throws -> Self {
    try JSONDecoder().decode(
      Self.self, from: await transport.get(
        baseURL.appendingPathComponent("captures/\(captureID.uuidString)")))
  }
}
