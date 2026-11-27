import Foundation

/// Training scratch files have no relaunch consumer. A cooling/paused owner still
/// needs them; a closed terminal owner or a durably saved result does not.
enum TrainingOutputPolicy {
  static func removalURL(for capture: LocalCapture, stage: TrainingStage,
    sessionClosed: Bool, resultSaved: Bool = false) -> URL? {
    guard sessionClosed else { return nil }
    let terminal = stage == .failed || stage == .cancelled || stage == .settling || stage == .result
    guard resultSaved || terminal else { return nil }
    return PhoneSplatRecord.directory(for: capture).appendingPathComponent("Training")
  }
}
