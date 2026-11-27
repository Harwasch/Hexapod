import Foundation
import Testing
@testable import AppSupport

@Test func trainingOutputCleanupPreservesActiveWorkAndCapture() {
  let capture = LocalCapture(directory: URL(fileURLWithPath: "/store/Captures/scan"), upload: nil)
  for stage in [TrainingStage.preparing, .training, .paused, .cooling] {
    #expect(TrainingOutputPolicy.removalURL(for: capture, stage: stage, sessionClosed: true) == nil)
  }
  for stage in [TrainingStage.failed, .cancelled, .settling, .result] {
    #expect(TrainingOutputPolicy.removalURL(for: capture, stage: stage, sessionClosed: false) == nil)
    let output = TrainingOutputPolicy.removalURL(for: capture, stage: stage, sessionClosed: true)
    #expect(output == PhoneSplatRecord.directory(for: capture).appendingPathComponent("Training"))
    #expect(output != capture.directory)
  }
  #expect(TrainingOutputPolicy.removalURL(for: capture, stage: .training,
    sessionClosed: true, resultSaved: true) != nil)
}
