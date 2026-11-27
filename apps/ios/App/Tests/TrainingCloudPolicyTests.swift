import Foundation
import Testing
@testable import AppSupport

@Test func trainingCloudPollingStopsAtTerminalStatesAndSkipsBusyWork() {
  #expect(TrainingCloudPolicy.decision(busy: true, status: nil, failed: false) == .skip)
  #expect(TrainingCloudPolicy.decision(busy: false, status: nil, failed: false) == .fetch)
  #expect(TrainingCloudPolicy.decision(busy: false, status: "complete", failed: false) == .stop)
  #expect(TrainingCloudPolicy.decision(busy: true, status: "complete", failed: false) == .stop)
  #expect(TrainingCloudPolicy.decision(busy: false, status: "failed", failed: false) == .stop)
  #expect(TrainingCloudPolicy.decision(busy: false, status: "error", failed: false) == .stop)
  #expect(TrainingCloudPolicy.decision(busy: false, status: nil, failed: true) == .stop)
}

@Test func trainingCloudManifestOnlyChangesWhenRouteChanges() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let capture = LocalCapture(directory: root.appendingPathComponent("Captures/scan"), upload: nil)
  let record = CloudSplatRecord(siteID: UUID(), scanID: UUID())
  #expect(try record.saveIfChanged(for: capture))
  #expect(try !record.saveIfChanged(for: capture))
  #expect(try CloudSplatRecord(siteID: record.siteID, scanID: UUID()).saveIfChanged(for: capture))
}
