import Foundation
import XCTest

@testable import PhoneTrain

final class TrainingPolicyTests: XCTestCase {
  func testBoundaryControlAndCriticalLatch() {
    var control = TrainingControl()
    XCTAssertEqual(control.decision(thermal: 0), .step)
    XCTAssertEqual(control.decision(thermal: 2), .throttle)
    control.active = false
    XCTAssertEqual(control.decision(thermal: 0), .pause)
    XCTAssertEqual(control.decision(thermal: 3), .critical)
    control.active = true
    control.manuallyPaused = true
    XCTAssertEqual(control.decision(thermal: 0), .pause)
    control.manuallyPaused = false
    control.criticalObserved = true
    XCTAssertEqual(control.decision(thermal: 0), .critical)
    control.cancelled = true
    XCTAssertEqual(control.decision(thermal: 3), .cancel)
  }

  func testTrainingPolicyRejectsInvalidResolutionSchedule() throws {
    try PhoneTrainingConfig.best.validate()
    var bad = PhoneTrainingConfig.quick
    bad.resolutionSchedule = 2_000
    XCTAssertThrowsError(try bad.validate())
    bad = .quick
    bad.numDownscales = -1
    XCTAssertThrowsError(try bad.validate())
    bad = .quick
    bad.memoryBudgetFraction = .nan
    XCTAssertThrowsError(try bad.validate())
  }

  /// msplat's fixed-interval scheduler (TrainingPlan.swift nativeResolutionSchedule /
  /// nativeDownscaleFactor) must reproduce every declared stage at both ends, or the
  /// session throws before training. Best's first draft broke this at iteration 3001.
  func testResolutionStagesMatchMsplatScheduler() throws {
    let configs =
      [PhoneTrainingConfig.quick, .best]
      + BenchmarkCase.sweep.map { PhoneTrainingConfig.benchmark(maxGaussians: $0.maxGaussians) }
    for config in configs {
      try config.validate()
      let stages = config.resolutionStages()
      XCTAssertEqual(stages.count, Int(config.numDownscales) + 1)
      XCTAssertEqual(stages.first?.iterations.lowerBound, 1)
      XCTAssertEqual(stages.last?.iterations.upperBound, config.iterations)
      let schedule = stages.count > 1 ? stages[1].iterations.lowerBound : config.iterations + 1
      func factor(_ iteration: Int32) -> Int32 {
        Int32(1) << max(Int(config.numDownscales) - Int(iteration / schedule), 0)
      }
      for (index, stage) in stages.enumerated() {
        XCTAssertEqual(factor(stage.iterations.lowerBound), stage.downscaleFactor, "\(config)")
        XCTAssertEqual(factor(stage.iterations.upperBound), stage.downscaleFactor, "\(config)")
        if index > 0 {
          XCTAssertEqual(stage.iterations.lowerBound, stages[index - 1].iterations.upperBound + 1)
        }
      }
    }
    var single = PhoneTrainingConfig.quick
    single.numDownscales = 0
    XCTAssertEqual(single.resolutionStages().map(\.iterations), [1...single.iterations])
    single.numDownscales = 1
    single.resolutionSchedule = 1
    XCTAssertThrowsError(try single.validate())
  }

  func testSweepIsSixComparableRuns() throws {
    XCTAssertEqual(BenchmarkCase.sweep.count, 6)
    XCTAssertEqual(Set(BenchmarkCase.sweep.map(\.maxGaussians)), [200_000, 400_000, 800_000])
    XCTAssertEqual(Set(BenchmarkCase.sweep.map(\.maxSide)), [640, 960])
    for run in BenchmarkCase.sweep {
      let config = PhoneTrainingConfig.benchmark(maxGaussians: run.maxGaussians)
      XCTAssertEqual(config.iterations, 600)
      XCTAssertEqual(config.shDegree, 1)
      XCTAssertEqual(config.numDownscales, 1)
      XCTAssertEqual(config.maxGaussians, run.maxGaussians)
      XCTAssertEqual(
        try JSONDecoder().decode(
          PhoneTrainingConfig.self,
          from: JSONEncoder().encode(config)), config)
    }
    XCTAssertEqual(PhoneTrainingConfig.quick.maxGaussians, 300_000)
    XCTAssertEqual(PhoneTrainingConfig.best.maxGaussians, 800_000)
  }

  func testCheckpointAndPartialCriticalReportRoundTrip() throws {
    let checkpoint = TrainingCheckpoint(
      capturePath: "/saved/capture", prep: PrepConfig(),
      training: .quick, step: 17)
    let decoded = try JSONDecoder().decode(
      TrainingCheckpoint.self, from: JSONEncoder().encode(checkpoint))
    XCTAssertEqual(decoded.step, 17)
    XCTAssertEqual(decoded.prep.holdoutEvery, 8)
    XCTAssertEqual(decoded.training, .quick)
    var report = BenchmarkReport(
      capturePath: checkpoint.capturePath, startedAt: Date(),
      device: "fixture", operatingSystem: "test", physicalMemory: 1, prep: .init())
    report.stoppedForCriticalThermal = true
    report.stopReason = "critical"
    var row = BenchmarkRun(configuration: BenchmarkCase.sweep[0])
    row.status = "critical"
    row.thermalAtEnd = 3
    report.runs = [row]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try report.write(to: url)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let restored = try decoder.decode(BenchmarkReport.self, from: Data(contentsOf: url))
    XCTAssertTrue(restored.stoppedForCriticalThermal)
    XCTAssertEqual(restored.runs.count, 1)
    XCTAssertNil(restored.runs[0].psnr)
    XCTAssertEqual(restored.runs[0].worstThermal, 3)
  }
}
