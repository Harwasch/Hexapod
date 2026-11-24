import Foundation
import Testing

@testable import Perception

private final class RequestResults: @unchecked Sendable {
  private let lock = NSLock()
  private var storage: [Int] = []
  func append(_ value: Int) { lock.withLock { storage.append(value) } }
  var values: [Int] { lock.withLock { storage } }
}

private actor RequestGate {
  var values: [Int] = []
  var active = 0
  var maximumActive = 0
  var arrived: CheckedContinuation<Void, Never>?
  var releaseFirst: CheckedContinuation<Void, Never>?
  func run(_ value: Int) async -> Int {
    active += 1
    maximumActive = max(maximumActive, active)
    values.append(value)
    if value == 1 {
      arrived?.resume()
      arrived = nil
      await withCheckedContinuation { releaseFirst = $0 }
    }
    active -= 1
    return value
  }
  func waitForFirst() async {
    if !values.isEmpty { return }
    await withCheckedContinuation { arrived = $0 }
  }
  func release() {
    releaseFirst?.resume()
    releaseFirst = nil
  }
}

@Test func inferenceStartsImmediatelyAfterCompletionAndKeepsOnlyTheNewestFrame() async {
  let gate = RequestGate()
  // A frozen clock proves admission never waits for a minimum elapsed interval.
  let loop = ForegroundInferenceLoop<Int, Int>(
    clock: { 1000 }, process: { await gate.run($0) }, result: { _ in })
  loop.offer(1, capturedAt: 1000)
  await gate.waitForFirst()
  loop.offer(2, capturedAt: 1000)
  loop.offer(3, capturedAt: 1000)
  #expect(await gate.values == [1])
  await gate.release()
  await loop.waitUntilIdle()
  #expect(await gate.values == [1, 3])
  #expect(await gate.maximumActive == PerceptionLimits.maximumInFlightRequests)
}

@Test func disablingAndResumingDoesNotStartAnotherRequestWhileAnOldOneIsRunning() async {
  let gate = RequestGate()
  let results = RequestResults()
  let loop = ForegroundInferenceLoop<Int, Int>(
    process: { await gate.run($0) }, result: { results.append($0) })
  loop.offer(1, capturedAt: ProcessInfo.processInfo.systemUptime)
  await gate.waitForFirst()
  loop.setEnabled(false)
  loop.setEnabled(true)
  loop.offer(4, capturedAt: ProcessInfo.processInfo.systemUptime)
  #expect(await gate.values == [1])
  await gate.release()
  await loop.waitUntilIdle()
  #expect(await gate.values == [1, 4])
  #expect(await gate.maximumActive == 1)
  #expect(results.values == [4])
}

@Test func rateIsATargetRatherThanAnAdmissionIntervalAndMetricsUseCompletionTimes() {
  #expect(PerceptionLimits.targetInferenceHz == 10)
  #expect(PerceptionLimits.maximumInFlightRequests == 1)
  #expect(PerceptionLimits.maximumRetainedFrames == 1)
  var metrics = InferenceMetrics()
  metrics.record(duration: 0.04, completedAt: 1)
  metrics.record(duration: 0.06, completedAt: 1.05)
  #expect(abs(metrics.averageDurationMilliseconds - 50) < 1e-9)
  #expect(abs(metrics.achievedHz - 20) < 1e-9)
}

@Test func staleInputIsDroppedBeforeARequestStarts() async {
  let gate = RequestGate()
  let loop = ForegroundInferenceLoop<Int, Int>(
    clock: { 1000 }, process: { await gate.run($0) }, result: { _ in })
  loop.offer(99, capturedAt: 1000 - PerceptionLimits.maximumInputAge * 2)
  await loop.waitUntilIdle()
  #expect(await gate.values.isEmpty)
  #expect(loop.metrics.requestCount == 0)
}

@Test func rollingMetricsBoundTheWindowAndIncludeEachRequestDuration() {
  var metrics = InferenceMetrics()
  metrics.record(duration: 1, completedAt: 0)
  for i in 1...PerceptionLimits.metricsWindow {
    metrics.record(duration: 0.02, completedAt: Double(i) / 50)
  }
  #expect(metrics.requestCount == PerceptionLimits.metricsWindow + 1)
  #expect(abs(metrics.averageDurationMilliseconds - 20) < 1e-9)
  #expect(abs(metrics.achievedHz - 50) < 1e-9)
}

@Test func namedStalenessBoundsRejectExpiredAndFutureData() {
  for limit in [
    PerceptionLimits.maximumInputAge, PerceptionLimits.maximumTapAge,
    PerceptionLimits.maximumCameraAge,
  ] {
    #expect(PerceptionLimits.isCurrent(capturedAt: 1, now: 1 + limit / 2, maximumAge: limit))
    #expect(!PerceptionLimits.isCurrent(capturedAt: 1, now: 1 + limit * 2, maximumAge: limit))
    #expect(!PerceptionLimits.isCurrent(capturedAt: 2, now: 1, maximumAge: limit))
    #expect(!PerceptionLimits.isCurrent(capturedAt: .nan, now: 1, maximumAge: limit))
  }
  #expect(PerceptionLimits.maximumRetainedResults == 1)
  #expect(PerceptionLimits.maximumPendingTaps == 2)
  #expect(PerceptionLimits.maximumGPUFramesInFlight == 3)
}

@Test func cameraFrameDroppingDoesNotReplaceQueuedUserTaps() {
  var taps = InferenceIntentQueue<Int>()
  for value in 0..<PerceptionLimits.maximumPendingTaps {
    let accepted = taps.offer(value)
    #expect(accepted)
  }
  let overflow = taps.offer(999)
  #expect(!overflow)
  for value in 0..<PerceptionLimits.maximumPendingTaps {
    let next = taps.take()
    #expect(next == value)
  }
  let empty = taps.take()
  #expect(empty == nil)
  let accepted = taps.offer(1)
  #expect(accepted)
  taps.clear()
  let cleared = taps.take()
  #expect(cleared == nil)
}

@Test func idleAdmissionDoesNotCopyOrRequestAndBusyArrivalsAreDropped() async {
  let gate = RequestGate()
  let loop = ForegroundInferenceLoop<Int, Int>(
    clock: { 1000 }, process: { await gate.run($0) }, result: { _ in })
  var copies = 0
  for _ in 0..<60 {
    loop.offerIfIdle(capturedAt: 1000, hasDemand: false) { copies += 1; return 99 }
  }
  await loop.waitUntilIdle()
  #expect(copies == 0 && loop.metrics.requestCount == 0)
  loop.offerIfIdle(capturedAt: 1000, hasDemand: true) { copies += 1; return 1 }
  await gate.waitForFirst()
  loop.offerIfIdle(capturedAt: 1000, hasDemand: true) { copies += 1; return 2 }
  #expect(copies == 1)
  await gate.release()
  await loop.waitUntilIdle()
  loop.offerIfIdle(capturedAt: 1000, hasDemand: true) { copies += 1; return 3 }
  await loop.waitUntilIdle()
  #expect(await gate.values == [1, 3])
  #expect(copies == 2)
  #expect(await gate.maximumActive == 1)
}
