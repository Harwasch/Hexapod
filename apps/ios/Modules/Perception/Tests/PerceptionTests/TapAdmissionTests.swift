import Foundation
import Testing

@testable import Perception

@Suite struct TapAdmissionTests {
  let point = ImagePoint(x: 0.5, y: 0.5)

  @Test func trackingRecoversOnNextFrame() {
    var taps = TapAdmission<Int>()
    #expect(
      taps.submit(point, now: 10) { _ in
        SelectionFrameAdmission.capture(
          enabled: true, hasViewport: true, trackingNormal: false, hasDepth: true, copy: { 42 })
      } == .waiting(.trackingNotReady))
    #expect(
      taps.retry(now: 10.1) { _ in
        SelectionFrameAdmission.capture(
          enabled: true, hasViewport: true, trackingNormal: true, hasDepth: true, copy: { 42 })
      } == [.accepted])
    #expect(taps.take(now: 10.1).frame == 42)
  }

  @Test func missingDepthExpiresWithReason() {
    var taps = TapAdmission<Int>()
    #expect(taps.submit(point, now: 10) { _ in .failure(.noDepth) } == .waiting(.noDepth))
    #expect(taps.retry(now: 11.49) { _ in .failure(.noDepth) }.isEmpty)
    #expect(taps.expire(now: 11.5) == [.refused(.noDepth)])
    #expect(taps.isEmpty)
  }

  @Test func frameCopyFailureAndQueueOverflowAreExplicit() {
    var taps = TapAdmission<Int>()
    #expect(
      taps.submit(point, now: 10) { _ in .failure(.frameCopyFailed("copy broke")) }
        == .refused(.frameCopyFailed("copy broke")))
    for _ in 0..<PerceptionLimits.maximumPendingTaps {
      #expect(taps.submit(point, now: 10) { _ in .success(42) } == .accepted)
    }
    #expect(taps.submit(point, now: 10) { _ in .success(0) } == .refused(.queueFull))
  }

  @Test func queuedTapExpiryAndCancellationHaveReasons() {
    var taps = TapAdmission<Int>()
    _ = taps.submit(point, now: 10) { _ in .success(42) }
    #expect(taps.take(now: 20).events == [.refused(.expired)])
    _ = taps.submit(point, now: 21) { _ in .failure(.noDepth) }
    #expect(taps.cancel(reason: .staleGeneration) == [.refused(.staleGeneration)])
  }
  @Test func thrownCopyErrorBecomesTypedRefusal() {
    enum CopyError: Error { case brokenBuffer }
    var taps = TapAdmission<Int>()
    let result = taps.submit(point, now: 10) { _ in
      SelectionFrameAdmission.capture(
        enabled: true, hasViewport: true, trackingNormal: true, hasDepth: true,
        copy: { throw CopyError.brokenBuffer })
    }
    #expect(result == .refused(.frameCopyFailed("brokenBuffer")))
    #expect(taps.isEmpty)
  }

  @Test func readinessNeverCopiesUnreadyFrame() {
    for reason in [SelectionRefusal.disabled, .noViewport, .trackingNotReady, .noDepth] {
      let result: Result<Int, SelectionRefusal> = SelectionFrameAdmission.capture(
        enabled: reason != .disabled, hasViewport: reason != .noViewport,
        trackingNormal: reason != .trackingNotReady, hasDepth: reason != .noDepth,
        copy: {
          Issue.record("Unready frame copied")
          return 42
        })
      #expect(result == .failure(reason))
    }
  }

  @Test func retryDeadlineDoesNotExtendAndNewTapCanSucceed() {
    var taps = TapAdmission<Int>()
    _ = taps.submit(point, now: 10) { _ in .failure(.trackingNotReady) }
    #expect(taps.retry(now: 11) { _ in .failure(.noDepth) }.isEmpty)
    #expect(taps.retry(now: 11.5) { _ in .success(42) } == [.refused(.noDepth)])
    #expect(taps.submit(point, now: 12) { _ in .success(42) } == .accepted)
  }

  @Test func waitingTapsShareBoundedQueueAndPreserveOrder() {
    var taps = TapAdmission<Int>()
    _ = taps.submit(point, now: 10) { _ in .failure(.noDepth) }
    _ = taps.submit(point, now: 10) { _ in .success(2) }
    #expect(taps.readyFrame == nil)
    #expect(
      taps.submit(point, now: 10) { _ in
        Issue.record("Full queue tried to copy a frame")
        return .success(3)
      } == .refused(.queueFull))
    #expect(taps.retry(now: 10.1) { _ in .success(1) } == [.accepted])
    #expect(taps.readyFrame == 1)
    #expect(taps.take(now: 10.1).frame == 1)
    #expect(taps.take(now: 10.1).frame == 2)
  }

}
