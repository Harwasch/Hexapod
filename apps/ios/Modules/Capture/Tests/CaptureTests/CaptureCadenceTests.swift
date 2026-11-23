import Foundation
import Testing

@testable import Capture

@Test func fiveMinuteAdmissionIsFifteenHzAndHasConstantCost() {
  var cadence = CaptureCadence()
  var accepted = 0
  let start = DispatchTime.now().uptimeNanoseconds
  for frame in 0..<18_000 { if cadence.admit(123_456 + Double(frame) / 60) { accepted += 1 } }
  let elapsed = DispatchTime.now().uptimeNanoseconds - start
  #expect(accepted == 4_500)
  #expect(Double(elapsed) / 18_000 < 500_000)  // mean admission < 0.5 ms, monotonic clock
}
@Test func admissionRejectsDuplicatesRegressionAndDoesNotCatchUpAfterStall() {
  var cadence = CaptureCadence()
  let actual = [Double.nan, 1, 1, 0, 10, 10.001, 10.1].map { cadence.admit($0) }
  #expect(actual == [false, true, false, false, true, false, true])
}
