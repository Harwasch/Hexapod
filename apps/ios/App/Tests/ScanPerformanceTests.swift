import Capture
import Foundation
import LiveSplat
import XCTest

@testable import AppSupport

/// Discovered by the existing packages.py simulator CI job, as well as Linux swift test.
final class ScanPerformanceTests: XCTestCase {
  func testFiveMinuteAdmissionPerformance() {
    measure {
      var cadence = CaptureCadence()
      var recorded = 0
      let start = DispatchTime.now().uptimeNanoseconds
      for frame in 0..<18_000 {
        if cadence.admit(1000 + Double(frame) / 60) { recorded += 1 }
      }
      XCTAssertEqual(recorded, 4500)
      XCTAssertLessThan(Double(DispatchTime.now().uptimeNanoseconds - start) / 18_000, 500_000)
    }
  }
}
