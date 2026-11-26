import Perception
import Testing

@testable import AppSupport

@Suite struct CameraSelectionSupportTests {
  @Test @MainActor func throwingOverlayStillCreatesWorkingDriver() {
    let attachment = CameraAttachment.attach(
      makeOverlay: { () throws -> Int in throw SelectionRefusal.frameCopyFailed("Metal failed") },
      makeDriver: { overlay in
        #expect(overlay == nil)
        return TapAdmission<Int>()
      })
    #expect(attachment.overlayError?.contains("Metal failed") == true)
    var driver = attachment.driver
    #expect(driver.submit(ImagePoint(x: 0.5, y: 0.5), now: 10) { _ in .success(42) } == .accepted)
    #expect(driver.take(now: 10).frame == 42)
  }

  @Test func acceptanceClearsBannerButKeepsLastRefusalForHUD() {
    var feedback = CameraSelectionFeedback()
    feedback.receive(.refused(.noDepth))
    #expect(feedback.banner == "Can't select here")
    feedback.receive(.waiting(.trackingNotReady))
    #expect(feedback.banner == "Hold still…")
    feedback.receive(.accepted)
    #expect(feedback.banner == nil)
    #expect(feedback.lastRefusal == .noDepth)
  }
}
