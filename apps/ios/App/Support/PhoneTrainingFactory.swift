import Foundation
import PhoneTrain

#if os(iOS)
  import UIKit
#endif

/// The only app selection point. Production construction stays in the caller's closure.
enum PhoneTrainingFactory {
  @MainActor static func make(
    outputDirectory: URL,
    real: () async throws -> any PhoneTrainingSession
  ) async throws -> any PhoneTrainingSession {
    #if DEBUG && os(iOS)
      if LaunchMode.current.isUITest {
        guard let directory = LaunchMode.current.previewDirectory,
          let image = await CapturePreview.image(directory: directory),
          let png = image.pngData()
        else { throw PrepError.invalid("Fixture preview missing") }
        return FixturePhoneTrainingSession(outputDirectory: outputDirectory, previewPNG: png)
      }
    #endif
    return try await real()
  }
}
