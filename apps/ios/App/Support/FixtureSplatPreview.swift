import Foundation
import LiveSplat

enum FixtureSplatPreview {
  enum Failure: Error { case noTrackedFrame }

  /// Skip tracking pre-roll, ingest one measured frame, then stop pulling.
  @MainActor static func seed(renderer: any LiveSplatRenderer, directory: URL) async throws {
    let source = try BundleReplayFrameSource(bundleDirectory: directory)
    while let event = try await source.next() {
      try Task.checkCancellation()
      if case .frame(let frame) = event {
        renderer.ingest(frame)
        return
      }
    }
    throw Failure.noTrackedFrame
  }
}
