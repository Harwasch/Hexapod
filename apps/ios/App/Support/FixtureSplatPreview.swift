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

  /// Resolve one tip from the same static geometry, then let its dwell settle without a timer.
  @MainActor static func guidance(renderer: DefaultLiveSplatRenderer) -> ScanGuidance {
    var guidance = ScanGuidance()
    guard let frame = renderer.latestFrame, !renderer.splats.isEmpty else { return guidance }
    var center = SIMD3<Float>(repeating: 0)
    for splat in renderer.splats {
      center += SIMD3<Float>(
        Float(splat.position.x), Float(splat.position.y), Float(splat.position.z))
    }
    let count: Float = Float(renderer.splats.count)
    center /= count
    let position = frame.camera.position
    let camera = SIMD3<Float>(Float(position.x), Float(position.y), Float(position.z))
    let sample = ScanGuidanceSample(
      center: center, camera: camera, observedCells: renderer.splats.count,
      shortSpreadFraction: 1, timestamp: 0)
    guidance.update(sample, blur: 0, now: 0)
    guidance.update(nil, blur: 0, now: 1)
    return guidance
  }
}
