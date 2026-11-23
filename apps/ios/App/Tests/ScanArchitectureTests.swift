import Foundation
import Testing

@Test func scanDelegateDoesNotRunOnMainQueueOrConvertPixels() throws {
  let app = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
  let source = try String(
    contentsOf: app.appendingPathComponent("ARCameraView.swift"), encoding: .utf8)
  #expect(!source.contains("delegateQueue = .main"))
  let actions = try String(
    contentsOf: app.appendingPathComponent("CameraActions.swift"), encoding: .utf8)
  #expect(!actions.contains("recorder.append(frame)"))
  #expect(!actions.contains("try renderer.ingest(frame"))
}
