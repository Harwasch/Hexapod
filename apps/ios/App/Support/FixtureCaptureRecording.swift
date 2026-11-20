import Capture
import Foundation

/// Explicit debug replay only. Exercise the same disk finalizer as the AR recorder,
/// preserving paired frame/depth records and a real fixture video, without camera permission.
enum FixtureCaptureRecording {
  static func save(source: URL, destination: URL, details: CaptureDetails) async throws -> URL {
    let disk = try await RecordingDiskWriter.create(
      directory: destination, device: "Simulated fixture")
    let records = try await Task.detached {
      try Data(contentsOf: source.appendingPathComponent("frames.jsonl")).split(separator: 10)
        .map { try JSONDecoder().decode(CaptureFrame.self, from: Data($0)) }
    }.value
    for frame in records {
      let packet = try await Task.detached {
        var files: [RecordedFile] = []
        for reference in [frame.depth, frame.smoothedDepth].compactMap({ $0 }) {
          for name in [reference.file, reference.confidence] {
            guard CaptureBundleWriter.safeName(name) else { throw CaptureError.invalidFile }
            files.append(
              RecordedFile(
                name: name, data: try Data(contentsOf: source.appendingPathComponent(name))))
          }
        }
        return RecordingPacket.frame(frame, files)
      }.value
      try await disk.write(packet)
    }
    try await Task.detached {
      try FileManager.default.copyItem(
        at: source.appendingPathComponent("rgb.mov"),
        to: destination.appendingPathComponent("rgb.mov"))
    }.value
    try await disk.writeDetails(details)
    try await disk.finish()
    return destination
  }
}
