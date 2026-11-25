/// Reconstruction RGB budget for the App's nominal 15 fps shared-session recorder.
/// 0.65 bits/pixel/frame gives ~27 Mbit/s at 1920x1440; cap larger formats at 30 Mbit/s.
/// This is an average-rate target, not a hard file-size or encoder-throughput guarantee.
public struct CaptureVideoSettings: Sendable {
  public let width: Int
  public let height: Int
  public let averageBitRate: Int

  public init(width: Int, height: Int) {
    precondition(width > 0 && height > 0)
    self.width = width
    self.height = height
    averageBitRate = Int(min(Double(width) * Double(height) * 15 * 0.65, 30_000_000).rounded())
  }
}

#if canImport(AVFoundation)
  import AVFoundation

  extension CaptureVideoSettings {
    /// Pure settings construction; no device, session or writer is needed.
    public var outputSettings: [String: Any] {
      [
        AVVideoCodecKey: AVVideoCodecType.hevc,
        AVVideoWidthKey: width, AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          // Use HEVC average-bitrate control; do not assume AVVideoQualityKey support.
          AVVideoAverageBitRateKey: averageBitRate,
          AVVideoExpectedSourceFrameRateKey: 15,
          AVVideoMaxKeyFrameIntervalKey: 15,
          AVVideoAllowFrameReorderingKey: false,
        ],
      ]
    }
  }
#endif
