import Testing
@testable import Capture

@Test func reconstructionVideoBudgetScalesWithPixelsAndCapsLargeFormats() {
  let settings = CaptureVideoSettings(width: 1920, height: 1440)
  #expect(settings.averageBitRate == 26_956_800)
  #expect(settings.averageBitRate * 48 / 8 == 161_740_800)
  #expect(CaptureVideoSettings(width: 1440, height: 1920).averageBitRate == settings.averageBitRate)
  #expect(CaptureVideoSettings(width: 1280, height: 720).averageBitRate == 8_985_600)
  #expect(CaptureVideoSettings(width: 3840, height: 2160).averageBitRate == 30_000_000)
}

#if canImport(AVFoundation)
  import AVFoundation

  @Test func reconstructionWriterUsesHEVCWithExplicitCompressionSettings() throws {
    let output = CaptureVideoSettings(width: 1920, height: 1440).outputSettings
    #expect(output[AVVideoCodecKey] as? AVVideoCodecType == .hevc)
    #expect(output[AVVideoWidthKey] as? Int == 1920)
    #expect(output[AVVideoHeightKey] as? Int == 1440)
    let compression = try #require(output[AVVideoCompressionPropertiesKey] as? [String: Any])
    #expect(compression[AVVideoAverageBitRateKey] as? Int == 26_956_800)
    #expect(compression[AVVideoExpectedSourceFrameRateKey] as? Int == 15)
    #expect(compression[AVVideoMaxKeyFrameIntervalKey] as? Int == 15)
    #expect(compression[AVVideoAllowFrameReorderingKey] as? Bool == false)
    #expect(compression[AVVideoQualityKey] == nil)
  }
#endif
