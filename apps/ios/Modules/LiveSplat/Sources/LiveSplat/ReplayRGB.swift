#if canImport(AVFoundation) && canImport(CoreGraphics)
  import AVFoundation
  import CoreGraphics

  /// Debug replay decoder, not a real-time capture path. Kept out of Linux builds.
  @MainActor enum ReplayRGB {
    static func colors(
      video: URL, seconds: Double, imageWidth: Int, imageHeight: Int, width: Int, height: Int
    ) async throws -> [SplatVector] {
      let generator = AVAssetImageGenerator(asset: AVURLAsset(url: video))
      generator.appliesPreferredTrackTransform = false
      generator.requestedTimeToleranceBefore = .zero
      generator.requestedTimeToleranceAfter = .zero
      let result = try await generator.image(
        at: CMTime(seconds: seconds, preferredTimescale: 600_000))
      let image = result.image
      guard image.width == imageWidth, image.height == imageHeight else {
        throw BundleReplayError.invalidPayload("rgb.mov dimensions")
      }
      return try sample(image: image, width: width, height: height)
    }

    static func sample(image: CGImage, width: Int, height: Int) throws -> [SplatVector] {
      let imageWidth = image.width
      let imageHeight = image.height
      // Draw full resolution before nearest-pixel sampling, matching depth's integer
      // pixel centres. Read raw bitmap rows; no UIKit display-coordinate flip.
      var bytes = [UInt8](repeating: 0, count: imageWidth * imageHeight * 4)
      try bytes.withUnsafeMutableBytes { buffer in
        guard
          let context = CGContext(
            data: buffer.baseAddress, width: imageWidth, height: imageHeight,
            bitsPerComponent: 8, bytesPerRow: imageWidth * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              | CGBitmapInfo.byteOrder32Big.rawValue)
        else { throw LiveSplatError.unavailable }
        context.draw(image, in: CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
      }
      return (0..<(width * height)).map { i in
        let x = (i % width) * imageWidth / width
        let y = (i / width) * imageHeight / height
        let offset = (y * imageWidth + x) * 4
        return SplatVector(
          Double(bytes[offset]) / 255, Double(bytes[offset + 1]) / 255,
          Double(bytes[offset + 2]) / 255)
      }
    }
  }
#endif
