#if canImport(UIKit) && canImport(AVFoundation)
  import AVFoundation
  import Foundation
  import UIKit

  @MainActor enum CapturePreview {
    static func image(directory: URL) async -> UIImage? {
      let thumbnail = directory.appendingPathComponent("thumbnail.jpg")
      if let data = try? await Task.detached(
        priority: .utility,
        operation: {
          try Data(contentsOf: thumbnail)
        }
      ).value, let image = UIImage(data: data) {
        return image
      }
      do {
        let generator = AVAssetImageGenerator(
          asset: AVURLAsset(url: directory.appendingPathComponent("rgb.mov")))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 640, height: 640)
        let result = try await generator.image(at: .zero)
        let image = UIImage(cgImage: result.image)
        if let data = image.jpegData(compressionQuality: 0.8) {
          try? await Task.detached(priority: .utility) {
            try data.write(to: thumbnail, options: .atomic)
          }.value
        }
        return image
      } catch { return nil }
    }
  }
#endif
