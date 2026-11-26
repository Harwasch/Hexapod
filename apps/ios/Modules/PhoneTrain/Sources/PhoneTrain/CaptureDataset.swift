#if canImport(Msplat) && canImport(AVFoundation)
  import AVFoundation
  import CoreImage
  import ImageIO
  import Msplat
  import PhoneTrainPrep
  import UniformTypeIdentifiers

  /// Owns disposable JPEGs only. Keep alive until the trainer closes; release with
  /// cleanup(). Rebuild from the immutable bundle after cache eviction/relaunch.
  public struct PreparedDataset: Sendable {
    public let descriptor: DatasetDescriptor
    public let frames: [PreparedFrame]
    public let cacheDirectory: URL
    public let width: Int
    public let height: Int
    public let voxelSize: Double
    public func cleanup() throws { try FileManager.default.removeItem(at: cacheDirectory) }
  }

  /// Decode and CPU preparation run off MainActor. Two sequential video passes keep
  /// memory bounded: small blur rasters, then JPEGs/seeds for selected views only.
  public actor CaptureDatasetBuilder {
    public init() {}

    public func prepare(bundle: URL, config: PrepConfig = .init()) async throws -> PreparedDataset {
      try config.validate()
      let frames = try CaptureFrame.read(from: bundle)
      let asset = AVURLAsset(url: bundle.appendingPathComponent("rgb.mov"))
      guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw PrepError.invalid("Capture has no RGB track")
      }
      let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("PhoneTrain/\(UUID().uuidString)")
      try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
      do {
        var scores: [Int: Double] = [:]
        let context = CIContext(options: [.cacheIntermediates: false])
        try decode(asset: asset, track: track, frames: frames) { image, frame in
          guard frame.trackingState == "normal" else { return }
          let dimensions = Self.dimensions(frame, maxSide: config.blurMaxSide)
          let rgb = try Self.raster(
            image, context: context, width: dimensions.0, height: dimensions.1)
          let luma = stride(from: 0, to: rgb.count, by: 3).map {
            (0.2126 * Double(rgb[$0]) + 0.7152 * Double(rgb[$0 + 1]) + 0.0722 * Double(rgb[$0 + 2]))
              / 255
          }
          scores[frame.index] = try Keyframes.blurScore(
            luma: luma, width: dimensions.0, height: dimensions.1)
        }
        let selected = try Keyframes.select(frames: frames, scores: scores, config: config)
        let byIndex = Dictionary(uniqueKeysWithValues: selected.map { ($0.source.index, $0) })
        var seeds = VoxelSeeds(size: config.voxelSize, maximum: config.maximumSeedPoints)
        var datasetFrames: [DatasetFrame] = []
        var width = 0
        var height = 0
        try decode(asset: asset, track: track, frames: frames) { image, frame in
          guard let keyframe = byIndex[frame.index] else { return }
          let dimensions = Self.dimensions(frame, maxSide: config.trainingMaxSide)
          (width, height) = dimensions
          let rgb = try Self.raster(image, context: context, width: width, height: height)
          let url = cache.appendingPathComponent(String(format: "%08d.jpg", frame.index))
          try Self.jpeg(
            rgb: rgb, width: width, height: height, quality: config.jpegQuality, to: url)
          let k = frame.scaledIntrinsics(width: width, height: height)
          let calibration = try DatasetCalibration(
            width: width, height: height,
            fx: Float(k[0]), fy: Float(k[4]), cx: Float(k[2]), cy: Float(k[5]))
          datasetFrames.append(
            try DatasetFrame(
              id: String(format: "%08d", frame.index),
              calibrationID: "arkit-\(frame.index)", imageURL: url,
              rasterOrientation: .encodedPixels,
              calibration: calibration,
              cameraToWorld: CameraPose(elements: keyframe.pose.map(Float.init))))
          // Held-out images must never contribute geometry/colour to initialization.
          if !keyframe.isHoldout, let depth = frame.smoothedDepth ?? frame.depth {
            let depthData = try Self.payload(depth.file, bundle: bundle)
            let confidence = try Self.payload(depth.confidence, bundle: bundle)
            guard depth.width > 0, depth.height > 0 else {
              throw PrepError.invalid("Depth dimensions")
            }
            let values = try SeedPoints.decodeDepth(depthData, count: depth.width * depth.height)
            seeds.add(
              try SeedPoints.backProject(
                depth: values, confidence: [UInt8](confidence),
                width: depth.width, height: depth.height,
                intrinsics: frame.scaledIntrinsics(width: depth.width, height: depth.height),
                pose: keyframe.pose, rgb: rgb, rgbWidth: width, rgbHeight: height, config: config))
          }
        }
        let points = seeds.points
        guard !points.isEmpty else {
          throw PrepError.invalid("No confident depth seeds in training views")
        }
        let sparse = try DatasetSparsePointSet(
          xyz: points.flatMap { [Float($0.x), Float($0.y), Float($0.z)] },
          rgb: points.flatMap(\.rgb))
        let descriptor = try DatasetDescriptor(frames: datasetFrames, points: sparse)
        return PreparedDataset(
          descriptor: descriptor, frames: selected, cacheDirectory: cache,
          width: width, height: height, voxelSize: seeds.size)
      } catch {
        try? FileManager.default.removeItem(at: cache)
        throw error
      }
    }

    private static func payload(_ name: String, bundle: URL) throws -> Data {
      guard !name.isEmpty, name.utf8.count < 100,
        name.utf8.allSatisfy({
          (48...57).contains($0) || (65...90).contains($0)
            || (97...122).contains($0) || [45, 46, 95].contains($0)
        }),
        name != ".", name != ".."
      else { throw PrepError.invalid("Unsafe depth filename") }
      let url = bundle.appendingPathComponent(name)
      guard try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
        throw PrepError.invalid("Depth symlink")
      }
      return try Data(contentsOf: url)
    }

    private static func dimensions(_ frame: CaptureFrame, maxSide: Int) -> (Int, Int) {
      let scale = min(1, Double(maxSide) / Double(max(frame.imageWidth, frame.imageHeight)))
      return (
        max(3, Int(Double(frame.imageWidth) * scale)),
        max(3, Int(Double(frame.imageHeight) * scale))
      )
    }

    /// Synchronous reader pull loop is isolated to this CPU actor, never MainActor.
    /// AVAsset's track discovery above uses its async load API. No AV image-generator
    /// synchronous overload (unavailable from Swift concurrency) is used.
    private func decode(
      asset: AVAsset, track: AVAssetTrack, frames: [CaptureFrame],
      consume: (CIImage, CaptureFrame) throws -> Void
    ) throws {
      let reader = try AVAssetReader(asset: asset)
      let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
      output.alwaysCopiesSampleData = false
      guard reader.canAdd(output) else { throw PrepError.invalid("Cannot decode RGB") }
      reader.add(output)
      guard reader.startReading() else {
        throw reader.error ?? PrepError.invalid("RGB reader failed")
      }
      defer { if reader.status == .reading { reader.cancelReading() } }
      var index = 0
      while let sample = output.copyNextSampleBuffer() {
        try Task.checkCancellation()
        guard index < frames.count, let buffer = CMSampleBufferGetImageBuffer(sample) else {
          throw PrepError.invalid("Video/log frame mismatch")
        }
        let frame = frames[index]
        let pts = CMTimeConvertScale(
          CMSampleBufferGetPresentationTimeStamp(sample),
          timescale: 1_000_000_000, method: .roundHalfAwayFromZero
        ).value
        guard pts == Int64((frame.timestamp * 1e9).rounded()),
          CVPixelBufferGetWidth(buffer) == frame.imageWidth,
          CVPixelBufferGetHeight(buffer) == frame.imageHeight
        else {
          throw PrepError.invalid("RGB timestamp/dimensions mismatch at \(index)")
        }
        try autoreleasepool { try consume(CIImage(cvPixelBuffer: buffer), frame) }
        index += 1
      }
      guard reader.status == .completed, index == frames.count else {
        throw reader.error ?? PrepError.invalid("Video/log count mismatch")
      }
    }

    private static func raster(_ image: CIImage, context: CIContext, width: Int, height: Int) throws
      -> [UInt8]
    {
      let scaled = image.transformed(
        by: CGAffineTransform(
          scaleX: CGFloat(width) / image.extent.width,
          y: CGFloat(height) / image.extent.height))
      let bounds = CGRect(x: 0, y: 0, width: width, height: height)
      guard let cg = context.createCGImage(scaled, from: bounds) else {
        throw PrepError.invalid("RGB conversion")
      }
      var rgba = [UInt8](repeating: 0, count: width * height * 4)
      let rendered = rgba.withUnsafeMutableBytes { bytes -> Bool in
        guard
          let canvas = CGContext(
            data: bytes.baseAddress, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return false }
        canvas.translateBy(x: 0, y: CGFloat(height))
        canvas.scaleBy(x: 1, y: -1)
        canvas.draw(cg, in: bounds)
        return true
      }
      guard rendered else { throw PrepError.invalid("RGB raster allocation") }
      return stride(from: 0, to: rgba.count, by: 4).flatMap {
        [rgba[$0], rgba[$0 + 1], rgba[$0 + 2]]
      }
    }

    private static func jpeg(rgb: [UInt8], width: Int, height: Int, quality: Double, to url: URL)
      throws
    {
      guard let provider = CGDataProvider(data: Data(rgb) as CFData),
        let image = CGImage(
          width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
          bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(rawValue: 0),
          provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
        let destination = CGImageDestinationCreateWithURL(
          url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
      else { throw PrepError.invalid("JPEG allocation") }
      CGImageDestinationAddImage(
        destination, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
      guard CGImageDestinationFinalize(destination) else { throw PrepError.invalid("JPEG write") }
    }
  }
#endif
