#if canImport(Vision) && canImport(CoreML) && canImport(CoreImage)
  @preconcurrency import Vision
  @preconcurrency import CoreML
  import CoreImage
  import CoreGraphics
  import CoreVideo
  import Foundation

  private func pixelBuffer(_ image: CameraImage) throws -> CVPixelBuffer {
    guard image.width > 0, image.height > 0,
      image.rgba.count == image.width * image.height * 4
    else { throw PerceptionError.invalidGeometry }
    var buffer: CVPixelBuffer?
    guard
      CVPixelBufferCreate(
        kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32BGRA,
        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer) == kCVReturnSuccess,
      let buffer
    else { throw PerceptionError.invalidGeometry }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
      throw PerceptionError.invalidGeometry
    }
    for y in 0..<image.height {
      let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(buffer)).assumingMemoryBound(
        to: UInt8.self)
      for x in 0..<image.width {
        let i = (y * image.width + x) * 4
        row[x * 4] = image.rgba[i + 2]
        row[x * 4 + 1] = image.rgba[i + 1]
        row[x * 4 + 2] = image.rgba[i]
        row[x * 4 + 3] = image.rgba[i + 3]
      }
    }
    return buffer
  }

  /// Vision discovers foreground instances, without claiming a semantic object identity.
  /// Prompt and output both use unrotated top-left camera coordinates.
  /// Device-only: mask orientation and Vision instance quality need an iPhone acceptance run.
  public actor VisionSegmenter: InstanceSegmenting {
    public init() {}
    public func instances(_ image: CameraImage) async throws -> [SubjectMask] {
      try Task.checkCancellation()
      let buffer = try pixelBuffer(image)
      let request = VNGenerateForegroundInstanceMaskRequest()
      try VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
      guard let observation = request.results?.first else { return [] }
      let map = observation.instanceMask
      CVPixelBufferLockBaseAddress(map, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(map) else { throw PerceptionError.noSubject }
      let width = CVPixelBufferGetWidth(map)
      let height = CVPixelBufferGetHeight(map)
      return observation.allInstances.map { id in
        var pixels = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
          let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(
            to: UInt8.self)
          for x in 0..<width { pixels[y * width + x] = Int(row[x]) == id ? 255 : 0 }
        }
        return SubjectMask(width: width, height: height, pixels: pixels)
      }
    }
    public func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
      try Task.checkCancellation()
      let buffer = try pixelBuffer(image)
      let request = VNGenerateForegroundInstanceMaskRequest()
      try VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
      guard let observation = request.results?.first else { throw PerceptionError.noSubject }
      let instanceMap = observation.instanceMask
      CVPixelBufferLockBaseAddress(instanceMap, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(instanceMap, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(instanceMap) else {
        throw PerceptionError.noSubject
      }
      let width = CVPixelBufferGetWidth(instanceMap)
      let height = CVPixelBufferGetHeight(instanceMap)
      let rowBytes = CVPixelBufferGetBytesPerRow(instanceMap)
      func instance(_ x: Int, _ y: Int) -> UInt8 {
        base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)[x]
      }
      var chosen: UInt8 = 0
      switch prompt {
      case .point(let p):
        guard p.x.isFinite, p.y.isFinite, (0..<1).contains(p.x), (0..<1).contains(p.y) else {
          throw PerceptionError.noSubject
        }
        chosen = instance(Int(p.x * Double(width)), Int(p.y * Double(height)))
      case .box(let box):
        var counts: [UInt8: Int] = [:]
        for y in 0..<height {
          for x in 0..<width
          where box.contains(
            .init(x: (Double(x) + 0.5) / Double(width), y: (Double(y) + 0.5) / Double(height)))
          {
            let value = instance(x, y)
            if value > 0 { counts[value, default: 0] += 1 }
          }
        }
        chosen =
          counts.max { $0.value == $1.value ? $0.key > $1.key : $0.value < $1.value }?.key ?? 0
      }
      guard chosen > 0 else { throw PerceptionError.noSubject }
      var pixels = [UInt8](repeating: 0, count: width * height)
      for y in 0..<height {
        for x in 0..<width {
          pixels[y * width + x] = instance(x, y) == chosen ? 255 : 0
        }
      }
      return SubjectMask(width: width, height: height, pixels: pixels)
    }
  }

  /// Interface for a promptable single-stage mobile export. A model with a separate image
  /// encoder/decoder must supply its own Segmenter; no unverified tensor layout is inferred.
  public struct CoreMLPromptLayout: Sendable {
    public var image: String
    public var point: String
    public var box: String
    public var mask: String
    public var threshold: Double
    public init(image: String, point: String, box: String, mask: String, threshold: Double = 0) {
      self.image = image
      self.point = point
      self.box = box
      self.mask = mask
      self.threshold = threshold
    }
  }

  /// Device-only integration, unverified without the selected export and Xcode.
  /// Contract: image input, normalized [1,2] point / [1,4] box (-1 marks unused prompt),
  /// and a [H,W] or [1,1,H,W] logit output. Unsupported exports fail to Vision.
  public actor CoreMLSegmenter: Segmenter {
    private let model: MLModel
    private let layout: CoreMLPromptLayout
    public init(modelURL: URL, layout: CoreMLPromptLayout) throws {
      let configuration = MLModelConfiguration()
      configuration.computeUnits = .all
      let compiled = try MLModel.compileModel(at: modelURL)
      model = try MLModel(contentsOf: compiled, configuration: configuration)
      self.layout = layout
    }
    public func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
      try Task.checkCancellation()
      let point = try MLMultiArray(shape: [1, 2], dataType: .float32)
      let box = try MLMultiArray(shape: [1, 4], dataType: .float32)
      for i in 0..<2 { point[i] = -1 }
      for i in 0..<4 { box[i] = -1 }
      switch prompt {
      case .point(let p):
        point[0] = NSNumber(value: p.x)
        point[1] = NSNumber(value: p.y)
      case .box(let b):
        for (i, value) in [b.min.x, b.min.y, b.max.x, b.max.y].enumerated() {
          box[i] = NSNumber(value: value)
        }
      }
      guard
        let constraint = model.modelDescription.inputDescriptionsByName[layout.image]?
          .imageConstraint
      else {
        throw PerceptionError.unsupportedModel
      }
      let original = try pixelBuffer(image)
      var resized: CVPixelBuffer?
      guard
        CVPixelBufferCreate(
          kCFAllocatorDefault, constraint.pixelsWide, constraint.pixelsHigh,
          kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
          &resized) == kCVReturnSuccess, let resized
      else { throw PerceptionError.invalidGeometry }
      let ci = CIImage(cvPixelBuffer: original).transformed(
        by: CGAffineTransform(
          scaleX: Double(constraint.pixelsWide) / Double(image.width),
          y: Double(constraint.pixelsHigh) / Double(image.height)))
      CIContext().render(ci, to: resized)
      let input = try MLDictionaryFeatureProvider(dictionary: [
        layout.image: MLFeatureValue(pixelBuffer: resized),
        layout.point: MLFeatureValue(multiArray: point),
        layout.box: MLFeatureValue(multiArray: box),
      ])
      let output = try model.prediction(from: input)
      guard let mask = output.featureValue(for: layout.mask)?.multiArrayValue,
        mask.shape.count == 2 || (mask.shape.count == 4 && mask.shape[0] == 1 && mask.shape[1] == 1)
      else { throw PerceptionError.unsupportedModel }
      let height = mask.shape[mask.shape.count - 2].intValue
      let width = mask.shape.last!.intValue
      guard width > 0, height > 0 else { throw PerceptionError.invalidGeometry }
      // Multidimensional indexing respects non-contiguous Core ML strides.
      var pixels: [UInt8] = []
      for y in 0..<height {
        for x in 0..<width {
          let indices: [NSNumber] =
            mask.shape.count == 2
            ? [NSNumber(value: y), NSNumber(value: x)]
            : [0, 0, NSNumber(value: y), NSNumber(value: x)]
          pixels.append(mask[indices].doubleValue > layout.threshold ? 255 : 0)
        }
      }
      return SubjectMask(width: width, height: height, pixels: pixels)
    }
  }
  /// Lazily downloads and verifies configured weights in Application Support. A missing or
  /// incompatible export uses Vision for this session; changing configuration creates a new instance.
  public actor RuntimeSegmenter: Segmenter {
    private let configuration: ModelConfiguration?
    private let layout: CoreMLPromptLayout?
    private let directory: URL
    private let fallback = VisionSegmenter()
    private var primary: CoreMLSegmenter?
    private var attempted = false
    public init(
      configuration: ModelConfiguration? = nil, layout: CoreMLPromptLayout? = nil,
      directory: URL? = nil
    ) {
      self.configuration = configuration
      self.layout = layout
      self.directory =
        directory
        ?? FileManager.default.urls(
          for: .applicationSupportDirectory,
          in: .userDomainMask)[0].appendingPathComponent("Hex/Segmentation", isDirectory: true)
    }
    public func segment(_ image: CameraImage, prompt: SubjectPrompt) async throws -> SubjectMask {
      if !attempted {
        attempted = true
        if let configuration, let layout {
          // Weight acquisition must not delay the first on-device answer.
          Task { [weak self] in await self?.load(configuration, layout: layout) }
        }
      }
      return try await FallbackSegmenter(primary: primary, fallback: fallback).segment(
        image, prompt: prompt)
    }
    private func load(_ configuration: ModelConfiguration, layout: CoreMLPromptLayout) async {
      do {
        let url = try await ModelCache(directory: directory).model(configuration)
        primary = try CoreMLSegmenter(modelURL: url, layout: layout)
      } catch { primary = nil }
    }

  }

  /// Generic image classification is deliberately reported with its own confidence.
  /// It is not evidence for fine-grained species identification or a safety assessment.
  public actor VisionIdentifier: ImageIdentifying {
    public init() {}
    public func identify(_ image: CameraImage, mask: SubjectMask) async throws -> DeviceLabel? {
      let isolated = try SubjectImage.isolated(image, mask: mask)
      let request = VNClassifyImageRequest()
      try VNImageRequestHandler(cvPixelBuffer: try pixelBuffer(isolated), orientation: .up).perform(
        [request])
      guard let result = request.results?.first, result.confidence >= 0.5 else { return nil }
      return DeviceLabel(
        label: result.identifier.replacingOccurrences(of: "_", with: " "),
        confidence: Double(result.confidence))
    }
  }
#endif
