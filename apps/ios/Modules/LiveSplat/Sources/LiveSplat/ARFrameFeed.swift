#if canImport(ARKit) && canImport(UIKit)
  import ARKit
  import CoreVideo
  import UIKit

  // DEVICE VALIDATION REQUIRED: ARKit buffer formats, RGB/depth alignment and colour.
  @MainActor extension LiveSplatRenderer {
    /// The App continues to own ARSession configuration, delegate and lifetime.
    /// Enable .sceneDepth on its ARWorldTrackingConfiguration before feeding frames.
    public func start(session: ARSession) {
      start()
      if !ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
        fallback("Scene depth unavailable")
      }
    }

    /// Forward from the shared session on the main actor. No ARFrame crosses an
    /// actor boundary here; the App should marshal its existing delegate accordingly.
    public func ingest(_ frame: ARFrame, orientation: UIInterfaceOrientation, viewportSize: CGSize)
      throws
    {
      guard status != .stopped else { return }
      guard case .normal = frame.camera.trackingState,
        viewportSize.width > 0, viewportSize.height > 0,
        let depth = frame.sceneDepth, let confidence = depth.confidenceMap
      else {
        trackingUnavailable()
        return
      }
      let map = depth.depthMap
      let rgb = frame.capturedImage
      guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8,
        CVPixelBufferGetPlaneCount(rgb) == 2
      else { throw LiveSplatError.invalidFrame }
      let format = CVPixelBufferGetPixelFormatType(rgb)
      guard
        format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
          || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
      else { throw LiveSplatError.invalidFrame }
      let width = CVPixelBufferGetWidth(map)
      let height = CVPixelBufferGetHeight(map)
      guard width <= 512, height <= 512, width > 0, height > 0,
        CVPixelBufferGetWidth(confidence) == width, CVPixelBufferGetHeight(confidence) == height
      else {
        throw LiveSplatError.invalidFrame
      }
      guard CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess else {
        throw LiveSplatError.invalidFrame
      }
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      guard CVPixelBufferLockBaseAddress(confidence, .readOnly) == kCVReturnSuccess else {
        throw LiveSplatError.invalidFrame
      }
      defer { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
      guard CVPixelBufferLockBaseAddress(rgb, .readOnly) == kCVReturnSuccess else {
        throw LiveSplatError.invalidFrame
      }
      defer { CVPixelBufferUnlockBaseAddress(rgb, .readOnly) }
      guard let depths = CVPixelBufferGetBaseAddress(map),
        let confidenceBase = CVPixelBufferGetBaseAddress(confidence),
        let luma = CVPixelBufferGetBaseAddressOfPlane(rgb, 0),
        let chroma = CVPixelBufferGetBaseAddressOfPlane(rgb, 1)
      else { throw LiveSplatError.invalidFrame }
      let imageWidth = CVPixelBufferGetWidthOfPlane(rgb, 0)
      let imageHeight = CVPixelBufferGetHeightOfPlane(rgb, 0)
      let chromaWidth = CVPixelBufferGetWidthOfPlane(rgb, 1)
      let chromaHeight = CVPixelBufferGetHeightOfPlane(rgb, 1)
      guard imageWidth > 0, imageHeight > 0, chromaWidth > 0, chromaHeight > 0 else {
        throw LiveSplatError.invalidFrame
      }
      let matrix =
        CVBufferGetAttachment(rgb, kCVImageBufferYCbCrMatrix, nil)?.takeUnretainedValue() as? String
      let bt709 = matrix == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String)
      let videoRange = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
      var d: [Float] = []
      var conf: [UInt8] = []
      var colors: [SplatVector] = []
      d.reserveCapacity(width * height)
      conf.reserveCapacity(width * height)
      colors.reserveCapacity(width * height)
      for y in 0..<height {
        let depthRow = depths.advanced(by: y * CVPixelBufferGetBytesPerRow(map))
          .assumingMemoryBound(to: Float.self)
        let confidenceRow = confidenceBase.advanced(by: y * CVPixelBufferGetBytesPerRow(confidence))
          .assumingMemoryBound(to: UInt8.self)
        let py = min(imageHeight - 1, y * imageHeight / height)
        let cy = min(chromaHeight - 1, y * chromaHeight / height)
        let yRow = luma.advanced(by: py * CVPixelBufferGetBytesPerRowOfPlane(rgb, 0))
          .assumingMemoryBound(to: UInt8.self)
        let cRow = chroma.advanced(by: cy * CVPixelBufferGetBytesPerRowOfPlane(rgb, 1))
          .assumingMemoryBound(to: UInt8.self)
        for x in 0..<width {
          d.append(depthRow[x])
          conf.append(confidenceRow[x])
          let px = min(imageWidth - 1, x * imageWidth / width)
          let cx = min(chromaWidth - 1, x * chromaWidth / width)
          let yy = (Double(yRow[px]) - (videoRange ? 16 : 0)) / (videoRange ? 219 : 255)
          let cb = (Double(cRow[2 * cx]) - 128) / (videoRange ? 224 : 255)
          let cr = (Double(cRow[2 * cx + 1]) - 128) / (videoRange ? 224 : 255)
          func clamp(_ v: Double) -> Double { min(1, max(0, v)) }
          colors.append(
            SplatVector(
              clamp(yy + (bt709 ? 1.5748 : 1.402) * cr),
              clamp(yy - (bt709 ? 0.187324 : 0.344136) * cb - (bt709 ? 0.468124 : 0.714136) * cr),
              clamp(yy + (bt709 ? 1.8556 : 1.772) * cb)))
        }
      }
      let t = frame.camera.transform
      let k = frame.camera.intrinsics
      let camera = try SplatCamera(
        cameraToWorld: SplatMatrix((0..<9).map { Double(t[$0 % 3][$0 / 3]) }),
        position: SplatVector(Double(t[3][0]), Double(t[3][1]), Double(t[3][2])),
        fx: Double(k[0][0]), fy: Double(k[1][1]), cx: Double(k[2][0]), cy: Double(k[2][1]),
        width: imageWidth, height: imageHeight)
      let input = try SplatFrame(
        timestamp: frame.timestamp, camera: camera, width: width, height: height,
        depths: d, confidence: conf, colors: colors)
      let transform = frame.displayTransform(for: orientation, viewportSize: viewportSize)
      setDisplayTransform(
        [transform.a, transform.b, transform.c, transform.d, transform.tx, transform.ty].map(
          Double.init))
      ingest(input)
    }
  }
#endif
