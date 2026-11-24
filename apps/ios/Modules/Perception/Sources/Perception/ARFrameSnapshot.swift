import Foundation

public struct FrameSnapshot: Sendable {
  public var image: CameraImage
  public var depth: DepthImage
  public var camera: CameraGeometry
  public init(image: CameraImage, depth: DepthImage, camera: CameraGeometry) {
    self.image = image
    self.depth = depth
    self.camera = camera
  }
}

#if canImport(ARKit) && canImport(CoreImage)
  @preconcurrency import ARKit
  import CoreImage
  import CoreGraphics
  import CoreVideo
  import Foundation

  /// Owns copies of image/depth data and camera values, never an ARKit frame or camera.
  /// Native YUV copying is synchronous; the expensive RGBA conversion runs on the worker.
  public final class CapturedCameraFrame: @unchecked Sendable {
    fileprivate let image: CVPixelBuffer
    public let depth: DepthImage
    public let camera: CameraGeometry
    public init(frame: ARFrame) throws {
      guard let map = frame.sceneDepth?.depthMap,
        CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32
      else { throw PerceptionError.insufficientDepth }
      image = try Self.copyPixelBuffer(frame.capturedImage)
      camera = CameraGeometry(camera: frame.camera)
      guard CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess else {
        throw PerceptionError.insufficientDepth
      }
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      let width = CVPixelBufferGetWidth(map)
      let height = CVPixelBufferGetHeight(map)
      guard let base = CVPixelBufferGetBaseAddress(map) else {
        throw PerceptionError.insufficientDepth
      }
      var values = [Float](repeating: .nan, count: width * height)
      for y in 0..<height {
        let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(
          to: Float.self)
        for x in 0..<width { values[y * width + x] = row[x] }
      }
      depth = DepthImage(width: width, height: height, metres: values)
    }
    static func copyPixelBuffer(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
      var destination: CVPixelBuffer?
      guard
        CVPixelBufferCreate(
          kCFAllocatorDefault, CVPixelBufferGetWidth(source), CVPixelBufferGetHeight(source),
          CVPixelBufferGetPixelFormatType(source), nil, &destination) == kCVReturnSuccess,
        let destination
      else { throw PerceptionError.insufficientDepth }
      guard CVPixelBufferLockBaseAddress(source, .readOnly) == kCVReturnSuccess else {
        throw PerceptionError.insufficientDepth
      }
      defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
      guard CVPixelBufferLockBaseAddress(destination, []) == kCVReturnSuccess else {
        throw PerceptionError.insufficientDepth
      }
      defer { CVPixelBufferUnlockBaseAddress(destination, []) }
      let planes = CVPixelBufferGetPlaneCount(source)
      guard planes == CVPixelBufferGetPlaneCount(destination) else {
        throw PerceptionError.insufficientDepth
      }
      for plane in 0..<max(1, planes) {
        let src =
          planes == 0
          ? CVPixelBufferGetBaseAddress(source)
          : CVPixelBufferGetBaseAddressOfPlane(source, plane)
        let dst =
          planes == 0
          ? CVPixelBufferGetBaseAddress(destination)
          : CVPixelBufferGetBaseAddressOfPlane(destination, plane)
        let srcStride =
          planes == 0
          ? CVPixelBufferGetBytesPerRow(source)
          : CVPixelBufferGetBytesPerRowOfPlane(source, plane)
        let dstStride =
          planes == 0
          ? CVPixelBufferGetBytesPerRow(destination)
          : CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
        let height =
          planes == 0
          ? CVPixelBufferGetHeight(source)
          : CVPixelBufferGetHeightOfPlane(source, plane)
        guard let src, let dst else { throw PerceptionError.insufficientDepth }
        for y in 0..<height {
          dst.advanced(by: y * dstStride).copyMemory(
            from: src.advanced(by: y * srcStride), byteCount: min(srcStride, dstStride))
        }
      }
      CVBufferPropagateAttachments(source, destination)
      return destination
    }
  }

  extension FrameSnapshot {
    public init(frame: ARFrame) throws {
      self.init(captured: try CapturedCameraFrame(frame: frame))
    }
    public init(captured: CapturedCameraFrame) {
      let width = CVPixelBufferGetWidth(captured.image)
      let height = CVPixelBufferGetHeight(captured.image)
      let ci = CIImage(cvPixelBuffer: captured.image)
      var rgba = [UInt8](repeating: 0, count: width * height * 4)
      let context = CIContext(options: [.cacheIntermediates: false])
      context.render(
        ci, toBitmap: &rgba, rowBytes: width * 4,
        bounds: CGRect(x: 0, y: 0, width: width, height: height),
        format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
      image = CameraImage(width: width, height: height, rgba: rgba)
      depth = captured.depth
      camera = captured.camera
    }
  }

  extension CameraGeometry {
    public init(camera: ARCamera) {
      let k = camera.intrinsics
      let t = camera.transform
      self.init(
        fx: Double(k[0][0]), fy: Double(k[1][1]), cx: Double(k[2][0]), cy: Double(k[2][1]),
        imageWidth: Int(camera.imageResolution.width),
        imageHeight: Int(camera.imageResolution.height),
        transform: (0..<4).flatMap { c in (0..<4).map { Double(t[c][$0]) } })
    }
  }
#endif
