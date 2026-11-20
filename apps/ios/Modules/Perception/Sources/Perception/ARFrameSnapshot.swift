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

  extension FrameSnapshot {
    /// Copies ARKit-owned buffers before async inference. Device-only, unverified on Linux.
    public init(frame: ARFrame) throws {
      guard let map = frame.sceneDepth?.depthMap,
        CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32
      else {
        throw PerceptionError.insufficientDepth
      }
      let width = CVPixelBufferGetWidth(frame.capturedImage)
      let height = CVPixelBufferGetHeight(frame.capturedImage)
      let ci = CIImage(cvPixelBuffer: frame.capturedImage)
      var rgba = [UInt8](repeating: 0, count: width * height * 4)
      let context = CIContext(options: [.cacheIntermediates: false])
      // Core Image's bitmap row convention requires an asymmetric fixture check on device.
      context.render(
        ci, toBitmap: &rgba, rowBytes: width * 4,
        bounds: CGRect(x: 0, y: 0, width: width, height: height),
        format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
      image = CameraImage(width: width, height: height, rgba: rgba)
      CVPixelBufferLockBaseAddress(map, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      let dw = CVPixelBufferGetWidth(map)
      let dh = CVPixelBufferGetHeight(map)
      guard let base = CVPixelBufferGetBaseAddress(map) else {
        throw PerceptionError.insufficientDepth
      }
      var values = [Float](repeating: .nan, count: dw * dh)
      for y in 0..<dh {
        let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(
          to: Float.self)
        for x in 0..<dw { values[y * dw + x] = row[x] }
      }
      depth = DepthImage(width: dw, height: dh, metres: values)
      let k = frame.camera.intrinsics
      let t = frame.camera.transform
      camera = CameraGeometry(
        fx: Double(k[0][0]), fy: Double(k[1][1]), cx: Double(k[2][0]), cy: Double(k[2][1]),
        imageWidth: width, imageHeight: height,
        transform: (0..<4).flatMap { c in (0..<4).map { Double(t[c][$0]) } })
    }
  }
#endif
