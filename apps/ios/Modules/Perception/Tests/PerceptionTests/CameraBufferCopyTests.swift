#if canImport(ARKit) && canImport(CoreImage)
  import CoreVideo
  import Testing
  @testable import Perception

  @Test(arguments: [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange])
  func nativeCameraBufferCopyOwnsEveryPlane(format: OSType) throws {
    var buffer: CVPixelBuffer?
    #expect(
      CVPixelBufferCreate(kCFAllocatorDefault, 8, 8, format, nil, &buffer) == kCVReturnSuccess)
    let source = try #require(buffer)
    CVPixelBufferLockBaseAddress(source, [])
    let planes = CVPixelBufferGetPlaneCount(source)
    for plane in 0..<max(1, planes) {
      let base = try #require(
        planes == 0
          ? CVPixelBufferGetBaseAddress(source)
          : CVPixelBufferGetBaseAddressOfPlane(source, plane))
      base.storeBytes(of: UInt8(60 + plane), as: UInt8.self)
    }
    CVPixelBufferUnlockBaseAddress(source, [])
    let copy = try CapturedCameraFrame.copyPixelBuffer(source)
    CVPixelBufferLockBaseAddress(source, [])
    CVPixelBufferLockBaseAddress(copy, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(copy, .readOnly)
      CVPixelBufferUnlockBaseAddress(source, [])
    }
    for plane in 0..<max(1, planes) {
      let original = try #require(
        planes == 0
          ? CVPixelBufferGetBaseAddress(source)
          : CVPixelBufferGetBaseAddressOfPlane(source, plane))
      let owned = try #require(
        planes == 0
          ? CVPixelBufferGetBaseAddress(copy)
          : CVPixelBufferGetBaseAddressOfPlane(copy, plane))
      original.storeBytes(of: UInt8(0), as: UInt8.self)
      #expect(owned.load(as: UInt8.self) == UInt8(60 + plane))
    }
  }
#endif
