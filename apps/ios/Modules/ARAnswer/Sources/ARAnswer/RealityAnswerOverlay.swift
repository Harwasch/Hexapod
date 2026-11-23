#if canImport(RealityKit) && canImport(AVFoundation) && canImport(UIKit) && os(iOS)
  @preconcurrency import RealityKit
  @preconcurrency import Metal
  import AVFoundation
  import EntitySchema
  import Foundation
  import UIKit

  /// Device-only: RealityKit callbacks, Metal validation, pin hit-testing and speech audio
  /// routing have not been compiled or exercised on Linux.
  @MainActor public final class RealityAnswerOverlay: AnswerOverlaySink {
    private weak var view: ARView?
    private let speech = AVSpeechSynthesizer()
    private let processor: HighlightProcessor
    private var scanning = false
    private var installed = false

    public init(view: ARView) throws {
      self.view = view
      speech.usesApplicationAudioSession = false
      processor = try HighlightProcessor()

    }
    /// Numbered pins live in the SwiftUI viewport, keeping their size independent of depth.
    public func update(entities: [EntitySchema.Entity]) {}
    public func setScanning(_ value: Bool) {
      scanning = value
      if value { setMask(nil) }
    }
    public func setMask(_ mask: HighlightMask?) {
      let active = !scanning && mask?.isValid == true
      processor.setMask(active ? mask : nil)
      if active {
        let processor = processor
        view?.renderCallbacks.postProcess = { @Sendable context in processor.render(context) }
        installed = true
      } else if installed {
        view?.renderCallbacks.postProcess = nil
        installed = false
      }
    }
    public func speak(_ text: String) {
      guard !text.isEmpty else { return }
      speech.stopSpeaking(at: .immediate)
      let utterance = AVSpeechUtterance(string: text)
      speech.speak(utterance)
    }
    public func stopSpeaking() { speech.stopSpeaking(at: .immediate) }
    public func remove() {
      stopSpeaking()
      setMask(nil)
    }
  }

  /// Immutable Metal objects are shared safely; the mask mailbox is protected by lock.
  /// No RealityKit render callback reaches into UI/MainActor state.
  private final class HighlightProcessor: @unchecked Sendable {
    private let lock = NSLock()
    private var texture: MTLTexture?
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    init() throws {
      guard let device = MTLCreateSystemDefaultDevice(),
        let function = try device.makeDefaultLibrary(bundle: .module)
          .makeFunction(name: "subjectHighlight")
      else { throw CocoaError(.fileReadUnknown) }
      self.device = device
      pipeline = try device.makeComputePipelineState(function: function)
    }
    func setMask(_ mask: HighlightMask?) {
      guard let mask, mask.isValid else {
        lock.withLock { texture = nil }
        return
      }
      let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: mask.width, height: mask.height, mipmapped: false)
      descriptor.usage = .shaderRead
      descriptor.storageMode = .shared
      let next = device.makeTexture(descriptor: descriptor)
      mask.pixels.withUnsafeBytes { bytes in
        if let base = bytes.baseAddress {
          next?.replace(
            region: MTLRegionMake2D(0, 0, mask.width, mask.height), mipmapLevel: 0,
            withBytes: base, bytesPerRow: mask.width)
        }
      }
      // Immutable for its GPU lifetime. Upload once per changed mask, not every display frame.
      lock.withLock { texture = next }
    }
    func render(_ context: ARView.PostProcessContext) {
      guard let texture = lock.withLock({ texture }),
        let encoder = context.commandBuffer.makeComputeCommandEncoder()
      else {
        let blit = context.commandBuffer.makeBlitCommandEncoder()
        blit?.copy(from: context.sourceColorTexture, to: context.targetColorTexture)
        blit?.endEncoding()
        return
      }
      var enabled: UInt32 = 1
      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(context.sourceColorTexture, index: 0)
      encoder.setTexture(context.targetColorTexture, index: 1)
      encoder.setTexture(texture, index: 2)
      encoder.setBytes(&enabled, length: MemoryLayout<UInt32>.size, index: 0)
      let w = pipeline.threadExecutionWidth
      let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
      encoder.dispatchThreads(
        MTLSize(
          width: context.targetColorTexture.width,
          height: context.targetColorTexture.height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
      encoder.endEncoding()
    }
  }
#endif
