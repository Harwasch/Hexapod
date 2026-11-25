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
    private var worldHighlightEnabled = false
    public var worldSurfaceRenderer: WorldSurfaceRenderer? { processor.worldSurfaces }
    public var pipelineChecks: [OverlayPipelineCheck] { processor.pipelineChecks }

    public init(view: ARView) throws {
      self.view = view
      speech.usesApplicationAudioSession = false
      processor = HighlightProcessor()

    }
    /// Numbered pins live in the SwiftUI viewport, keeping their size independent of depth.
    public func update(entities: [EntitySchema.Entity]) {}
    public func setScanning(_ value: Bool) {
      scanning = value
      if value { setMask(nil) }
    }
    public func enableWorldHighlight() {
      guard !scanning, !worldHighlightEnabled, processor.worldSurfaces != nil else { return }
      worldHighlightEnabled = true
      processor.setWorldEnabled(true)
      let processor = processor
      view?.renderCallbacks.postProcess = { @Sendable context in processor.render(context) }
      installed = true
    }
    public func setMask(_ mask: HighlightMask?) {
      worldHighlightEnabled = false
      processor.setWorldEnabled(false)
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
      worldSurfaceRenderer?.clear()
    }
  }

  /// Immutable Metal objects are shared safely; the mask mailbox is protected by lock.
  /// No RealityKit render callback reaches into UI/MainActor state.
  private final class HighlightProcessor: @unchecked Sendable {
    private enum BuildError: Error { case noMetalDevice, missingSubjectHighlightFunction }
    private let lock = NSLock()
    private var texture: MTLTexture?
    private var worldEnabled = false
    let worldSurfaces: WorldSurfaceRenderer?
    let pipelineChecks: [OverlayPipelineCheck]
    private let device: MTLDevice?
    private let pipeline: MTLComputePipelineState?
    init() {
      let device = MTLCreateSystemDefaultDevice()
      self.device = device
      let built = OverlayPipelineCheck.build(
        highlight: { () throws -> MTLComputePipelineState in
          guard let device else { throw BuildError.noMetalDevice }
          guard
            let function = try device.makeDefaultLibrary(bundle: .module)
              .makeFunction(name: "subjectHighlight")
          else { throw BuildError.missingSubjectHighlightFunction }
          return try device.makeComputePipelineState(function: function)
        },
        surface: { () throws -> WorldSurfaceRenderer in
          guard let device else { throw BuildError.noMetalDevice }
          return try WorldSurfaceRenderer(device: device)
        })
      pipeline = built.highlight
      worldSurfaces = built.surface
      pipelineChecks = built.checks
    }
    func setWorldEnabled(_ value: Bool) {
      lock.withLock {
        worldEnabled = value
        texture = nil
      }
    }
    func setMask(_ mask: HighlightMask?) {
      guard let mask, mask.isValid, let device else {
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
      let state = lock.withLock { (worldEnabled, texture) }
      let mask = state.0 ? worldSurfaces?.mask(context) : state.1
      guard let texture = mask, let pipeline,
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
