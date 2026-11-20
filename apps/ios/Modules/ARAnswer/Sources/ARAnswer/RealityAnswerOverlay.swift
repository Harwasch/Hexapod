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
    private var anchors: [AnchorEntity] = []
    private var entities: [EntitySchema.Entity] = []
    private let speech = AVSpeechSynthesizer()
    private let processor: HighlightProcessor

    public init(view: ARView) throws {
      self.view = view
      speech.usesApplicationAudioSession = false
      processor = try HighlightProcessor()
      let processor = processor
      view.renderCallbacks.postProcess = { @Sendable context in processor.render(context) }
    }
    public func update(entities: [EntitySchema.Entity]) {
      guard let view else { return }
      for anchor in anchors { view.scene.removeAnchor(anchor) }
      anchors.removeAll()
      self.entities = entities
      for (index, entity) in entities.enumerated() {
        let p = entity.anchor.world
        let anchor = AnchorEntity(world: SIMD3(Float(p.x), Float(p.y), Float(p.z)))
        let mesh = MeshResource.generateText(
          String(index + 1), extrusionDepth: 0.002,
          font: .systemFont(ofSize: 0.10, weight: .bold))
        let pin = ModelEntity(mesh: mesh, materials: [UnlitMaterial(color: .systemMint)])
        pin.name = "hex-pin-\(index)"
        pin.generateCollisionShapes(recursive: true)
        anchor.addChild(pin)
        view.scene.addAnchor(anchor)
        anchors.append(anchor)
      }
    }
    public func updateCamera() {
      guard let view else { return }
      for anchor in anchors {
        anchor.children.first?.setOrientation(view.cameraTransform.rotation, relativeTo: nil)
      }
    }
    public func setMask(_ mask: HighlightMask?) { processor.setMask(mask) }
    public func pin(at point: CGPoint) -> EntitySchema.Entity? {
      guard var hit = view?.entity(at: point) else { return nil }
      while !hit.name.hasPrefix("hex-pin-") {
        guard let parent = hit.parent else { return nil }
        hit = parent
      }
      guard let index = Int(hit.name.dropFirst("hex-pin-".count)), entities.indices.contains(index)
      else { return nil }
      return entities[index]
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
      for anchor in anchors { view?.scene.removeAnchor(anchor) }
      anchors.removeAll()
      view?.renderCallbacks.postProcess = nil
    }
  }

  /// Immutable Metal objects are shared safely; the mask mailbox is protected by lock.
  /// No RealityKit render callback reaches into UI/MainActor state.
  private final class HighlightProcessor: @unchecked Sendable {
    private let lock = NSLock()
    private var mask: HighlightMask?
    private let pipeline: MTLComputePipelineState
    init() throws {
      guard let device = MTLCreateSystemDefaultDevice(),
        let function = try device.makeDefaultLibrary(bundle: .module)
          .makeFunction(name: "subjectHighlight")
      else { throw CocoaError(.fileReadUnknown) }
      pipeline = try device.makeComputePipelineState(function: function)
    }
    func setMask(_ mask: HighlightMask?) {
      lock.lock()
      self.mask = mask?.isValid == true ? mask : nil
      lock.unlock()
    }
    func render(_ context: ARView.PostProcessContext) {
      lock.lock()
      let current = mask
      lock.unlock()
      let width = current?.width ?? 1
      let height = current?.height ?? 1
      let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm,
        width: width, height: height, mipmapped: false)
      descriptor.usage = .shaderRead
      descriptor.storageMode = .shared
      guard let texture = context.device.makeTexture(descriptor: descriptor),
        let encoder = context.commandBuffer.makeComputeCommandEncoder()
      else { return }
      let pixels = current?.pixels ?? [0]
      pixels.withUnsafeBytes {
        if let base = $0.baseAddress {
          texture.replace(
            region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
            withBytes: base, bytesPerRow: width)
        }
      }
      var enabled: UInt32 = current == nil ? 0 : 1
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
