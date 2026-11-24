#if canImport(RealityKit) && canImport(Metal) && os(iOS)
  @preconcurrency import Metal
  @preconcurrency import RealityKit
  import Foundation
  import Perception

  /// Immutable geometry buffers are uploaded by the inference worker, never by the main thread.
  /// The postprocess callback reprojects them on the GPU for every display frame.
  public final class WorldSurfaceRenderer: @unchecked Sendable {
    private struct Mesh {
      let points: MTLBuffer
      let indices: MTLBuffer
      let count: Int
    }
    private struct Slot {
      var texture: MTLTexture?
      var busy = false
    }
    private let lock = NSLock()
    private let device: MTLDevice
    private let pipeline: MTLRenderPipelineState
    private var meshes: [Mesh] = []
    private var generation = 0
    private var camera: SurfaceProjection?
    private var slots = Array(repeating: Slot(), count: PerceptionLimits.maximumGPUFramesInFlight)

    public init(device: MTLDevice) throws {
      self.device = device
      let library = try device.makeDefaultLibrary(bundle: .module)
      let descriptor = MTLRenderPipelineDescriptor()
      descriptor.vertexFunction = library.makeFunction(name: "worldSurfaceVertex")
      descriptor.fragmentFunction = library.makeFunction(name: "worldSurfaceMask")
      descriptor.colorAttachments[0].pixelFormat = .r8Unorm
      pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    }
    public func beginGeneration(_ value: Int) {
      lock.withLock {
        generation = value
        meshes = []
      }
    }
    public func setSurfaces(_ surfaces: [SubjectSurface], generation expected: Int) {
      guard lock.withLock({ generation == expected }) else { return }
      let next: [Mesh] = surfaces.compactMap { surface in
        guard !surface.mesh.points.isEmpty, !surface.mesh.triangles.isEmpty,
          surface.mesh.triangles.count % 3 == 0,
          surface.mesh.triangles.allSatisfy({ Int($0) < surface.mesh.points.count })
        else { return nil }
        let vertices = surface.mesh.points.map {
          SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 1)
        }
        let points = vertices.withUnsafeBytes { bytes in
          bytes.baseAddress.flatMap {
            device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared)
          }
        }
        let indices = surface.mesh.triangles.withUnsafeBytes { bytes in
          bytes.baseAddress.flatMap {
            device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared)
          }
        }
        guard let points, let indices else { return nil }
        return Mesh(points: points, indices: indices, count: surface.mesh.triangles.count)
      }
      lock.withLock { if generation == expected { meshes = next } }
    }
    public func setCamera(_ value: SurfaceProjection?) { lock.withLock { camera = value } }
    public func clear() {
      lock.withLock {
        meshes = []
        camera = nil
      }
    }

    func mask(_ context: ARView.PostProcessContext) -> MTLTexture? {
      let now = ProcessInfo.processInfo.systemUptime
      let state = lock.withLock { (meshes, camera) }
      guard let camera = state.1, camera.isCurrent(at: now), let uniforms = camera.uniforms else {
        return nil
      }
      // Geometry lives as long as the selection; only the display camera must be fresh.
      let visible = state.0
      guard !visible.isEmpty else { return nil }
      let width = context.targetColorTexture.width
      let height = context.targetColorTexture.height
      let acquired: (Int, MTLTexture)? = lock.withLock {
        guard let index = slots.firstIndex(where: { !$0.busy }) else { return nil }
        if slots[index].texture?.width != width || slots[index].texture?.height != height {
          let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
          descriptor.storageMode = .private
          descriptor.usage = [.renderTarget, .shaderRead]
          slots[index].texture = device.makeTexture(descriptor: descriptor)
        }
        guard let texture = slots[index].texture else { return nil }
        slots[index].busy = true
        return (index, texture)
      }
      guard let (slot, texture) = acquired else { return nil }  // Drop an overlay frame rather than wait for the GPU.
      let pass = MTLRenderPassDescriptor()
      pass.colorAttachments[0].texture = texture
      pass.colorAttachments[0].loadAction = .clear
      pass.colorAttachments[0].storeAction = .store
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
      guard let encoder = context.commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
        lock.withLock { slots[slot].busy = false }
        return nil
      }
      context.commandBuffer.addCompletedHandler { [weak self] _ in
        guard let self else { return }
        self.lock.withLock { self.slots[slot].busy = false }
      }
      encoder.setRenderPipelineState(pipeline)
      encoder.setCullMode(.none)
      uniforms.withUnsafeBytes { bytes in
        if let base = bytes.baseAddress {
          encoder.setVertexBytes(base, length: bytes.count, index: 1)
        }
      }
      for mesh in visible {
        encoder.setVertexBuffer(mesh.points, offset: 0, index: 0)
        encoder.drawIndexedPrimitives(
          type: .triangle, indexCount: mesh.count, indexType: .uint32, indexBuffer: mesh.indices,
          indexBufferOffset: 0)
      }
      encoder.endEncoding()
      return texture
    }
  }
#endif
