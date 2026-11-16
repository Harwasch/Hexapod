#if canImport(MetalKit) && canImport(UIKit) && canImport(SwiftUI)
  @preconcurrency import MetalKit
  import SwiftUI
  import UIKit

  /// Place above the App's RealityKit ARView in a ZStack. Transparent premultiplied
  /// output composites with RealityKit's camera; this host neither owns nor resets it.
  /// No captions or text are added. The App polls coverageFraction/status as needed.
  public struct LiveSplatView: UIViewRepresentable {
    public let renderer: DefaultLiveSplatRenderer
    public init(renderer: DefaultLiveSplatRenderer) { self.renderer = renderer }
    public func makeUIView(context: Context) -> LiveSplatOverlay {
      LiveSplatOverlay(renderer: renderer)
    }
    public func updateUIView(_ view: LiveSplatOverlay, context: Context) {}
  }

  @MainActor public final class LiveSplatOverlay: UIView {
    private let renderer: DefaultLiveSplatRenderer
    private let points: PointOverlay
    private var metalView: MTKView?
    private var driver: MetalSplatDriver?
    private var link: CADisplayLink?
    private var proxy: DisplayProxy?
    public init(renderer: DefaultLiveSplatRenderer) {
      self.renderer = renderer
      points = PointOverlay(renderer: renderer)
      super.init(frame: .zero)
      isUserInteractionEnabled = false
      backgroundColor = .clear
      isOpaque = false
      addSubview(points)
      // Automatic remains a capability-safe preview until a device acceptance run.
      if renderer.configuration.mode == .experimentalGaussian,
        let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7)
      {
        do {
          let view = MTKView(frame: .zero, device: device)
          let driver = try MetalSplatDriver(renderer: renderer, device: device)
          view.delegate = driver
          view.framebufferOnly = false
          view.colorPixelFormat = .bgra8Unorm
          view.isOpaque = false
          view.backgroundColor = .clear
          view.clearColor = MTLClearColorMake(0, 0, 0, 0)
          view.preferredFramesPerSecond = 30
          // Bounded preview resolution, independently of a full-resolution capture.
          view.autoResizeDrawable = false
          addSubview(view)
          metalView = view
          self.driver = driver
        } catch { renderer.fallback("Metal pipeline unavailable: \(error)") }
      }
    }
    required init?(coder: NSCoder) { nil }
    public override func layoutSubviews() {
      super.layoutSubviews()
      points.frame = bounds
      metalView?.frame = bounds
      if bounds.width > 0, bounds.height > 0 {
        let scale = 320 / max(bounds.width, bounds.height)
        metalView?.drawableSize = CGSize(
          width: max(1, (bounds.width * scale).rounded()),
          height: max(1, (bounds.height * scale).rounded()))
      }
    }
    public override func didMoveToWindow() {
      super.didMoveToWindow()
      link?.invalidate()
      link = nil
      proxy = nil
      metalView?.isPaused = window == nil
      if window != nil {
        let proxy = DisplayProxy(owner: self)
        let link = CADisplayLink(target: proxy, selector: #selector(DisplayProxy.tick(_:)))
        self.proxy = proxy
        self.link = link
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 15, maximum: 30, preferred: 30)
        link.add(to: .main, forMode: .common)
      }
    }
    fileprivate func refresh() {
      if driver != nil, renderer.status == .lidarPreview { renderer.activateGaussian() }
      let gaussian = renderer.status == .experimentalGaussian
      metalView?.isHidden = !gaussian || renderer.latestFrame == nil
      points.isHidden = gaussian || renderer.status == .stopped
      points.setNeedsDisplay()
    }
  }

  @MainActor private final class DisplayProxy: NSObject {
    weak var owner: LiveSplatOverlay?
    init(owner: LiveSplatOverlay) { self.owner = owner }
    @objc func tick(_ sender: CADisplayLink) {
      guard let owner else {
        sender.invalidate()
        return
      }
      owner.refresh()
    }
  }

  @MainActor private final class PointOverlay: UIView {
    let renderer: DefaultLiveSplatRenderer
    init(renderer: DefaultLiveSplatRenderer) {
      self.renderer = renderer
      super.init(frame: .zero)
      backgroundColor = .clear
      isOpaque = false
      isUserInteractionEnabled = false
    }
    required init?(coder: NSCoder) { nil }
    override func draw(_ rect: CGRect) {
      guard renderer.status != .stopped, let frame = renderer.latestFrame,
        let context = UIGraphicsGetCurrentContext()
      else { return }
      let t = renderer.displayTransform
      func point(_ u: Double, _ v: Double) -> CGPoint {
        CGPoint(
          x: (t[0] * u + t[2] * v + t[4]) * bounds.width,
          y: (t[1] * u + t[3] * v + t[5]) * bounds.height)
      }
      context.setFillColor(UIColor.cyan.withAlphaComponent(0.12).cgColor)
      for i in renderer.coverage.covered.indices where !renderer.coverage.covered[i] {
        let u = Double(i % 32) / 32
        let v = Double(i / 32) / 24
        context.beginPath()
        context.move(to: point(u, v))
        context.addLine(to: point(u + 1 / 32, v))
        context.addLine(to: point(u + 1 / 32, v + 1 / 24))
        context.addLine(to: point(u, v + 1 / 24))
        context.closePath()
        context.fillPath()
      }
      for g in renderer.splats {
        guard let p = frame.camera.project(g.position), p.x >= 0, p.y >= 0,
          p.x < Double(frame.camera.width), p.y < Double(frame.camera.height)
        else { continue }
        let screen = point(p.x / Double(frame.camera.width), p.y / Double(frame.camera.height))
        context.setFillColor(
          UIColor(red: g.color.x, green: g.color.y, blue: g.color.z, alpha: 0.8).cgColor)
        context.fillEllipse(in: CGRect(x: screen.x - 1.5, y: screen.y - 1.5, width: 3, height: 3))
      }
    }
  }

  // DEVICE VALIDATION REQUIRED: this entire path is excluded by Linux. An explicit
  // experimental setting is required; an unavailable pipeline or measured overrun
  // returns to the same host's LiDAR preview. At most one command buffer is in flight.
  @MainActor private final class MetalSplatDriver: NSObject, @preconcurrency MTKViewDelegate {
    let renderer: DefaultLiveSplatRenderer
    let device: MTLDevice
    let queue: MTLCommandQueue
    let project: MTLComputePipelineState, bin: MTLComputePipelineState
    let composite: MTLComputePipelineState, optimize: MTLComputePipelineState
    var pending: MTLCommandBuffer?
    var pendingSplats: MTLBuffer?
    var pendingSnapshot: [Gaussian] = []
    var pendingSession: UInt64 = 0
    var pendingIDs: [UInt64] = []
    var lastSubmittedGeneration: UInt64?
    var offset: UInt32 = 0
    init(renderer: DefaultLiveSplatRenderer, device: MTLDevice) throws {
      self.renderer = renderer
      self.device = device
      guard let queue = device.makeCommandQueue(),
        let url = Bundle.module.url(
          forResource: "LiveSplat", withExtension: "metal", subdirectory: "Shaders")
      else {
        throw LiveSplatError.unavailable
      }
      self.queue = queue
      let options = MTLCompileOptions()
      options.fastMathEnabled = false
      let library = try device.makeLibrary(
        source: String(contentsOf: url, encoding: .utf8), options: options)
      func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
          throw LiveSplatError.unavailable
        }
        return try device.makeComputePipelineState(function: function)
      }
      project = try pipeline("projectGaussians")
      bin = try pipeline("binAndSort")
      composite = try pipeline("compositeGaussians")
      optimize = try pipeline("optimizeGaussians")
      super.init()
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    func cameraBytes(_ frame: SplatFrame, width: Int, height: Int) -> [Float] {
      let c = frame.camera
      let sx = Double(width) / Double(c.width)
      let sy = Double(height) / Double(c.height)
      return
        (c.worldToOptical.values + c.position.values
        + [c.fx * sx, c.fy * sy, c.cx * sx, c.cy * sy, Double(width), Double(height)]).map(
          Float.init)
    }
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
      guard !values.isEmpty else { throw LiveSplatError.unavailable }
      let result = values.withUnsafeBytes { bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let result else { throw LiveSplatError.unavailable }
      return result
    }
    func allocate(_ length: Int) throws -> MTLBuffer {
      guard let b = device.makeBuffer(length: length, options: .storageModePrivate) else {
        throw LiveSplatError.unavailable
      }
      return b
    }
    func encode(
      _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState,
      buffers: [MTLBuffer], count: Int, texture: MTLTexture? = nil
    ) throws {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw LiveSplatError.unavailable
      }
      encoder.setComputePipelineState(pipeline)
      for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
      if let texture {
        encoder.setTexture(texture, index: 0)
        encoder.dispatchThreads(
          MTLSize(width: texture.width, height: texture.height, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
      } else {
        encoder.dispatchThreads(
          MTLSize(width: count, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
      }
      encoder.endEncoding()
    }
    func draw(in view: MTKView) {
      if let command = pending {
        if command.status == .error {
          renderer.fallback("Metal command failed")
          pending = nil
          pendingSplats = nil
          return
        }
        guard command.status == .completed else { return }
        if let data = pendingSplats {
          let floats = data.contents().assumingMemoryBound(to: Float.self)
          for i in pendingSnapshot.indices {
            pendingSnapshot[i].parameters = (0..<14).map { Double(floats[i * 14 + $0]) }
          }
          renderer.acceptGPU(
            pendingSnapshot, ids: pendingIDs, seconds: command.gpuEndTime - command.gpuStartTime,
            session: pendingSession)
        }
        pending = nil
        pendingSplats = nil
        pendingSnapshot = []
      }
      guard renderer.status != .stopped, !renderer.budget.exceeded else { return }
      if case .fallback = renderer.status { return }
      if ProcessInfo.processInfo.thermalState == .serious
        || ProcessInfo.processInfo.thermalState == .critical
        || ProcessInfo.processInfo.isLowPowerModeEnabled
      {
        renderer.fallback("Thermal or power limit")
        return
      }
      renderer.activateGaussian()
      guard let frame = renderer.latestFrame, !renderer.splats.isEmpty,
        lastSubmittedGeneration != renderer.generation, let drawable = view.currentDrawable
      else { return }
      do {
        let count = renderer.splats.count
        let width = frame.width
        let height = frame.height
        let tileCount = ((width + 15) / 16) * ((height + 15) / 16)
        let splats = try buffer(renderer.splats.flatMap { $0.parameters.map(Float.init) })
        let projections = try allocate(count * 8 * MemoryLayout<Float>.stride)
        let bins = try allocate(tileCount * 65 * MemoryLayout<UInt32>.stride)
        let overflow = try allocate(tileCount * MemoryLayout<UInt32>.stride)
        let settings = try buffer([UInt32(count), UInt32(width), UInt32(height), offset])
        let camera = try buffer(cameraBytes(frame, width: width, height: height))
        let coverage = try buffer(renderer.coverage.covered.map { UInt32($0 ? 1 : 0) })
        let t = renderer.displayTransform
        let inverse = CGAffineTransform(a: t[0], b: t[1], c: t[2], d: t[3], tx: t[4], ty: t[5])
          .inverted()
        let transform = try buffer(
          [inverse.a, inverse.b, inverse.c, inverse.d, inverse.tx, inverse.ty].map(Float.init))
        guard let command = queue.makeCommandBuffer() else { throw LiveSplatError.unavailable }
        // Two bounded steps, on the two most recent keyframes. A reused keyframe
        // still contributes a new step as the current view/seed set changes.
        for keyframe in renderer.recentKeyframes.suffix(renderer.budget.optimizationSteps) {
          let keyCamera = try buffer(
            cameraBytes(keyframe, width: keyframe.width, height: keyframe.height))
          let rgb = try buffer(keyframe.colors.flatMap { $0.values.map(Float.init) })
          try encode(
            command, pipeline: optimize, buffers: [splats, keyCamera, rgb, settings],
            count: min(count, 32))
        }
        try encode(
          command, pipeline: project, buffers: [splats, projections, camera, settings], count: count
        )
        try encode(
          command, pipeline: bin, buffers: [projections, bins, overflow, settings], count: tileCount
        )
        try encode(
          command, pipeline: composite,
          buffers: [splats, projections, bins, settings, coverage, transform, overflow],
          count: 0, texture: drawable.texture)
        pendingSnapshot = renderer.splats
        pendingSession = renderer.sessionGeneration
        pendingIDs = renderer.splatIDs
        pendingSplats = splats
        pending = command
        lastSubmittedGeneration = renderer.generation
        offset = (offset + 32) % UInt32(count)
        command.present(drawable)
        command.commit()
      } catch { renderer.fallback("Metal resources unavailable: \(error)") }
    }
  }
#endif
