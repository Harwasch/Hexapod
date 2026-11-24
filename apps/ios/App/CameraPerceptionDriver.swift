#if canImport(ARKit) && canImport(UIKit)
  @preconcurrency import ARKit
  import ARAnswer
  import Foundation
  import EntitySchema
  import Perception
  import UIKit

  struct CameraSelectionResult: Sendable {
    let selection: SceneSelection
    let generation: Int
    let tapped: Bool
    let card: EntitySchema.Entity?
  }

  /// ARFrame is an immutable retained snapshot. Its buffers are copied only by the worker.
  private final class CameraInferenceFrame: @unchecked Sendable {
    let frame: ARFrame
    let mapping: [Double]
    let capturedAt: TimeInterval
    let generation: Int
    let tap: ImagePoint?
    init(frame: ARFrame, mapping: [Double], generation: Int, tap: ImagePoint? = nil) {
      self.frame = frame
      self.mapping = mapping
      self.generation = generation
      self.tap = tap
      capturedAt = frame.timestamp
    }
  }

  /// Fixed-cost frame admission on the AR delegate, one request and one replacing mailbox.
  final class CameraPerceptionDriver: @unchecked Sendable {
    private final class State: @unchecked Sendable {
      let lock = NSLock()
      var size = CGSize.zero
      var orientation = UIInterfaceOrientation.portrait
      var generation = 0
      var enabled = true
      var taps = InferenceIntentQueue<CameraInferenceFrame>()
      var needsReset = false
      var renderer: WorldSurfaceRenderer?
      var result: CameraSelectionResult?
    }
    private let state: State
    private let worker: ForegroundInferenceLoop<CameraInferenceFrame, CameraSelectionResult>
    init(scene: ScenePerception, renderer: WorldSurfaceRenderer?) {
      let state = State()
      self.state = state
      state.renderer = renderer
      worker = ForegroundInferenceLoop(
        process: { input in
          let request = state.lock.withLock { () -> (CameraInferenceFrame, Bool)? in
            guard state.enabled, state.generation == input.generation else { return nil }
            let reset = state.needsReset
            state.needsReset = false
            while let tap = state.taps.take() {
              guard tap.generation == input.generation else { continue }
              if PerceptionLimits.isCurrent(
                capturedAt: tap.capturedAt,
                now: ProcessInfo.processInfo.systemUptime,
                maximumAge: PerceptionLimits.maximumTapAge)
              {
                return (tap, reset)
              }
            }
            return (input, reset)
          }
          guard let request else { throw CancellationError() }
          let chosen = request.0
          if request.1 { await scene.clear() }
          let snapshot = try FrameSnapshot(frame: chosen.frame)
          let previousCount = await scene.selection().entities.count
          let selection = try await scene.foregroundFrame(
            snapshot, viewportToImage: chosen.mapping,
            capturedAt: chosen.capturedAt, tap: chosen.tap)
          return CameraSelectionResult(
            selection: selection, generation: chosen.generation, tapped: chosen.tap != nil,
            card: chosen.tap != nil && selection.entities.count > previousCount
              ? selection.entities.last : nil)
        },
        result: { output in
          let renderer: WorldSurfaceRenderer? = state.lock.withLock {
            guard state.enabled, state.generation == output.generation else { return nil }
            // The UI consumes one latest result on its summary timer. Preserve a tap's card intent.
            if let previous = state.result, previous.tapped, !output.tapped {
              state.result = CameraSelectionResult(
                selection: output.selection, generation: output.generation,
                tapped: true, card: previous.card)
            } else {
              state.result = output
            }
            return state.renderer
          }
          renderer?.setSurfaces(output.selection.surfaces, generation: output.generation)
        })
    }
    func setRenderer(_ renderer: WorldSurfaceRenderer?) {
      let generation = state.lock.withLock {
        state.renderer = renderer
        return state.generation
      }
      state.lock.withLock { state.renderer }?.beginGeneration(generation)
    }
    func resetSelection(enabled: Bool) {
      worker.setEnabled(false)
      let generation = state.lock.withLock {
        state.generation += 1
        state.enabled = enabled
        state.taps.clear()
        state.result = nil
        state.needsReset = true
        return state.generation
      }
      state.lock.withLock { state.renderer }?.beginGeneration(generation)
      worker.setEnabled(enabled)
    }
    func takeResult() -> CameraSelectionResult? {
      state.lock.withLock {
        let result = state.result
        state.result = nil
        return result
      }
    }
    var metrics: InferenceMetrics { worker.metrics }
    func isCurrent(_ generation: Int) -> Bool {
      state.lock.withLock { state.enabled && state.generation == generation }
    }
    func setEnabled(_ enabled: Bool) {
      let generation: Int? = state.lock.withLock {
        guard state.enabled != enabled else { return nil }
        state.enabled = enabled
        state.generation += 1
        state.taps.clear()
        state.result = nil
        return state.generation
      }
      guard let generation else { return }
      state.lock.withLock { state.renderer }?.beginGeneration(generation)
      worker.setEnabled(enabled)
    }
    func waitUntilIdle() async { await worker.waitUntilIdle() }
    func setViewport(size: CGSize, orientation: UIInterfaceOrientation) {
      state.lock.withLock {
        state.size = size
        state.orientation = orientation
      }
    }
    func offer(_ frame: ARFrame) {
      guard let input = input(frame) else { return }
      worker.offer(input, capturedAt: input.capturedAt)
    }
    func select(_ point: ImagePoint, frame: ARFrame) -> Bool {
      guard let base = input(frame),
        let mapped = ViewportMapping(viewportToImage: base.mapping).imagePoint(
          x: point.x, y: point.y)
      else { return false }
      let tap = CameraInferenceFrame(
        frame: frame, mapping: base.mapping, generation: base.generation,
        tap: ImagePoint(x: mapped.x, y: mapped.y))
      guard
        state.lock.withLock({
          state.enabled && state.generation == tap.generation && state.taps.offer(tap)
        })
      else { return false }
      worker.offer(base, capturedAt: base.capturedAt)
      return true
    }
    /// Called by CADisplayLink with the latest ARFrame, independent of Vision completion.
    func display(_ frame: ARFrame) {
      guard case .normal = frame.camera.trackingState else {
        state.lock.withLock { state.renderer }?.setCamera(nil)
        return
      }
      let viewport = state.lock.withLock { (state.size, state.orientation) }
      guard viewport.0.width > 0, viewport.0.height > 0 else { return }
      let inverse = frame.displayTransform(for: viewport.1, viewportSize: viewport.0).inverted()
      let mapping = [inverse.a, inverse.b, inverse.c, inverse.d, inverse.tx, inverse.ty].map(
        Double.init)
      state.lock.withLock { state.renderer }?.setCamera(
        SurfaceProjection(
          camera: CameraGeometry(camera: frame.camera),
          viewportToImage: mapping, capturedAt: frame.timestamp))
    }
    private func input(_ frame: ARFrame) -> CameraInferenceFrame? {
      guard case .normal = frame.camera.trackingState, frame.sceneDepth != nil else { return nil }
      let viewport = state.lock.withLock {
        (state.size, state.orientation, state.generation, state.enabled)
      }
      guard viewport.3, viewport.0.width > 0, viewport.0.height > 0 else { return nil }
      let inverse = frame.displayTransform(for: viewport.1, viewportSize: viewport.0).inverted()
      return CameraInferenceFrame(
        frame: frame,
        mapping: [inverse.a, inverse.b, inverse.c, inverse.d, inverse.tx, inverse.ty].map(
          Double.init),
        generation: viewport.2)
    }
  }
#endif
