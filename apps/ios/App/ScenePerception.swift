import EntitySchema
import Perception
import Router

/// App adapter keeps the scene package independent of routing and supplies device defaults.
struct ScenePerception: SceneQuerying {
  private let scene: PerceptionScene
  init(
    segmenter: any Segmenter = RuntimeSegmenter(),
    instances: any InstanceSegmenting = VisionSegmenter(),
    identifier: any ImageIdentifying = VisionIdentifier()
  ) {
    scene = PerceptionScene(segmenter: segmenter, instances: instances, identifier: identifier)
  }
  func update(_ frame: FrameSnapshot, viewportToImage: [Double], capturedAt: Double) async {
    await scene.update(frame, viewportToImage: viewportToImage, capturedAt: capturedAt)
  }
  func clear() async { await scene.clear() }
  func selection() async -> SceneSelection { await scene.selection() }
  func select(_ point: ImagePoint) async throws -> SceneSelection { try await scene.select(point) }
  func refreshSelection() async throws -> SceneSelection { try await scene.refreshSelection() }
  func inspect(_ plan: AnswerPlan) async throws -> [EntitySchema.Entity] {
    try await scene.inspect(subject: plan.subject, count: plan.tools.contains(.count))
  }
}
