import EntitySchema
import Foundation

/// Supplies current on-device perception. A new query selects by tap or requested subject;
/// count queries return distinct foreground instances, never duplicated tracked anchors.
public protocol SceneQuerying: Sendable {
  func inspect(_ plan: AnswerPlan) async throws -> [Entity]
}

public struct DeviceFastAnswer: FastAnswering {
  private let scene: any SceneQuerying
  public init(scene: any SceneQuerying) { self.scene = scene }
  public func answer(_ request: AskRequest, plan: AnswerPlan) async throws -> AsyncThrowingStream<
    AskEvent, Error
  > {
    let entities = try await scene.inspect(plan)
    try Task.checkCancellation()
    return AsyncThrowingStream { c in
      for entity in entities { c.yield(.entity(entity)) }
      var phrases: [String] = []
      if plan.tools.contains(.count) { phrases.append("I found \(entities.count).") }
      if let entity = entities.first {
        if plan.tools.contains(.measure) {
          let question = request.question.lowercased()
          let dimension: Double
          if question.contains("wide") || question.contains("width") {
            dimension = entity.extent.x
          } else if question.contains("long") || question.contains("length") {
            dimension = entity.extent.z
          } else {
            dimension = entity.extent.y
          }
          phrases.append(
            String(
              format: "About %.2f metres.", locale: Locale(identifier: "en_US_POSIX"), dimension))
        }
        if plan.tools.contains(.identifyOnDevice) {
          phrases.append(
            entity.attrs["identifiedOnDevice"] == .bool(true)
              ? entity.label : "I can't identify this yet.")
        }
      } else if !plan.tools.contains(.count) {
        phrases.append("Point at the subject.")
      }
      if !phrases.isEmpty { c.yield(.speech(phrases.joined(separator: " "))) }
      c.yield(.done)
      c.finish()
    }
  }
}
