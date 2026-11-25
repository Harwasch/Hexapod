import Perception

public struct OverlayPipelineCheck: Equatable, Sendable {
  public let name: String
  public let error: String?
  public var description: String { "\(name): \(error.map { "failed: \($0)" } ?? "built")" }

  /// Attempt both pipelines independently, even if the first throws.
  public static func build<Highlight, Surface>(
    highlight: () throws -> Highlight, surface: () throws -> Surface
  ) -> (highlight: Highlight?, surface: Surface?, checks: [OverlayPipelineCheck]) {
    var checks: [OverlayPipelineCheck] = []
    func attempt<T>(_ name: String, _ factory: () throws -> T) -> T? {
      do {
        let result = try factory()
        let check = OverlayPipelineCheck(name: name, error: nil)
        checks.append(check)
        CameraDiagnostics.record(check.description)
        return result
      } catch {
        let check = OverlayPipelineCheck(name: name, error: String(describing: error))
        checks.append(check)
        CameraDiagnostics.record(check.description)
        return nil
      }
    }
    let h = attempt("HighlightProcessor", highlight)
    let s = attempt("WorldSurfaceRenderer", surface)
    return (h, s, checks)
  }
}
