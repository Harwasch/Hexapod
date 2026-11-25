import Testing

@testable import ARAnswer

@Suite struct OverlayPipelineCheckTests {
  enum Failure: Error { case metal }
  @Test func bothPipelinesAttemptedAfterHighlightFailure() {
    let result = OverlayPipelineCheck.build(
      highlight: { () throws -> Int in throw Failure.metal }, surface: { 42 })
    #expect(result.highlight == nil)
    #expect(result.surface == 42)
    #expect(result.checks.count == 2)
    #expect(result.checks[0].description == "HighlightProcessor: failed: metal")
    #expect(result.checks[1].description == "WorldSurfaceRenderer: built")
  }
  @Test func surfaceFailurePreservesFlatHighlight() {
    let result = OverlayPipelineCheck.build(
      highlight: { 42 }, surface: { () throws -> Int in throw Failure.metal })
    #expect(result.highlight == 42)
    #expect(result.surface == nil)
    #expect(result.checks[0].error == nil)
    #expect(result.checks[1].description == "WorldSurfaceRenderer: failed: metal")
  }
  @Test func bothFailuresStillProduceTwoNamedReports() {
    let result = OverlayPipelineCheck.build(
      highlight: { () throws -> Int in throw Failure.metal },
      surface: { () throws -> Int in throw Failure.metal })
    #expect(result.highlight == nil && result.surface == nil)
    #expect(result.checks.map(\.name) == ["HighlightProcessor", "WorldSurfaceRenderer"])
    #expect(result.checks.allSatisfy { $0.error == "metal" })
  }

}
