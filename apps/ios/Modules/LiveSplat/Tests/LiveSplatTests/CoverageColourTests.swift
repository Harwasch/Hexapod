import Foundation
import Testing

@testable import LiveSplat

@Suite struct CoverageColourTests {
  @Test func scoreIsMonotonicAndRequiresConfidenceAndSpread() {
    var previous: Float = 0
    for i in 0...32 {
      let score = CoverageAppearance.score(
        observations: Float(i), confidentRun: Float(i), azimuth: 3, elevation: 1)
      #expect(score >= previous && score >= 0 && score <= 1)
      previous = score
    }
    #expect(
      CoverageAppearance.score(observations: 8, confidentRun: 6, azimuth: 3, elevation: 1) == 1)
    #expect(
      CoverageAppearance.score(observations: 32, confidentRun: 0, azimuth: 24, elevation: 4) == 0)
    #expect(
      CoverageAppearance.score(observations: 32, confidentRun: 32, azimuth: 1, elevation: 1) < 1)
    #expect(
      CoverageAppearance.score(observations: 8, confidentRun: 6, azimuth: 1, elevation: 2) == 1)
    for az in 0..<24 {
      #expect(
        CoverageAppearance.score(observations: 8, confidentRun: 6, azimuth: az + 1, elevation: 0)
          >= CoverageAppearance.score(observations: 8, confidentRun: 6, azimuth: az, elevation: 0))
    }
  }
  @Test func appearanceDevelopsSmoothlyToOpaqueTrueColourAndPulseExpires() {
    let colour = SIMD3<Float>(0.8, 0.25, 0.1)
    var previous: Float = 0
    for i in 0...100 {
      let time = Float(i) * CoverageAppearance.transitionSeconds / 100
      let progress = CoverageAppearance.developed(previous: 0, target: 1, elapsed: time)
      #expect(progress >= previous && progress - previous < 0.02)
      let appearance = CoverageAppearance.appearance(colour: colour, progress: progress)
      #expect(appearance.opacity >= CoverageAppearance.minimumOpacity && appearance.opacity <= 1)
      previous = progress
    }
    #expect(CoverageAppearance.developed(previous: 0, target: 1, elapsed: 0) == 0)
    let final = CoverageAppearance.appearance(colour: colour, progress: 1)
    #expect(final.opacity == 1)
    #expect(final.colour == colour)
    let start = CoverageAppearance.appearance(colour: colour, progress: 0)
    #expect(start.colour.z > colour.z && start.colour.x < colour.x)
    let pulse = CoverageAppearance.appearance(
      colour: colour, progress: 1, pulseAge: CoverageAppearance.pulseSeconds / 2)
    #expect(pulse.colour.x > final.colour.x)
    #expect(
      CoverageAppearance.appearance(
        colour: colour, progress: 1, pulseAge: CoverageAppearance.pulseSeconds) == final)
    let epsilon: Float = 0.0001
    #expect(CoverageAppearance.ease(epsilon) / epsilon < 0.001)
    #expect((1 - CoverageAppearance.ease(1 - epsilon)) / epsilon < 0.001)
    // Retarget from the displayed value, so new observations never pop.
    let halfway = CoverageAppearance.developed(previous: 0, target: 0.6, elapsed: 0.3)
    #expect(CoverageAppearance.developed(previous: halfway, target: 1, elapsed: 0) == halfway)
  }
  @Test func gpuLayoutsAndShaderMirrorStayInSync() throws {
    #expect(MemoryLayout<CoverageGPUCell>.stride == 32)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.key) == 0)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.views) == 16)
    #expect(MemoryLayout<CoverageDevelopment>.stride == 16)
    #expect(MemoryLayout<CoverageDevelopment>.offset(of: \.animation) == 0)
    #expect(MemoryLayout<CoverageSummary>.stride == 32)
    #expect(MemoryLayout<CoverageSummary>.offset(of: \.centerCount) == 0)
    #expect(MemoryLayout<CoverageSummary>.offset(of: \.counts) == 16)
    #expect(MemoryLayout<CoverageStyleUniforms>.stride == 64)
    #expect(MemoryLayout<CoverageStyleUniforms>.offset(of: \.thresholds) == 0)
    #expect(MemoryLayout<CoverageStyleUniforms>.offset(of: \.timing) == 16)
    #expect(MemoryLayout<CoverageStyleUniforms>.offset(of: \.effects) == 32)
    #expect(MemoryLayout<CoverageStyleUniforms>.offset(of: \.tint) == 48)
    let u = CoverageStyleUniforms()
    #expect(
      u.thresholds
        == SIMD4(
          CoverageAppearance.observations, CoverageAppearance.confidentRun,
          CoverageAppearance.azimuthBins, CoverageAppearance.elevationBins))
    #expect(u.timing.y == CoverageAppearance.transitionSeconds)
    #expect(u.effects.x == CoverageAppearance.minimumOpacity)
    let shader = try String(
      contentsOf: sourceRoot.appendingPathComponent("Shaders/LiveSplat.metal"), encoding: .utf8)
    #expect(shader.contains("struct CoverageCell { int4 key; uint4 views; };"))
    #expect(shader.contains("struct CoverageStyle { float4 thresholds, timing, effects, tint; };"))
    #expect(shader.contains("struct CoverageSummary { float4 centerCount; uint4 counts; };"))
    #expect(shader.contains("constant uint coverageCapacity = \(CoverageHash.capacity);"))
    #expect(shader.contains("constant uint coverageSectors = \(CoverageHash.sectorCount);"))
    #expect(shader.contains("constant float coverageCellSize = 0.1f;"))
    #expect(shader.contains("return t * t * (3 - 2 * t);"))
    #expect(shader.components(separatedBy: "coverageAppearance(cell.color.xyz").count == 3)
  }
  @Test func retiredRenderingAndResourcesAreAbsent() throws {
    let fm = FileManager.default
    #expect(
      !fm.fileExists(
        atPath: sourceRoot.appendingPathComponent("Resources/coverage_credit.json").path))
    let removed = [
      "CoverageHair", "coverageHair", "prepareCoverageHairs", "creditCoverageCells",
      "ScanInset", "ScanOrbit", "OrbitUniforms", "compositeScanInset", "phoneFrustum",
      "coverage_credit.json", "insetEnabled",
    ]
    let files = try #require(fm.enumerator(at: sourceRoot, includingPropertiesForKeys: nil))
    for case let url as URL in files where ["swift", "metal"].contains(url.pathExtension) {
      let source = try String(contentsOf: url, encoding: .utf8)
      for name in removed {
        #expect(!source.contains(name), "\(url.lastPathComponent) still references \(name)")
      }
    }
    let app = sourceRoot.deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("App")
    for name in ["CameraScreen.swift", "CameraActions.swift"] {
      let source = try String(contentsOf: app.appendingPathComponent(name), encoding: .utf8)
      for symbol in removed { #expect(!source.contains(symbol)) }
    }
  }
  @Test func coarseGridDedupesReplacesAndKeepsFlatStorage() {
    var grid = CoverageHash(capacity: 16)
    grid.observe(position: .zero, camera: SIMD3(1, 0, 0), epoch: 1)
    grid.observe(position: .zero, camera: SIMD3(1, 0, 0), epoch: 1)
    #expect(grid.cells.compactMap { $0 }.first?.observations == 1)
    for i in 2...1000 {
      grid.observe(
        position: SIMD3(Float(i) * 0.125, -0.01, 0), camera: SIMD3(0, 2, 0), epoch: UInt32(i))
    }
    #expect(grid.cells.count == 16 && grid.cells.compactMap { $0 }.count == 16)
    #expect(grid.cells.compactMap { $0 }.contains { $0.key.y == -1 })
    #expect(CoverageHash.direction(SIMD3(1, 0, 0)).elevation == 2)
    #expect(CoverageHash.direction(SIMD3(0, 1, 0)).elevation == 8)
  }
  private var sourceRoot: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent("Sources/LiveSplat")
  }
}
