import Foundation
import Testing
@testable import LiveSplat

@Test func savedSortOnlyChangesAfterThreeDegreesAndIgnoresTranslation() {
  let initial = SIMD3<Float>(0, 0, 1)
  func rows(_ degrees: Float, translation: Float = 0) -> [Float] {
    let angle = degrees * Float.pi / 180
    return [1,0,0,0, 0,1,0,0, sin(angle),0,cos(angle),translation, 0,0,0,1]
  }
  #expect(SplatSortPolicy.shouldSort(previous: nil, viewRows: rows(0)))
  #expect(!SplatSortPolicy.shouldSort(previous: initial, viewRows: rows(2)))
  #expect(SplatSortPolicy.shouldSort(previous: initial, viewRows: rows(4)))
  #expect(!SplatSortPolicy.shouldSort(previous: initial, viewRows: rows(0, translation: 10)))
  #expect(!SplatSortPolicy.shouldSort(previous: initial, viewRows: []))
}

@Test func compactSortDataMatchesFullGeometryWithoutRetainingGaussianAttributes() throws {
  let point = try Gaussian(position: SplatVector(0, 0, -1), scale: SplatVector(0.02, 0.03, 0.01),
    rotation: [1, 0, 0, 0], opacity: 0.8, color: SplatVector(1, 0, 0))
  let scene = try SavedSplat(data: SplatPLY.encode([point]))
  let compact = SavedSplatSortData(scene: scene)
  let rows: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
  #expect(compact.sortedIndices(viewRows: rows) == scene.sortedIndices(viewRows: rows))
  #expect(compact.center == scene.center && compact.radius == scene.radius)
  #expect(MemoryLayout<SIMD3<Float>>.stride < MemoryLayout<SavedGaussian>.stride)
}
