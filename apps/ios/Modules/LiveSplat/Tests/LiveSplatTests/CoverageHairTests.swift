import Foundation
import Testing

@testable import LiveSplat

@Suite struct CoverageHairTests {
  @Test func packedCellABI() {
    #expect(MemoryLayout<CoverageGPUCell>.size == 32)
    #expect(MemoryLayout<CoverageGPUCell>.stride == 32)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.normal) == 0)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.viewCount) == 18)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.epoch) == 20)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.flags) == 22)
    #expect(MemoryLayout<CoverageGPUCell>.offset(of: \.bestMmPerPx) == 24)
    var cell = CoverageGPUCell()
    for h in 0..<14 { cell[h] = UInt8(h + 1) }
    for h in 0..<14 { #expect(cell[h] == h + 1) }
    #expect(cell.viewCount == 0 && cell.flags == 0 && cell.bestMmPerPx == 0)
  }

  @Test func coneSharpnessSaturationAndResolution() {
    var p = CoverageCreditParameters()
    let edge = cos(p.coneDeg * .pi / 180)
    #expect(CoverageCreditReference.credit(cosine: edge, mmPerPx: 1, parameters: p) == 0)
    #expect(CoverageCreditReference.credit(cosine: -1, mmPerPx: 1, parameters: p) == 0)
    #expect(CoverageCreditReference.credit(cosine: 1, mmPerPx: 1, parameters: p) == 1)
    #expect(CoverageCreditReference.credit(cosine: 1, mmPerPx: 100, parameters: p) == 1)
    let mid = (1 + edge) / 2
    let linear = CoverageCreditReference.credit(cosine: mid, mmPerPx: 1, parameters: p)
    p.sharpness = 2
    let squared = CoverageCreditReference.credit(cosine: mid, mmPerPx: 1, parameters: p)
    #expect(abs(squared - linear * linear) < Float.ulpOfOne * 4)
    p.targetMmPerPx = 2
    #expect(CoverageCreditReference.credit(cosine: 1, mmPerPx: 4, parameters: p) == 0.5)
    #expect(CoverageCreditReference.credit(cosine: 1, mmPerPx: 1, parameters: p) == 1)
    for mm: Float in [0, -1, .infinity, .nan] {
      #expect(CoverageCreditReference.credit(cosine: 1, mmPerPx: mm, parameters: p) == 0)
    }
    #expect(CoverageCreditReference.credit(cosine: .nan, mmPerPx: 1, parameters: p) == 0)
    var credit: UInt8 = 0
    for _ in 0..<8 {
      credit = CoverageCreditReference.creditedByte(credit, credit: 1, viewsToFill: 8)
    }
    #expect(credit == 255)
    #expect(CoverageCreditReference.creditedByte(credit, credit: 1, viewsToFill: 8) == 255)
    #expect(CoverageCreditReference.creditedByte(0, credit: 0, viewsToFill: 8) == 0)
    #expect(
      CoverageCreditReference.remaining(128, parallax: false, parameters: p) == 1 - Float(128) / 255
    )
    #expect(CoverageCreditReference.remaining(255, parallax: false, parameters: p) == 0)
  }

  @Test func octahedralNormalsAcrossBothHemispheres() {
    // Quantization error bound from two 16-bit coordinates, not measured hardware arithmetic.
    let bound: Float = 8 / 32767
    for x in -4...4 {
      for y in -4...4 {
        for z in -4...4 where x != 0 || y != 0 || z != 0 {
          let n = CoverageCreditReference.unit(SIMD3(Float(x), Float(y), Float(z)))
          let decoded = CoverageCreditReference.decodeNormal(
            CoverageCreditReference.encodeNormal(n))
          let error = decoded - n
          #expect(sqrt(CoverageCreditReference.dot(error, error)) <= bound)
        }
      }
    }
    #expect(CoverageCreditReference.decodeNormal(.zero) == SIMD3(0, 0, 1))
    #expect(CoverageCreditReference.encodeNormal(SIMD3(0, 0, 1)) == .zero)
  }

  @Test func bundledAndPartialJSONWithInvalidValuesRejected() throws {
    let bundled = try CoverageCreditParameters.bundled()
    #expect(bundled == CoverageCreditParameters())
    #expect(bundled.hairs == 14 && bundled.viewsToFill == 8 && bundled.coneDeg == 25)
    #expect(bundled.targetMmPerPx == nil && bundled.parallaxMinDeg == nil)
    let partial = try CoverageCreditParameters.load(
      Data(#"{"sharpness":2,"targetMmPerPx":3,"parallaxMinDeg":5}"#.utf8))
    #expect(partial.sharpness == 2 && partial.targetMmPerPx == 3 && partial.parallaxMinDeg == 5)
    #expect(partial.hairs == 14 && partial.viewsToFill == 8)
    #expect(try CoverageCreditParameters.load(JSONEncoder().encode(partial)) == partial)
    for json in [
      #"{"hairs":15}"#, #"{"hairs":0}"#, #"{"coneDeg":0}"#,
      #"{"coneDeg":90}"#, #"{"viewsToFill":0}"#, #"{"sharpness":-1}"#,
      #"{"targetMmPerPx":0}"#, #"{"parallaxMinDeg":180}"#, #"{"normalEMA":2}"#,
      #"{"baseLength":0}"#, "invalid",
    ] {
      #expect(throws: (any Error).self) { try CoverageCreditParameters.load(Data(json.utf8)) }
    }
  }

  @Test func hashDedupeNormalEMAAndEpoch() throws {
    let position = SIMD3<Float>(0.025, 0.025, 0.025)
    var grid = CoverageHash(capacity: 16)
    let observed = grid.observe(
      position: position, normal: SIMD3(0, 0, 1), camera: SIMD3(0, 0, 2), focalPixels: 1000,
      epoch: 1)
    let slot = try #require(observed)
    #expect(grid.count == 1)
    let before = grid.cells[slot]
    grid.observe(
      position: position, normal: SIMD3(1, 0, 1), camera: SIMD3(1, 0, 2), focalPixels: 1000,
      epoch: 1)
    #expect(grid.count == 1 && grid.cells[slot].viewCount == 1)
    #expect(grid.cells[slot].normal == before.normal)
    #expect(grid.cells[slot][13] == before[13])
    grid.observe(
      position: position, normal: SIMD3(1, 0, 1), camera: SIMD3(1, 0, 2), focalPixels: 1000,
      epoch: 2)
    let mean = CoverageCreditReference.decodeNormal(grid.cells[slot].normal)
    #expect(mean.x > 0 && mean.x < 0.3 && mean.z > 0.9)
    #expect(grid.cells[slot].viewCount == 2 && grid.cells[slot].epoch == 2)
    #expect(grid.cells[slot].bestMmPerPx > 0)
    #expect(CoverageKey(position: SIMD3(-0.01, -0.11, 0.11)).xyz == SIMD3(-1, -2, 1))
    #expect(
      (grid.observe(
        position: SIMD3(.nan, 0, 0), normal: SIMD3(0, 0, 1), camera: .zero, focalPixels: 1000,
        epoch: 3) == nil))
    #expect(
      (grid.observe(position: position, normal: .zero, camera: .zero, focalPixels: 1000, epoch: 3)
        == nil))
    #expect(grid.count == 1)
  }

  @Test func syntheticHemisphereOrbitFillsPlaneAndRejectsOutsideCone() throws {
    let p = CoverageCreditParameters()
    let point = SIMD3<Float>(0.025, 0.025, 0)
    let n = SIMD3<Float>(0, 0, 1)
    var grid = CoverageHash(capacity: 4, parameters: p)
    var epoch: UInt16 = 0
    // Equal-area orbit visits each of the 14 hemisphere directions repeatedly.
    // At each exact hair direction the expected increment is 255/8 rounded to 32.
    for repeatIndex in 0..<8 {
      for h in 0..<p.hairs {
        epoch += 1
        let direction = CoverageCreditReference.hair(h, count: p.hairs, normal: n)
        let observed = grid.observe(
          position: point, normal: n, camera: point + direction * 2, focalPixels: 1000, epoch: epoch
        )

        let slot = try #require(observed)
        #expect(grid.cells[slot][h] >= min(255, (repeatIndex + 1) * 32))
      }
    }
    let slot = try #require(grid.keys.firstIndex { $0 == CoverageKey(position: point) })
    for h in 0..<p.hairs {
      #expect(CoverageCreditReference.filled(grid.cells[slot][h], parallax: false, parameters: p))
    }
    #expect(grid.count == 1)
    // A camera orbit on the opposite azimuth cannot fill hair zero. Back views
    // use the previously established plane orientation, so cannot flip it.
    var outside = CoverageHash(capacity: 4)
    let h0 = CoverageCreditReference.hair(0, count: 14, normal: n)
    let other = SIMD3(-h0.x, -h0.y, h0.z)
    for e: UInt16 in 1...100 {
      outside.observe(
        position: point, normal: n, camera: point + other * 2, focalPixels: 1000, epoch: e)
    }
    let s = try #require(outside.keys.firstIndex { $0 != nil })
    #expect(outside.cells[s][0] == 0)
    outside.observe(
      position: point, normal: n, camera: point - n * 2, focalPixels: 1000, epoch: 101)
    #expect(outside.cells[s][0] == 0)
  }

  @Test func referenceKernelHasExpectedResolutionCreditAndParallaxGate() throws {
    var p = CoverageCreditParameters()
    p.targetMmPerPx = 1
    p.parallaxMinDeg = 5
    let n = SIMD3<Float>(0, 0, 1)
    let point = SIMD3<Float>(0.025, 0.025, 0)
    let h = CoverageCreditReference.hair(13, count: p.hairs, normal: n)
    var grid = CoverageHash(capacity: 4, parameters: p)
    // Choose focal so distance/incidence gives exactly 2 mm/px algebraically.
    let focal = 1000 / h.z
    let observed = grid.observe(
      position: point, normal: n, camera: point + h * 2, focalPixels: focal, epoch: 1)
    let slot = try #require(observed)
    #expect(grid.cells[slot][13] == 16)  // half-resolution weight, 255/8/2 -> 16
    #expect(abs(Float(grid.cells[slot].bestMmPerPx) - 2) <= Float(Float16(2).ulp))
    for e: UInt16 in 2...20 {
      grid.observe(position: point, normal: n, camera: point + h * 2, focalPixels: focal, epoch: e)
    }
    #expect(grid.cells[slot][13] == 255)
    #expect(
      !CoverageCreditReference.filled(
        255, parallax: grid.cells[slot].flags & (1 << 13) != 0, parameters: p))
    #expect(CoverageCreditReference.remaining(255, parallax: false, parameters: p) == 0.15)
    let tangent = CoverageCreditReference.unit(CoverageCreditReference.cross(h, n))
    func view(_ degrees: Float) -> SIMD3<Float> {
      h * cos(degrees * .pi / 180) + tangent * sin(degrees * .pi / 180)
    }
    grid.observe(
      position: point, normal: n, camera: point + view(2) * 2, focalPixels: focal, epoch: 21)
    #expect(grid.cells[slot].flags & (1 << 13) == 0)
    grid.observe(
      position: point, normal: n, camera: point + view(8) * 2, focalPixels: focal, epoch: 22)
    #expect(grid.cells[slot].flags & (1 << 13) != 0)
    #expect(CoverageCreditReference.filled(255, parallax: true, parameters: p))
    #expect(CoverageCreditReference.remaining(255, parallax: true, parameters: p) == 0)
  }

  @Test func capacityCollisionsEvictionAndFlatStorage() throws {
    // A single four-way bucket forces collision, replacement and stale-history reset.
    var p = CoverageCreditParameters()
    p.parallaxMinDeg = 5
    var small = CoverageHash(capacity: 4, parameters: p)
    for i in 0..<4 {
      small.observe(
        position: SIMD3(Float(i), 0, 0), normal: SIMD3(0, 0, 1), camera: SIMD3(Float(i), 0, 2),
        focalPixels: 1000, epoch: UInt16(i))
    }
    #expect(small.count == 4)
    let observed = small.observe(
      position: SIMD3(10, 0, 0), normal: SIMD3(0, 0, 1), camera: SIMD3(10, 0, 2), focalPixels: 1000,
      epoch: 5)
    let replaced = try #require(observed)
    #expect(small.cells[replaced].viewCount == 1 && small.cells[replaced].flags == 0)
    #expect(small.keys[replaced] == CoverageKey(position: SIMD3(10, 0, 0)))
    var full = CoverageHash()
    let allocated = full.allocatedSlots
    // Four times capacity exercises replacement; fixed buffers are the memory invariant.
    for i in 0..<CoverageHash.capacity * 4 {
      let point = SIMD3<Float>(Float(i) * 0.1 + 0.025, 0.025, 0)
      full.observe(
        position: point, normal: SIMD3(0, 0, 1), camera: point + SIMD3(0, 0, 2), focalPixels: 1000,
        epoch: UInt16(i % 3000))
      if i == CoverageHash.capacity - 1 {
        #expect(full.count == CoverageHash.capacity)
        #expect(full.allocatedSlots == allocated)
      }
    }
    #expect(full.count == CoverageHash.capacity)
    #expect(full.cells.count == CoverageHash.capacity && full.keys.count == CoverageHash.capacity)
    #expect(full.allocatedSlots == allocated)
  }
}
