import Testing

@testable import LiveSplat

@Test func insetControlsStayBoundedAndOrbitContinuously() {
  var orbit = ScanOrbit()
  #expect(orbit.rect == SIMD4<Float>(0.60, 0.17, 0.36, 0.25))
  #expect(orbit.angle(at: 60) > orbit.angle(at: 0))
  #expect(orbit.angle(at: 120) == orbit.angle(at: 0))
  #expect(orbit.angle(at: .nan) == orbit.yaw)
  orbit.rotate(horizontal: 100, vertical: 100)
  #expect(abs(orbit.yaw) < 2 * .pi && orbit.pitch == 1.2)
  orbit.rotate(horizontal: 0, vertical: -100)
  #expect(orbit.pitch == -0.8)
  let before = orbit
  orbit.rotate(horizontal: .nan, vertical: 0)
  #expect(orbit == before)
  orbit.enlarged = true
  let rect = orbit.rect
  #expect(rect.x >= 0 && rect.y >= 0 && rect.x + rect.z <= 1 && rect.y + rect.w <= 1)
  #expect(rect.z > before.rect.z && rect.w > before.rect.w)
}
