import Testing
import PhoneTrain

@Test func productSnapshotsOnlyRunAtSafeBoundaries() {
  #expect(TrainingSnapshotPolicy.preview(step: 40, requested: true, active: true,
    paused: false, cancelled: false, thermal: 2))
  #expect(!TrainingSnapshotPolicy.preview(step: 39, requested: true, active: true,
    paused: false, cancelled: false, thermal: 0))
  #expect(!TrainingSnapshotPolicy.preview(step: 40, requested: true, active: false,
    paused: false, cancelled: false, thermal: 0))
  #expect(!TrainingSnapshotPolicy.preview(step: 40, requested: true, active: true,
    paused: true, cancelled: false, thermal: 0))
  #expect(!TrainingSnapshotPolicy.evaluate(step: 250, requested: true, active: true,
    paused: false, cancelled: true, thermal: 0))
  #expect(!TrainingSnapshotPolicy.evaluate(step: 250, requested: true, active: true,
    paused: false, cancelled: false, thermal: 3))
  #expect(!TrainingSnapshotPolicy.evaluate(step: 250, requested: false, active: true,
    paused: false, cancelled: false, thermal: 0))
  #expect(TrainingSnapshotPolicy.evaluate(step: 250, requested: true, active: true,
    paused: false, cancelled: false, thermal: 0))
}
@Test func productETAUsesCompletedActiveSteps() {
  var eta = TrainingETA()
  #expect(eta.record(stepSeconds: 2, remaining: 9) == 18)
  #expect(eta.record(stepSeconds: 4, remaining: 8) == 24)
  #expect(eta.record(stepSeconds: .nan, remaining: 8) == 0)
  #expect(eta.record(stepSeconds: 3, remaining: -1) == 0)
}
