import Testing
@testable import AppSupport

@Test func trainingCameraPolicySuspendsVisionAndKeepsOnlyTheNeededWorld() {
  #expect(!TrainingCameraPolicy.perceptionEnabled(trainingBusy: true, captureBusy: false, talking: false, answering: false))
  #expect(TrainingCameraPolicy.perceptionEnabled(trainingBusy: false, captureBusy: false, talking: false, answering: false))
  #expect(!TrainingCameraPolicy.perceptionEnabled(trainingBusy: false, captureBusy: true, talking: false, answering: false))
  #expect(!TrainingCameraPolicy.perceptionEnabled(trainingBusy: false, captureBusy: false, talking: true, answering: false))
  #expect(!TrainingCameraPolicy.perceptionEnabled(trainingBusy: false, captureBusy: false, talking: false, answering: true))
  #expect(TrainingCameraPolicy.keepSession(trainingBusy: true, needsInPlace: true))
  #expect(!TrainingCameraPolicy.keepSession(trainingBusy: true, needsInPlace: false))
  #expect(TrainingCameraPolicy.keepSession(trainingBusy: false, needsInPlace: false))
}
