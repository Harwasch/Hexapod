/// Training suppresses inference even when the scan phase is already saved.
/// Keep an existing AR world only when the result can be placed back into it.
enum TrainingCameraPolicy {
  static func perceptionEnabled(trainingBusy: Bool, captureBusy: Bool,
    talking: Bool, answering: Bool) -> Bool {
    !trainingBusy && !captureBusy && !talking && !answering
  }
  static func keepSession(trainingBusy: Bool, needsInPlace: Bool) -> Bool {
    !trainingBusy || needsInPlace
  }
}
