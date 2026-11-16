/// App-local seams until Capture and Voice provide their device integrations.
@MainActor
protocol RecordingControl {
    func toggleRecording()
}

@MainActor
protocol PushToTalkControl {
    func beginTalking()
    func endTalking()
}

/// Deliberately no-op: M0 does not claim to record media or recognize speech.
struct StubRecordingControl: RecordingControl {
    func toggleRecording() {}
}

struct StubPushToTalkControl: PushToTalkControl {
    func beginTalking() {}
    func endTalking() {}
}
