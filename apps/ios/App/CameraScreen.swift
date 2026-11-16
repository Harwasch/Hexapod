import SwiftUI

@MainActor
struct CameraScreen: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var isTalking = false
    var recording: any RecordingControl = StubRecordingControl()
    var voice: any PushToTalkControl = StubPushToTalkControl()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARCameraView(isActive: scenePhase == .active)
                .ignoresSafeArea()
            HStack(spacing: 48) {
                Button(action: recording.toggleRecording) {
                    Image(systemName: "record.circle")
                        .foregroundStyle(.red)
                        .font(.system(size: 64))
                }
                .accessibilityLabel("Record")

                Image(systemName: isTalking ? "mic.fill" : "mic")
                    .font(.system(size: 30))
                    .frame(width: 64, height: 64)
                    .background(isTalking ? Color.red : Color.black.opacity(0.5), in: Circle())
                    .contentShape(Circle())
                    .gesture(DragGesture(minimumDistance: 0)
                        .onChanged { _ in setTalking(true) }
                        .onEnded { _ in setTalking(false) })
                    .accessibilityLabel("Push to talk")
                    .accessibilityAddTraits(.isButton)
                    .accessibilityValue(isTalking ? "Pressed" : "Released")
                    .accessibilityAction { setTalking(!isTalking) }
            }
            .buttonStyle(.plain)
            .padding(.bottom, 24)
            .disabled(!ARCameraView.supportsDevice || scenePhase != .active)
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { setTalking(false) }
        }
        .onDisappear { setTalking(false) }
    }

    private func setTalking(_ value: Bool) {
        guard value != isTalking else { return }
        if value && (!ARCameraView.supportsDevice || scenePhase != .active) { return }
        isTalking = value
        if value { voice.beginTalking() } else { voice.endTalking() }
    }
}
