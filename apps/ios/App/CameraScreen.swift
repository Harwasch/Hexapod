import ARAnswer
import LiveSplat
import SwiftUI

@MainActor struct CameraScreen: View {
  @Environment(\.scenePhase) private var scenePhase
  @State private var model: CameraModel
  @State private var isPressing = false

  init(model: CameraModel) {
    _model = State(initialValue: model)
  }

  var body: some View {
    ZStack(alignment: .bottom) {
      ARCameraView(isActive: scenePhase == .active, model: model).ignoresSafeArea()
      if let renderer = model.liveSplat as? DefaultLiveSplatRenderer {
        LiveSplatView(renderer: renderer).allowsHitTesting(false).ignoresSafeArea()
      }
      VStack(spacing: 16) {
        if model.simulated { Text("Simulated test providers").font(.caption) }
        if let entity = model.card {
          EntityCard(entity: entity) { model.card = nil }
            .padding(.horizontal)
        }
        HStack(spacing: 48) {
          Button(action: model.toggleRecording) {
            Image(systemName: model.recording ? "stop.circle.fill" : "record.circle")
              .foregroundStyle(.red).font(.system(size: 64))
          }
          .accessibilityLabel(model.recording ? "Stop recording" : "Record")
          .disabled(!model.usesLiveCapture)
          Image(systemName: model.isTalking ? "mic.fill" : "mic")
            .font(.system(size: 30)).frame(width: 64, height: 64)
            .background(model.isTalking ? Color.red : Color.black.opacity(0.5), in: Circle())
            .contentShape(Circle())
            .gesture(
              DragGesture(minimumDistance: 0)
                .onChanged { _ in setPressing(true) }
                .onEnded { _ in setPressing(false) }
            )
            .accessibilityLabel("Push to talk")
            .accessibilityAddTraits(.isButton)
            .accessibilityValue(model.isTalking ? "Pressed" : "Released")
            .accessibilityAction {
              setPressing(!isPressing)
            }
        }
        .buttonStyle(.plain)
        .disabled(
          (model.usesLiveCapture && !ARCameraView.supportsDevice) || scenePhase != .active
            || !model.cameraReady)
      }
      .padding(.bottom, 24)
      if model.failed {
        VStack {
          Image(systemName: "exclamationmark.circle").accessibilityLabel("Unavailable")
          Spacer()
        }
        .padding(.top, 16)
        .allowsHitTesting(false)
      }
    }
    .onChange(of: scenePhase) { _, phase in
      if phase != .active {
        setPressing(false)
        model.suspend()
      }
    }
    .onDisappear {
      setPressing(false)
      model.suspend()
    }
  }

  private func setPressing(_ value: Bool) {
    guard value != isPressing else { return }
    if value && (!model.cameraReady || scenePhase != .active) { return }
    isPressing = value
    if value { model.beginTalking() } else { model.endTalking() }
  }
}
