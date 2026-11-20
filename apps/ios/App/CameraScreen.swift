import ARAnswer
import AVFoundation
import AVKit
import Globe
import LiveSplat
import SwiftUI
import Upload

private enum CameraStyle {
  static let background = Color(red: 13 / 255, green: 21 / 255, blue: 23 / 255)
  static let panel = Color(red: 20 / 255, green: 32 / 255, blue: 35 / 255)
  static let teal = Color(red: 63 / 255, green: 208 / 255, blue: 201 / 255)
  static let amber = Color(red: 1, green: 181 / 255, blue: 71 / 255)
  static let red = Color(red: 1, green: 77 / 255, blue: 77 / 255)
  static let text = Color(red: 228 / 255, green: 236 / 255, blue: 233 / 255)
  static let muted = Color(red: 138 / 255, green: 160 / 255, blue: 155 / 255)
}

/// Navigation belongs to HomeScreen; the camera remains the full-bleed home surface.
@MainActor struct CameraScreen: View {
  @Environment(\.scenePhase) private var scenePhase
  @State private var model: CameraModel
  @State private var isPressing = false
  @State private var showSites = false
  @State private var latestThumbnail: UIImage?
  private let onCaptures: (() -> Void)?
  private let onSettings: (() -> Void)?
  private let onGlobe: (() -> Void)?

  init(
    model: CameraModel, onCaptures: (() -> Void)? = nil,
    onSettings: (() -> Void)? = nil, onGlobe: (() -> Void)? = nil
  ) {
    _model = State(initialValue: model)
    self.onCaptures = onCaptures
    self.onSettings = onSettings
    self.onGlobe = onGlobe
  }

  var body: some View {
    ZStack {
      CameraStyle.background.ignoresSafeArea()
      ARCameraView(isActive: scenePhase == .active, model: model).ignoresSafeArea()
      if model.recording, model.splatEnabled,
        let renderer = model.liveSplat as? DefaultLiveSplatRenderer
      {
        LiveSplatView(renderer: renderer).allowsHitTesting(false).ignoresSafeArea()
      }
      LinearGradient(
        colors: [.clear, .black.opacity(0.65)], startPoint: .center, endPoint: .bottom
      )
      .ignoresSafeArea().allowsHitTesting(false)
      ForEach(model.screenPins) { pin in
        Button {
          model.openPin(pin.id)
        } label: {
          Group {
            if model.simulated {
              Text("\(pin.id + 1)").font(.headline).foregroundStyle(CameraStyle.teal)
            } else {
              Color.clear
            }
          }.frame(width: 44, height: 44).contentShape(Rectangle())
        }.accessibilityLabel("Subject \(pin.id + 1)")
          .accessibilityIdentifier("answer.pin.\(pin.id + 1)")
          .position(x: pin.x, y: pin.y)
      }
      TimelineView(.periodic(from: .now, by: 0.25)) { context in
        GeometryReader { geometry in
          VStack {
            header(at: context.date)
            Spacer()
            if model.simulated {
              Text("Simulated").font(.caption).padding(8).background(.black, in: Capsule())
            }
            if !model.cameraReady {
              Label(
                ARCameraView.supportsDevice ? "Camera unavailable" : "LiDAR required",
                systemImage: "camera.fill"
              )
              .font(.callout).padding(12).background(CameraStyle.panel, in: Capsule())
            } else if model.failed {
              Label(model.failureText, systemImage: "exclamationmark.circle")
                .font(.callout).padding(12).background(CameraStyle.panel, in: Capsule())
            }
            if let entity = model.card {
              EntityCard(
                entity: entity, pin: model.pinCard,
                ask: {
                  model.card = nil
                  setPressing(true)
                }
              ) { model.card = nil }.padding(.horizontal, 20)
            } else if model.isTalking {
              listening
            }
            if model.card == nil { controls }
          }
          .padding(.horizontal, 20).padding(.top, 8).padding(.bottom, 18)
          if model.recording, let gap = model.gapPosition {
            Label("Fill gap", systemImage: "arrow.down")
              .font(.system(size: 12, weight: .medium)).foregroundStyle(CameraStyle.text)
              .padding(.horizontal, 10).padding(.vertical, 6)
              .background(.black.opacity(0.6), in: Capsule())
              .position(x: gap.x * geometry.size.width, y: gap.y * geometry.size.height)
              .allowsHitTesting(false)
          }
        }
      }
      if let directory = model.savedCapture {
        CaptureUploadScreen(
          capture: LocalCapture(directory: directory, upload: nil),
          siteName: model.siteName, siteID: model.siteID, duration: model.captureDuration,
          onSettings: onSettings, onSite: { showSites = true }, done: model.dismissCapture)
      }
    }
    .foregroundStyle(CameraStyle.text)
    .accessibilityIdentifier("camera.screen")
    .toolbar(.hidden, for: .navigationBar)
    .sheet(isPresented: $showSites) { CameraSitePicker(model: model) }
    .task(id: model.latestCapture) { await loadLatestThumbnail() }
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

  private func header(at date: Date) -> some View {
    HStack {
      icon("globe.americas", label: "Globe", action: onGlobe)
        .opacity(model.capturePhase.isBusy ? 0 : 1).disabled(model.capturePhase.isBusy)
      Spacer()
      Button {
        if !model.capturePhase.isBusy { showSites = true }
      } label: {
        HStack(spacing: 8) {
          if model.recording {
            Circle().fill(CameraStyle.red).frame(width: 9, height: 9)
            Text(CapturePhase.time(model.capturePhase.elapsed(at: date))).monospacedDigit()
          } else if case .stopping = model.capturePhase {
            ProgressView().tint(CameraStyle.teal)
            Text("Saving")
          } else if case .starting = model.capturePhase {
            ProgressView().tint(CameraStyle.teal)
            Text("Starting")
          } else {
            Image(systemName: "mappin.and.ellipse").foregroundStyle(CameraStyle.teal)
            if let name = model.siteName { Text(name).lineLimit(1) }
          }
        }
        .font(.system(size: 13, weight: .medium)).padding(.horizontal, 12).frame(height: 44)
        .background(.black.opacity(0.55), in: Capsule())
      }
      .disabled(model.capturePhase.isBusy)
      .accessibilityLabel(
        model.recording
          ? "Recording, \(CapturePhase.time(model.capturePhase.elapsed(at: date)))"
          : model.siteName ?? "Choose site")
      Spacer()
      if model.recording {
        ProgressRing(progress: model.coverage).frame(width: 44, height: 44)
          .accessibilityLabel("Current view coverage \(Int(model.coverage * 100)) percent")
      } else {
        icon("gearshape", label: "Settings", action: onSettings)
      }
    }
  }

  private var controls: some View {
    GeometryReader { geometry in
      ZStack {
        Button {
          onCaptures?()
        } label: {
          ZStack {
            if let latestThumbnail {
              Image(uiImage: latestThumbnail).resizable().scaledToFill()
            } else {
              CameraStyle.panel
            }
            Image(systemName: "square.grid.2x2").font(.system(size: 23))
          }
          .frame(width: 52, height: 52).clipShape(RoundedRectangle(cornerRadius: 14))
          .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.85), lineWidth: 2))
        }.accessibilityLabel("Captures").accessibilityIdentifier("home.captures").disabled(
          onCaptures == nil || model.capturePhase.isBusy
        )
        .opacity(model.capturePhase.isBusy || model.isTalking || !model.question.isEmpty ? 0 : 1)
        .position(x: 34, y: 42)
        Button(action: model.toggleRecording) {
          ZStack {
            Circle().stroke(.white, lineWidth: 5).frame(width: 79, height: 79)
            if model.recording {
              RoundedRectangle(cornerRadius: 8).fill(CameraStyle.red).frame(width: 32, height: 32)
            } else if model.capturePhase.isBusy {
              ProgressView().tint(CameraStyle.red)
            } else {
              Circle().fill(CameraStyle.red).frame(width: 62, height: 62)
            }
          }.frame(width: 84, height: 84)
        }
        .accessibilityLabel(model.recording ? "Stop scan" : "Start scan")
        .accessibilityIdentifier(model.recording ? "scan.stop" : "home.record")
        .disabled(
          !model.cameraReady || !model.canRecord
            || (!model.recording && model.capturePhase.isBusy)
        )
        .opacity(model.isTalking || !model.question.isEmpty ? 0 : 1)
        .allowsHitTesting(!model.isTalking && model.question.isEmpty)
        .position(x: geometry.size.width / 2, y: 42)
        if !model.question.isEmpty && !model.isTalking {
          answer.frame(width: max(44, geometry.size.width - 88))
            .position(x: (geometry.size.width - 88) / 2, y: 42)
        }
        // Keep the same gesture host alive when Listening moves the microphone.
        microphone(size: 56).scaleEffect(model.isTalking ? 80.0 / 56 : 1)
          .opacity(model.capturePhase.isBusy ? 0 : 1)
          .allowsHitTesting(!model.capturePhase.isBusy)
          .position(x: model.isTalking ? geometry.size.width / 2 : geometry.size.width - 36, y: 42)
        if model.recording && model.splatFallback {
          Image(systemName: "bolt.slash").accessibilityLabel("Point preview; training unavailable")
            .frame(width: 44, height: 44).position(x: geometry.size.width - 36, y: 42)
        }
      }
    }.frame(height: 84).padding(.top, 20).buttonStyle(.plain)
  }

  private var listening: some View {
    VStack(spacing: 24) {
      Text(
        model.transcript.isEmpty
          ? (model.preparingVoice ? "Preparing voice" : "Listening") : model.transcript
      )
      .font(.system(size: 24, weight: .medium)).multilineTextAlignment(.center)
      .padding(.horizontal, 12)
      TimelineView(.animation(minimumInterval: 0.15)) { timeline in
        HStack(spacing: 5) {
          ForEach(0..<14) { i in
            let phase = timeline.date.timeIntervalSinceReferenceDate * 4 + Double(i)
            let height = CGFloat(10 + 28 * abs(sin(phase)))
            Capsule().fill(CameraStyle.amber).frame(
              width: 5, height: height)
          }
        }.frame(height: 40)
      }.accessibilityLabel("Listening")
      Text("Release to ask").font(.caption).foregroundStyle(CameraStyle.muted)
    }.padding(.bottom, 12)
  }

  private var answer: some View {
    HStack(spacing: 12) {
      if model.answering {
        ProgressView().tint(CameraStyle.amber)
        Text(model.question).font(.system(size: 15, weight: .medium)).lineLimit(2)
      } else {
        Button(action: model.replayAnswer) {
          Label(
            model.answerText.isEmpty ? "No answer" : model.answerText,
            systemImage: "speaker.wave.2.fill"
          ).font(.system(size: 17, weight: .semibold)).lineLimit(3)
        }.accessibilityLabel("Replay answer: \(model.answerText)")
      }
      Button {
        model.clearAnswer()
      } label: {
        Image(systemName: "xmark").frame(width: 44, height: 44)
      }
      .accessibilityLabel("Dismiss answer")
    }
    .padding(.leading, 18).padding(.vertical, 4).background(.black.opacity(0.8), in: Capsule())
  }

  private func microphone(size: CGFloat) -> some View {
    Image(systemName: "mic.fill").font(.system(size: size == 80 ? 30 : 24))
      .foregroundStyle(CameraStyle.background).frame(width: size, height: size)
      .background(CameraStyle.amber, in: Circle())
      .overlay(
        Circle().stroke(CameraStyle.amber.opacity(model.isTalking ? 0.25 : 0), lineWidth: 10)
      )
      .contentShape(Circle())
      .gesture(
        DragGesture(minimumDistance: 0).onChanged { _ in setPressing(true) }
          .onEnded { _ in setPressing(false) }
      )
      .accessibilityLabel(model.isTalking ? "Release to ask" : "Hold to ask")
      .accessibilityIdentifier("ask.microphone")
      .accessibilityAddTraits(.isButton)
      .accessibilityAction { setPressing(!isPressing) }
      .allowsHitTesting(model.cameraReady && scenePhase == .active)
  }

  private func icon(_ symbol: String, label: String, action: (() -> Void)?) -> some View {
    Button {
      action?()
    } label: {
      Image(systemName: symbol).font(.system(size: 20)).frame(width: 44, height: 44)
        .background(.black.opacity(0.55), in: Circle())
    }.accessibilityLabel(label).accessibilityIdentifier("home.\(label.lowercased())").disabled(
      action == nil)
  }

  private func setPressing(_ value: Bool) {
    guard value != isPressing else { return }
    if value && (!model.cameraReady || scenePhase != .active) { return }
    isPressing = value
    if value { model.beginTalking() } else { model.endTalking() }
  }

  private func loadLatestThumbnail() async {
    var directory = model.latestCapture
    if directory == nil {
      let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Captures")
      directory = try? await Task.detached {
        try LocalCapture.list(at: root).max { left, right in
          let a = try? left.directory.resourceValues(forKeys: [.creationDateKey]).creationDate
          let b = try? right.directory.resourceValues(forKeys: [.creationDateKey]).creationDate
          return (a ?? .distantPast) < (b ?? .distantPast)
        }?.directory
      }.value
    }
    guard let directory else { return }
    latestThumbnail = await CapturePreview.image(directory: directory)
  }
}

private struct ProgressRing: View {
  let progress: Double
  var body: some View {
    ZStack {
      Circle().stroke(CameraStyle.muted.opacity(0.25), lineWidth: 4)
      Circle().trim(from: 0, to: min(1, max(0, progress)))
        .stroke(CameraStyle.teal, style: StrokeStyle(lineWidth: 4, lineCap: .round))
        .rotationEffect(.degrees(-90))
    }.padding(6)
  }
}

@MainActor private struct CaptureUploadScreen: View {
  let capture: LocalCapture
  let siteName: String?
  let siteID: UUID?
  let duration: TimeInterval
  let onSettings: (() -> Void)?
  let onSite: () -> Void
  let done: () -> Void
  @State private var preview: UIImage?
  @State private var replay = false
  @State private var queue = CaptureUploadQueue.shared
  @Environment(\.scenePhase) private var scenePhase
  private var state: UploadState? { queue.states[capture.id] }
  private var issue: String? { queue.issues[capture.id] }
  private var progress: Double { CaptureUploadQueue.progress(state) }

  var body: some View {
    ZStack {
      Color.black.opacity(0.62).ignoresSafeArea()
      VStack(spacing: 20) {
        Button {
          replay = true
        } label: {
          ZStack(alignment: .bottomTrailing) {
            Group {
              if let preview {
                Image(uiImage: preview).resizable().scaledToFill()
              } else {
                CameraStyle.background.overlay { Image(systemName: "photo").font(.largeTitle) }
              }
            }.frame(height: 220).frame(maxWidth: .infinity).clipped()
            Image(systemName: "arrow.counterclockwise").frame(width: 44, height: 44)
              .background(.black.opacity(0.6), in: Circle()).padding(10)
          }.clipShape(RoundedRectangle(cornerRadius: 18))
        }.accessibilityLabel("Replay capture")
        HStack(spacing: 16) {
          if issue == "Sign in" {
            Button("Sign in") { onSettings?() }.foregroundStyle(CameraStyle.amber)
              .frame(minWidth: 56, minHeight: 56).disabled(onSettings == nil)
          } else if let issue {
            Button {
              if issue == "Location needed" { onSite() } else { Task { await upload() } }
            } label: {
              Label(
                issue,
                systemImage: issue == "Location needed" ? "location.slash" : "arrow.clockwise"
              )
              .font(.caption).frame(minHeight: 56)
            }.accessibilityLabel(issue)
          } else {
            ZStack {
              ProgressRing(progress: progress)
              Text("\(Int(progress * 100))%").font(
                .system(size: 13, weight: .medium, design: .monospaced))
            }.frame(width: 56, height: 56).accessibilityLabel(
              "Uploaded \(Int(progress * 100)) percent")
          }
          VStack(alignment: .leading, spacing: 3) {
            Text(siteName ?? "Capture").font(.system(size: 17, weight: .semibold)).lineLimit(1)
            Text(CapturePhase.time(duration)).font(.system(size: 13, design: .monospaced))
              .foregroundStyle(CameraStyle.muted)
            if state?.jobID != nil {
              Text("Refining").font(.caption).foregroundStyle(CameraStyle.teal)
            }
          }
          Spacer(minLength: 0)
        }
        Button(action: done) {
          Text("Done").font(.system(size: 16, weight: .semibold)).foregroundStyle(
            CameraStyle.background
          )
          .frame(maxWidth: .infinity).frame(height: 52)
          .background(CameraStyle.amber, in: RoundedRectangle(cornerRadius: 14))
        }.accessibilityIdentifier("upload.done")
      }
      .padding(20).background(CameraStyle.panel, in: RoundedRectangle(cornerRadius: 28))
      .padding(.horizontal, 24)
    }
    .buttonStyle(.plain)
    .task {
      preview = await CapturePreview.image(directory: capture.directory)
      await upload()
      while !Task.isCancelled {
        await queue.refresh(capture)
        try? await Task.sleep(for: .seconds(1))
      }
    }
    .onChange(of: scenePhase) { _, phase in if phase == .active { Task { await upload() } } }
    .onChange(of: siteID) { _, _ in Task { await upload() } }
    .sheet(isPresented: $replay) { CaptureReplay(directory: capture.directory) }
  }
  private func upload() async { await queue.enqueue(capture, siteName: siteName, siteID: siteID) }
}

@MainActor private struct CaptureReplay: View {
  let directory: URL
  @Environment(\.dismiss) private var dismiss
  @State private var player: AVPlayer?
  var body: some View {
    VideoPlayer(player: player).ignoresSafeArea()
      .overlay(alignment: .topTrailing) {
        Button {
          dismiss()
        } label: {
          Image(systemName: "xmark").frame(width: 44, height: 44).background(.black, in: Circle())
        }
        .accessibilityLabel("Close replay").padding()
      }
      .task {
        player = AVPlayer(url: directory.appendingPathComponent("rgb.mov"))
        player?.play()
      }
      .onDisappear {
        player?.pause()
        player = nil
      }
  }
}

@MainActor private struct CameraSitePicker: View {
  let model: CameraModel
  @Environment(\.dismiss) private var dismiss
  @State private var sites: [GlobeSite] = []
  @State private var loading = true
  @State private var failed = false
  var body: some View {
    NavigationStack {
      List {
        Button("Use GPS") {
          model.chooseSite(id: nil, name: nil)
          dismiss()
        }
        .frame(minHeight: 44)
        if loading { ProgressView().accessibilityLabel("Loading sites") }
        if failed { Label("Sites unavailable", systemImage: "wifi.exclamationmark") }
        ForEach(sites) { site in
          Button(site.name) {
            model.chooseSite(id: site.id, name: site.name)
            dismiss()
          }
          .frame(minHeight: 44)
        }
      }
      .navigationTitle("Site").navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button {
            dismiss()
          } label: {
            Image(systemName: "xmark").frame(width: 44, height: 44)
          }
          .accessibilityLabel("Close sites")
        }
      }
      .task {
        defer { loading = false }
        guard let root = ServiceConfiguration(defaults: .standard).apiRoot else {
          failed = true
          return
        }
        do { sites = try await APIGlobeDataSource(baseURL: root).sites() } catch { failed = true }
      }
    }.tint(CameraStyle.amber).presentationDetents([.medium, .large])
  }
}
