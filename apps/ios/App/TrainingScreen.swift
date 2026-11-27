#if canImport(Msplat) && os(iOS)
  import Foundation
  import Globe
  import LiveSplat
  import SwiftUI
  import UIKit
  import Upload

  /// Overlay presentation keeps the scan's ARView mounted underneath this screen.
  @MainActor struct TrainingScreen: View {
    let registry: PieceRegistry?
    let camera: CameraModel?
    let done: () -> Void
    @State private var model: TrainingModel
    @State private var selection = SplatSelection()
    @State private var fixtureCloudReady = false
    @State private var cloudRoute: GlobeRoute?
    @State private var cloudController: TrainingCloudController?
    @State private var remote: CatalogCaptureState?
    @State private var cloudLabel = "Connecting…"
    @State private var refineConfirmation = false
    @State private var refineFailed = false
    @State private var refining = false
    @State private var closing = false
    @State private var showSettings = false
    @State private var drag = CGSize.zero
    @State private var magnification: CGFloat = 1
    @State private var queue = CaptureUploadQueue.shared
    @Environment(\.scenePhase) private var phase

    init(capture: LocalCapture, registry: PieceRegistry? = nil, camera: CameraModel? = nil,
      done: @escaping () -> Void) {
      self.registry = registry; self.camera = camera; self.done = done
      _model = State(initialValue: TrainingModel(capture: capture))
    }
    private var previewAspect: CGFloat? {
      guard let surface = model.surface else { return nil }
      return CGFloat(surface.width) / CGFloat(max(1, surface.height))
    }
    private var isResult: Bool { model.experience.stage == .result }
    private var canPlace: Bool {
      !LaunchMode.current.isUITest && camera?.canPlaceSplat(capture: model.capture.directory) == true
    }
    private var needsCloudSignIn: Bool {
      LaunchMode.current.captureAPI == nil && APICredentials.token.isEmpty
    }
    @ViewBuilder private var phonePreview: some View {
      if let image = model.previewImage {
        Image(uiImage: image).resizable().scaledToFit()
      } else if LaunchMode.current.isUITest {
        HexStyle.background
      } else {
        TrainingMetalView(surface: model.surface, renderer: model.renderer, paused: showCloud)
      }
    }
    private var showCloud: Bool { selection.source == .cloud && !model.inPlace }
    var body: some View {
      GeometryReader { geometry in
        ZStack {
          if !model.inPlace { HexStyle.background.ignoresSafeArea() }
          if !model.inPlace {
            phonePreview
              .aspectRatio(previewAspect, contentMode: .fit)
              .ignoresSafeArea().opacity(showCloud ? 0 : 1)
              .scaleEffect(model.experience.stage == .settling ? 1.025 : 1)
              .animation(.easeOut(duration: 0.6), value: model.experience.stage)
              .accessibilityHidden(showCloud)
              .accessibilityIdentifier(TrainingID.orbit)
              .accessibilityLabel(model.previewImage == nil ? "Splat. Drag to orbit, pinch to zoom, double tap to focus." : "Phone preview")
              .gesture(orbitGesture)
              .simultaneousGesture(zoomGesture)
              .onTapGesture(count: 2) { withAnimation { model.focus() } }
              .accessibilityAction(named: "Focus") { model.focus() }
              .accessibilityAction(named: "Zoom in") { model.orbit.zoom(factor: 1.25) }
              .accessibilityAction(named: "Zoom out") { model.orbit.zoom(factor: 0.8) }
          }
          if fixtureCloudReady, isResult, let image = model.previewImage {
            Image(uiImage: image).resizable().scaledToFit()
              .opacity(showCloud ? 1 : 0).allowsHitTesting(showCloud)
              .accessibilityHidden(!showCloud).accessibilityIdentifier(TrainingID.orbit)
              .accessibilityLabel("Simulated cloud scan")
          }
          if let cloudController, isResult {
            TrainingCloudPreview(controller: cloudController)
              .ignoresSafeArea().opacity(showCloud ? 1 : 0).allowsHitTesting(showCloud)
              .accessibilityHidden(!showCloud).accessibilityIdentifier(TrainingID.orbit)
              .accessibilityLabel("Cloud splat. Drag to orbit, pinch to zoom.")
              .accessibilityAction(named: "Focus") { cloudController.focus() }
          }
          VStack(spacing: 16) {
            header
            Spacer()
            if model.experience.stage == .choosing { choices }
            else if model.experience.stage == .failed { failure }
            else if isResult { resultControls }
            else { trainingControls }
          }.padding(24)
        }
        .animation(.easeInOut(duration: 0.7), value: selection.source)
        .animation(.easeInOut(duration: 0.4), value: model.inPlace)
        .task {
          await model.loadExisting()
          if LaunchMode.current.isUITest { return }
          while !Task.isCancelled {
            if !showCloud { await model.tick(size: geometry.size, camera: camera) }
            do { try await Task.sleep(for: .milliseconds(80)) } catch { return }
          }
        }
      }
      .preferredColorScheme(.dark).foregroundStyle(HexStyle.text)
      .background(model.inPlace ? Color.clear : HexStyle.background)
      .task { await pollCloud() }
      .onChange(of: queue.states[model.capture.id]) { _, _ in
        if LaunchMode.current.isUITest { Task { await refreshCloud() } }
      }
      .onChange(of: model.experience.stage) { _, stage in
        camera?.setTrainingBusy(stage.busy, capture: model.capture.directory)
        if LaunchMode.current.isUITest, stage == .result { Task { await refreshCloud() } }
      }
      .onChange(of: phase) { _, value in
        Task { await model.setActive(value == .active) }
        if value != .active { camera?.showTrainedSplat(nil) }
        else if model.inPlace { camera?.showTrainedSplat(model.renderer) }
      }
      .onChange(of: cloudController?.failed) { _, value in
        if value == true { withAnimation { selection.select(.phone) } }
      }
      .onChange(of: model.inPlace) { _, value in
        camera?.showTrainedSplat(value ? model.renderer : nil)
      }
      .onDisappear {
        camera?.showTrainedSplat(nil)
        cloudController?.onReady = nil
        cloudController?.webView.stopLoading()
        Task {
          await model.stop()
          camera?.setTrainingBusy(false, capture: model.capture.directory)
        }
      }
      .sheet(isPresented: $showSettings) {
        if let registry { NavigationStack { SettingsScreen(registry: registry) } }
      }
      .confirmationDialog("Refine in cloud?", isPresented: $refineConfirmation, titleVisibility: .visible) {
        Button("Refine in cloud") { Task { await refine() } }
          .accessibilityIdentifier(TrainingID.confirm)
        Button("Cancel", role: .cancel) {}.accessibilityIdentifier(TrainingID.decline)
      } message: {
        Text("Estimated \(CaptureStatus.cloudRefineEstimatedCost) per scan. Your phone version stays available.")
      }
    }
    private var header: some View {
      HStack {
        if isResult {
          Label("Your scan", systemImage: "sparkles").font(.headline)
        } else {
          Label("On your phone", systemImage: "iphone").font(.headline)
        }
        Spacer()
        if model.simulated || camera?.simulated == true || LaunchMode.current.isUITest {
          Text("Simulated").font(.caption)
        }
        Button { close() } label: {
          Group {
            if closing { ProgressView() } else { Image(systemName: "xmark") }
          }.frame(width: 48, height: 48)
            .background(.ultraThinMaterial, in: Circle())
        }.accessibilityLabel(model.experience.stage.busy ? "Cancel training" : "Close")
          .accessibilityIdentifier(model.experience.stage.busy ? TrainingID.cancel : TrainingID.close)
          .disabled(closing)
      }
    }
    private var choices: some View {
      VStack(spacing: 22) {
        Image(systemName: "sparkles.rectangle.stack").font(.system(size: 52, weight: .ultraLight))
          .foregroundStyle(HexStyle.teal)
        Text("Bring it to life").font(.system(size: 30, weight: .semibold, design: .rounded))
        HStack(spacing: 12) {
          preset("Quick", symbol: "bolt", best: false, id: TrainingID.quick)
          preset("Best", symbol: "sparkles", best: true, id: TrainingID.best)
        }
        Button { model.start() } label: {
          Label("Train on phone", systemImage: "iphone.gen3.radiowaves.left.and.right")
            .font(.headline).frame(maxWidth: .infinity).frame(height: 56)
        }.buttonStyle(.borderedProminent).tint(HexStyle.teal)
          .accessibilityIdentifier(TrainingID.begin)
      }.padding(24).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28))
    }
    private func preset(_ title: String, symbol: String, best: Bool, id: String) -> some View {
      Button { model.best = best; UISelectionFeedbackGenerator().selectionChanged() } label: {
        Label(title, systemImage: symbol).font(.subheadline.weight(.semibold))
          .frame(maxWidth: .infinity).frame(height: 48)
          .background(model.best == best ? HexStyle.teal.opacity(0.25) : Color.white.opacity(0.06),
            in: RoundedRectangle(cornerRadius: 14))
      }.accessibilityIdentifier(id).accessibilityAddTraits(model.best == best ? .isSelected : [])
    }
    private var trainingControls: some View {
      VStack(spacing: 16) {
        HStack(spacing: 18) {
          ZStack {
            Circle().stroke(Color.white.opacity(0.12), lineWidth: 2)
            Circle().trim(from: 0, to: CGFloat(model.progress)).stroke(HexStyle.teal,
              style: StrokeStyle(lineWidth: 2, lineCap: .round)).rotationEffect(.degrees(-90))
            Image(systemName: model.experience.stage == .settling ? "checkmark" : "sparkles")
          }.frame(width: 52, height: 52).accessibilityIdentifier(TrainingID.progress)
            .accessibilityValue("\(Int(model.progress * 100)) percent")
          VStack(alignment: .leading, spacing: 4) {
            Text(trainingTitle).font(.headline)
            Text(model.eta).font(.caption.monospaced()).foregroundStyle(HexStyle.muted)
          }
          Spacer()
          sharpness
        }
        if model.thermal {
          Label("Taking it gently", systemImage: "thermometer.medium").font(.caption)
            .foregroundStyle(HexStyle.muted).accessibilityIdentifier(TrainingID.thermal)
        }
        if model.experience.stage == .paused || model.experience.stage == .cooling {
          Button { Task { await model.resume() } } label: { Label("Resume", systemImage: "play.fill") }
            .frame(minHeight: 48).accessibilityIdentifier(TrainingID.resume)
        } else if model.experience.stage == .training {
          Button { Task { await model.pause() } } label: { Label("Pause", systemImage: "pause.fill") }
            .frame(minHeight: 48).accessibilityIdentifier(TrainingID.pause)
        }
      }.padding(20).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
    }
    private var trainingTitle: String {
      switch model.experience.stage {
      case .preparing: "Finding the details"
      case .paused: "Paused"
      case .cooling: "Let your phone cool"
      case .settling: "Looking good"
      case .cancelled: "Training stopped"
      default: "Coming to life"
      }
    }
    private var sharpness: some View {
      VStack(spacing: 3) {
        Text(sharpnessText).font(.system(size: 22, weight: .light, design: .rounded))
        Text("Sharpness").font(.caption2).foregroundStyle(HexStyle.muted)
      }.accessibilityIdentifier(TrainingID.sharpness)
    }
    private var sharpnessText: String {
      if let score = model.sharpness { return String(score) }
      return "—"
    }
    private var failure: some View {
      VStack(spacing: 16) {
        Image(systemName: "sparkles").font(.largeTitle).foregroundStyle(HexStyle.teal)
        Text("Let’s try that again").font(.title3.weight(.semibold))
        Text("Your scan is safe.").font(.footnote).foregroundStyle(HexStyle.muted)
        Button("Retry", systemImage: "arrow.clockwise") { model.start() }
          .frame(minHeight: 48).accessibilityIdentifier(TrainingID.retry)
      }.frame(maxWidth: .infinity).padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
    }
    private var resultControls: some View {
      VStack(spacing: 16) {
        if model.inPlace { inPlaceControls }
        else {
          sourceControls
          placementControl
          refinementControls
          retryControls
        }
      }.padding(18).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
    }
    private var inPlaceControls: some View {
      HStack {
        Button { model.cameraOnly.toggle() } label: {
          Label(model.cameraOnly ? "Splat" : "Camera", systemImage: model.cameraOnly ? "sparkles" : "camera")
        }.frame(minHeight: 48)
          .accessibilityIdentifier(model.cameraOnly ? TrainingID.splat : TrainingID.camera)
        Spacer()
        Button { model.inPlace = false } label: { Image(systemName: "arrow.down.right.and.arrow.up.left") }
          .frame(width: 48, height: 48).accessibilityLabel("Leave in-place view")
          .accessibilityIdentifier(TrainingID.leave)
      }
    }
    private var sourceControls: some View {
      HStack(spacing: 18) {
        sourceButton(.phone, label: "Phone", symbol: "iphone", id: TrainingID.phone)
        sourceButton(.cloud, label: "Cloud", symbol: "cloud", id: TrainingID.cloud)
          .disabled(!selection.cloudAvailable)
        Spacer()
        Button {
          if showCloud { cloudController?.focus() } else { model.focus() }
        } label: { Image(systemName: "scope") }
          .frame(width: 48, height: 48).accessibilityLabel("Focus")
          .accessibilityIdentifier(TrainingID.focus)
        if let url = shareURL {
          ShareLink(item: url) { Image(systemName: "square.and.arrow.up").frame(width: 48, height: 48) }
            .accessibilityLabel("Share splat").accessibilityIdentifier(TrainingID.share)
        }
      }
    }
    @ViewBuilder private var placementControl: some View {
      if canPlace {
        Button {
          model.cameraOnly = true
          model.inPlace = true
          Task {
            try? await Task.sleep(for: .milliseconds(100))
            withAnimation(.easeInOut(duration: 0.6)) { model.cameraOnly = false }
          }
        } label: { Label("See it in place", systemImage: "arkit") }
          .frame(minHeight: 48).accessibilityIdentifier(TrainingID.inPlace)
      }
    }
    @ViewBuilder private var refinementControls: some View {
      if remote?.status == "complete", !selection.cloudAvailable {
        Label("Opening cloud scan", systemImage: "cloud").font(.caption)
      }
      if !selection.cloudAvailable, remote?.status != "complete" {
        Button {
          if needsCloudSignIn { showSettings = true }
          else { refineConfirmation = true }
        } label: {
          Label("Refine in cloud", systemImage: "cloud")
            .overlay(alignment: .topLeading) {
              Image(systemName: "sparkles").font(.system(size: 10)).offset(x: 8, y: -4)
            }.frame(maxWidth: .infinity).frame(height: 48)
        }.buttonStyle(.borderedProminent).tint(HexStyle.teal)
          .accessibilityIdentifier(TrainingID.refine).disabled(refining || cloudRunning)
      }
      if remote != nil || refining || cloudRunning {
        Label(cloudLabel, systemImage: "icloud.and.arrow.up").font(.caption)
          .accessibilityIdentifier(TrainingID.refineProgress)
      }
    }
    @ViewBuilder private var retryControls: some View {
      if let cloudController, cloudController.failed {
        Button("Try cloud view again", systemImage: "arrow.clockwise") {
          selection.select(.phone)
          Task { await cloudController.load() }
        }.frame(minHeight: 48).accessibilityIdentifier(TrainingID.cloudRetry)
      }
      if refineFailed || queue.issues[model.capture.id] == "Retry cloud" {
        Text("Cloud needs another try. Your phone scan is safe.").font(.caption)
        Button("Try cloud again", systemImage: "arrow.clockwise") {
          if needsCloudSignIn { showSettings = true }
          else { refineConfirmation = true }
        }.frame(minHeight: 48).accessibilityIdentifier(TrainingID.refineRetry)
      }
    }
    private var cloudRunning: Bool {
      if queue.pendingRefinements.contains(model.capture.id) { return true }
      if remote?.status == "not-started", queue.states[model.capture.id]?.jobID == nil { return false }
      return cloudLabel.hasPrefix("Uploading") || cloudLabel == "Processing" || cloudLabel.hasPrefix("Training")
    }
    private var shareURL: URL? {
      if showCloud, let route = cloudRoute, let registry {
        guard let scanID = route.scanID else { return nil }
        return CloudSplatRecord(siteID: route.siteID, scanID: scanID)
          .shareURL(webBaseURL: registry.configuration.webBaseURL)
      }
      return model.shareURL
    }
    private func sourceButton(_ source: SplatSelection.Source, label: String, symbol: String, id: String) -> some View {
      Button { selection.select(source); UISelectionFeedbackGenerator().selectionChanged() } label: {
        Image(systemName: symbol).frame(width: 48, height: 48)
          .background(selection.source == source ? HexStyle.teal.opacity(0.25) : Color.clear, in: Capsule())
      }.accessibilityLabel(label).accessibilityIdentifier(id)
        .accessibilityAddTraits(selection.source == source ? .isSelected : [])
    }
    private var orbitGesture: some Gesture {
      DragGesture().onChanged { value in
        model.interacting = true
        let dx = Float(value.translation.width - drag.width)
        let dy = Float(value.translation.height - drag.height)
        model.orbit.drag(x: dx, y: dy)
        drag = value.translation
      }.onEnded { _ in drag = .zero; model.interacting = false }
    }
    private var zoomGesture: some Gesture {
      MagnificationGesture().onChanged { value in
        model.interacting = true
        model.orbit.zoom(factor: Float(value / magnification))
        magnification = value
      }.onEnded { _ in magnification = 1; model.interacting = false }
    }
    private func close() {
      closing = true
      camera?.showTrainedSplat(nil)
      Task { await model.stop(); done() }
    }
    private func refine() async {
      refining = true; refineFailed = false
      do { try await queue.requestRefinement(model.capture) }
      catch { refineFailed = true }
      refining = false
      await refreshCloud()
    }
    private func pollCloud() async {
      while !Task.isCancelled {
        await refreshCloud()
        if LaunchMode.current.isUITest { return }
        do { try await Task.sleep(for: .seconds(3)) } catch { return }
      }
    }
    private func refreshCloud() async {
      await queue.refresh(model.capture)
      let upload = queue.states[model.capture.id] ?? model.capture.upload
      let local = LocalCapture(directory: model.capture.directory, upload: upload)
      do {
        let configuration = ServiceConfiguration(defaults: .standard)
        guard let fetched = try await CaptureStatus.fetch(local, configuration: configuration) else {
          cloudLabel = CaptureStatus.label(upload: upload, remote: remote, unavailable: false)
          return
        }
        remote = fetched
        cloudLabel = CaptureStatus.label(upload: upload, remote: fetched, unavailable: false)
        guard let route = CaptureStatus.trainedRoute(upload: upload, remote: fetched) else { return }
        cloudRoute = route
        if let scanID = route.scanID {
          try? CloudSplatRecord(siteID: route.siteID, scanID: scanID).save(for: model.capture)
        }
        if LaunchMode.current.isUITest {
          fixtureCloudReady = true
          if isResult { selection.cloudReady() }
          return
        }
        // Published 3D Tiles use the existing renderer; native Phone remains mounted.
        if isResult, cloudController == nil, registry != nil {
          let controller = TrainingCloudController(route: route, configuration: configuration)
          controller.onReady = {
            model.inPlace = false
            withAnimation {
              if selection.cloudAvailable { selection.select(.cloud) }
              else { selection.cloudReady() }
            }
          }
          cloudController = controller
          await controller.load()
        }
      } catch { cloudLabel = "Connection paused" }
    }
  }
#endif
