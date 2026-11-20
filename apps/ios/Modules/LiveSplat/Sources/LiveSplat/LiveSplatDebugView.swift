#if DEBUG && canImport(SwiftUI)
  import SwiftUI

  /// Standalone, manually stepped replay. No App, ARSession, or Metal is required.
  @MainActor public struct LiveSplatDebugView: View {
    @StateObject private var model: DebugReplayModel

    public init(
      source: any SplatFrameSource, renderer: any LiveSplatRenderer = DefaultLiveSplatRenderer()
    ) {
      _model = StateObject(wrappedValue: DebugReplayModel(source: source, renderer: renderer))
    }

    public var body: some View {
      VStack(alignment: .leading, spacing: 16) {
        Text("Live splat — Simulated replay").font(.headline)
        Text("Coverage \(model.coverage * 100, specifier: "%.1f")%")
          .monospacedDigit()
        ProgressView(value: model.coverage)
        Text("\(model.points) seeded points · \(model.frames) frames")
        Text(model.message)
        HStack {
          Button("Next frame") { Task { await model.step() } }
            .disabled(model.busy || model.ended)
          Button("Restart") { model.restart() }.disabled(model.busy)
        }
      }
      .padding()
      .onDisappear { model.stop() }
    }
  }

  @MainActor private final class DebugReplayModel: ObservableObject {
    @Published var coverage = 0.0
    @Published var points = 0
    @Published var frames = 0
    @Published var message = "Ready"
    @Published var busy = false
    @Published var ended = false
    private let source: any SplatFrameSource
    private let renderer: any LiveSplatRenderer
    private var generation = 0

    init(source: any SplatFrameSource, renderer: any LiveSplatRenderer) {
      self.source = source
      self.renderer = renderer
      restart()
    }
    func restart() {
      generation += 1
      source.reset()
      renderer.start()
      coverage = 0
      points = 0
      frames = 0
      ended = false
      message = "Ready"
    }
    func stop() {
      generation += 1
      renderer.stop()
      ended = true
      message = "Stopped; restart to replay"
    }
    func step() async {
      guard !busy, !ended else { return }
      busy = true
      let currentGeneration = generation
      defer { busy = false }
      do {
        let event = try await source.next()
        guard generation == currentGeneration else { return }
        guard let event else {
          renderer.stop()
          ended = true
          message = "Replay complete"
          return
        }
        renderer.ingest(event)
        frames += 1
        coverage = renderer.coverageFraction
        points = renderer.splats.count
        switch event {
        case .frame: message = "Tracked frame"
        case .trackingUnavailable: message = "Skipped: tracking or depth unavailable"
        }
      } catch {
        guard generation == currentGeneration else { return }
        renderer.stop()
        ended = true
        message = "Replay failed: \(error)"
      }
    }
  }

  private struct LiveSplatDebugPreview: PreviewProvider {
    @MainActor static var previews: some View {
      if let source = try? FakeFrameSource.plane() {
        LiveSplatDebugView(source: source, renderer: FakeLiveSplat())
      } else {
        Text("Unable to build simulated plane")
      }
    }
  }
#endif
