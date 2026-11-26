#if canImport(Msplat)
  import Capture
  import PhoneTrain
  import SwiftUI

  /// Hidden measurement UI only. Saved captures and reports stay on this device.
  @MainActor struct TrainBenchmarkScreen: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var benchmark = PhoneTrainBenchmark()
    @State private var report: BenchmarkReport?
    @State private var reportURL: URL?
    @State private var work: Task<Void, Never>?
    @State private var polling: Task<Void, Never>?
    @State private var message = "200k · 400k · 800k / 640 · 960 px / 600 steps"
    @State private var running = false

    var body: some View {
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          Text(message).font(.footnote).foregroundStyle(HexStyle.muted)
          if running {
            HStack {
              ProgressView()
              Button("Cancel", systemImage: "stop.circle") { Task { await benchmark.cancel() } }
            }
          } else {
            Button("Benchmark train", systemImage: "play.circle") { start() }
              .buttonStyle(.borderedProminent)
          }
          if let report {
            VStack(spacing: 12) {
              ForEach(Array(report.runs.enumerated()), id: \.offset) { _, run in
                HStack(spacing: 12) {
                  Image(systemName: thermalIcon(run.worstThermal))
                    .foregroundStyle(run.worstThermal >= 2 ? Color.orange : Color.secondary)
                    .accessibilityLabel("Thermal state \(run.worstThermal)")
                  VStack(alignment: .leading, spacing: 4) {
                    Text(
                      "\(run.configuration.maxGaussians / 1000)k · \(run.configuration.maxSide) px"
                    )
                    .font(.subheadline.monospacedDigit())
                    HStack {
                      metric("timer", run.totalSeconds.map { String(format: "%.1fs", $0) } ?? "—")
                      metric(
                        "memorychip",
                        run.peakPhysicalFootprint.map { "\($0 / 1_048_576) MB" } ?? "—")
                      metric("eye", run.psnr.map { String(format: "%.1f", $0) } ?? "—")
                    }.font(.caption.monospacedDigit()).foregroundStyle(HexStyle.muted)
                    if let error = run.error { Text(error).font(.caption).foregroundStyle(.orange) }
                  }
                  Spacer(minLength: 0)
                  Image(
                    systemName: run.status == "finished"
                      ? "checkmark.circle" : "exclamationmark.circle"
                  )
                  .accessibilityLabel(run.status)
                }
              }
            }.padding().background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 16))
          }
          if let reportURL {
            ShareLink(item: reportURL) { Label("Share report", systemImage: "square.and.arrow.up") }
          }
        }.padding()
      }
      .background(HexStyle.background).navigationTitle("Benchmark train")
      .onChange(of: scenePhase) { _, phase in
        Task { await benchmark.setActive(phase == .active) }
      }
      .onDisappear {
        polling?.cancel()
        work?.cancel()
        // Cooperatively stop at the current completed step, then save the report.
        Task { await benchmark.cancel() }
      }
    }

    private func metric(_ icon: String, _ value: String) -> some View {
      Label(value, systemImage: icon)
    }
    private func thermalIcon(_ state: Int) -> String {
      switch state {
      case 3: "thermometer.sun.fill"
      case 2: "thermometer.high"
      case 1: "thermometer.medium"
      default: "thermometer.low"
      }
    }
    private func start() {
      running = true
      report = nil
      reportURL = nil
      message = "Preparing most recent saved scan…"
      work = Task {
        do {
          let capture = try await Task.detached {
            let root = FileManager.default.urls(
              for: .applicationSupportDirectory, in: .userDomainMask)[0]
              .appendingPathComponent("Captures")
            // Same local inventory as Captures; manifest date determines recency.
            let captures = try LocalCapture.list(at: root)
            return captures.compactMap { local -> (URL, String)? in
              guard
                let data = try? Data(
                  contentsOf: local.directory.appendingPathComponent("manifest.json")),
                let manifest = try? JSONDecoder().decode(CaptureManifest.self, from: data)
              else { return nil }
              return (local.directory, manifest.startedAt)
            }.max { $0.1 < $1.1 }?.0
          }.value
          try Task.checkCancellation()
          guard let capture else { throw PrepError.invalid("No saved capture found") }
          await benchmark.setActive(scenePhase == .active)
          polling = Task {
            while !Task.isCancelled {
              report = await benchmark.report()
              try? await Task.sleep(for: .milliseconds(500))
            }
          }
          let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
          reportURL = try await benchmark.run(capture: capture, documents: documents)
          report = await benchmark.report()
          message = report?.stopReason ?? "Benchmark complete. PSNR and SSIM are in the report."
        } catch { message = String(describing: error) }
        polling?.cancel()
        polling = nil
        running = false
        work = nil
      }
    }
  }
#endif
