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
              .buttonStyle(.borderedProminent).disabled(LaunchMode.current.isUITest)
          }
          if let report {
            VStack(spacing: 12) {
              ForEach(Array(report.runs.enumerated()), id: \.offset) { _, run in
                BenchmarkRunRow(run: run)
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

    private func start() {
      guard !LaunchMode.current.isUITest else { return }
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

  /// One benchmark run. Its own view with plainly typed values: the inline form, with
  /// optional maps and `??` inside the screen's ForEach, risks Xcode's type-check timeout.
  @MainActor private struct BenchmarkRunRow: View {
    let run: BenchmarkRun

    var body: some View {
      HStack(spacing: 12) {
        Image(systemName: thermalIcon)
          .foregroundStyle(thermalColor)
          .accessibilityLabel(Text("Thermal state \(run.worstThermal)"))
        VStack(alignment: .leading, spacing: 4) {
          Text(title).font(.subheadline.monospacedDigit())
          HStack {
            Label(seconds, systemImage: "timer")
            Label(memory, systemImage: "memorychip")
            Label(psnr, systemImage: "eye")
          }
          .font(.caption.monospacedDigit())
          .foregroundStyle(HexStyle.muted)
          if let error = run.error {
            Text(error).font(.caption).foregroundStyle(Color.orange)
          }
        }
        Spacer(minLength: 0)
        Image(systemName: statusIcon).accessibilityLabel(Text(run.status))
      }
    }

    private var title: String {
      let thousands: Int = run.configuration.maxGaussians / 1000
      let side: Int = run.configuration.maxSide
      return "\(thousands)k · \(side) px"
    }
    private var seconds: String {
      guard let value: Double = run.totalSeconds else { return "—" }
      return String(format: "%.1fs", value)
    }
    private var memory: String {
      guard let bytes: UInt64 = run.peakPhysicalFootprint else { return "—" }
      return "\(bytes / 1_048_576) MB"
    }
    private var psnr: String {
      guard let value: Float = run.psnr else { return "—" }
      return String(format: "%.1f", Double(value))
    }
    private var statusIcon: String {
      run.status == "finished" ? "checkmark.circle" : "exclamationmark.circle"
    }
    private var thermalColor: Color {
      run.worstThermal >= 2 ? Color.orange : Color.secondary
    }
    private var thermalIcon: String {
      switch run.worstThermal {
      case 3: return "thermometer.sun.fill"
      case 2: return "thermometer.high"
      case 1: return "thermometer.medium"
      default: return "thermometer.low"
      }
    }
  }
#endif
