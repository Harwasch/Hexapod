import AVFoundation
import AVKit
import Capture
import Foundation
import Globe
import LiveSplat
import SwiftUI
import UIKit
import Upload

@MainActor struct HomeScreen: View {
  @State private var registry = PieceRegistry()
  init() { UIFixtureLaunch.install() }
  var body: some View {
    NavigationStack { CameraHost(registry: registry) }
      .tint(HexStyle.amber)
      .preferredColorScheme(.dark)
  }
}

enum UIFixtureLaunch {
  static var active: Bool {
    #if DEBUG
      let args = ProcessInfo.processInfo.arguments
      return args.firstIndex(of: "-uiTestFixture").map {
        args.indices.contains($0 + 1) && args[$0 + 1] == "yard-small"
      } ?? false
    #else
      false
    #endif
  }
  static func install() {
    #if DEBUG
      if active { _ = URLProtocol.registerClass(UIFixtureAPIProtocol.self) }
    #endif
  }
}

private enum HexStyle {
  static let background = Color(red: 13 / 255, green: 21 / 255, blue: 23 / 255)
  static let panel = Color(red: 20 / 255, green: 32 / 255, blue: 35 / 255)
  static let text = Color(red: 228 / 255, green: 236 / 255, blue: 233 / 255)
  static let muted = Color(red: 138 / 255, green: 160 / 255, blue: 155 / 255)
  static let teal = Color(red: 63 / 255, green: 208 / 255, blue: 201 / 255)
  static let amber = Color(red: 1, green: 181 / 255, blue: 71 / 255)
}

@MainActor struct CapturesScreen: View {
  let registry: PieceRegistry
  @State private var captures: [CaptureRow] = []
  @State private var unavailable = false
  @Environment(\.scenePhase) private var phase
  init(registry: PieceRegistry) { self.registry = registry }

  var body: some View {
    ScrollView {
      LazyVStack(spacing: 12) {
        if unavailable {
          Label("Unavailable", systemImage: "wifi.exclamationmark").padding(32)
        } else if captures.isEmpty {
          Label("No scans yet", systemImage: "square.stack").padding(48)
            .accessibilityIdentifier("captures.empty")
        }
        ForEach(captures) { capture in
          NavigationLink {
            CaptureDetailScreen(registry: registry, capture: capture)
          } label: {
            HStack(spacing: 14) {
              CaptureThumbnail(video: capture.video)
                .frame(width: 76, height: 76).clipShape(RoundedRectangle(cornerRadius: 12))
              VStack(alignment: .leading, spacing: 6) {
                Text(capture.name).font(.system(size: 16, weight: .semibold)).lineLimit(1)
                Text(capture.subtitle).font(.system(size: 12, design: .monospaced))
                  .foregroundStyle(HexStyle.muted)
              }.frame(maxWidth: .infinity, alignment: .leading)
              CaptureStatusPill(label: capture.status)
            }
            .padding(12).background(HexStyle.panel, in: RoundedRectangle(cornerRadius: 18))
          }
          .buttonStyle(.plain)
          .accessibilityIdentifier("capture.row.\(capture.id)")
        }
      }.padding(20)
    }
    .background(HexStyle.background).foregroundStyle(HexStyle.text)
    .navigationBarBackButtonHidden().toolbar(.hidden, for: .navigationBar)
    .safeAreaInset(edge: .top, spacing: 0) { CapturePageHeader() }
    .refreshable { await reload() }
    .task(id: phase) {
      guard phase == .active else { return }
      while !Task.isCancelled {
        await reload()
        if captures.contains(where: { $0.serverUnavailable }) { return }
        do { try await Task.sleep(for: .seconds(3)) } catch { return }
      }
    }
  }
  private func reload() async {
    do {
      let root = CaptureRow.root
      var rows = try await Task.detached(priority: .utility) {
        try LocalCapture.list(at: root).map(CaptureRow.init).sorted { $0.date > $1.date }
      }.value
      let previous = Dictionary(
        uniqueKeysWithValues: captures.compactMap { row in
          row.remote.map { (row.id, $0) }
        })
      for index in rows.indices { rows[index].remote = previous[rows[index].id] }
      captures = rows
      unavailable = false
      for index in rows.indices {
        guard !Task.isCancelled else { return }
        await rows[index].refresh(configuration: registry.configuration)
        captures = rows
      }
      captures = rows
      unavailable = false
    } catch { unavailable = true }
  }
}

@MainActor private struct CapturePageHeader: View {
  @Environment(\.dismiss) private var dismiss
  var body: some View {
    HStack(spacing: 12) {
      Button {
        dismiss()
      } label: {
        Image(systemName: "chevron.left").font(.system(size: 20, weight: .medium))
          .foregroundStyle(HexStyle.text).frame(width: 44, height: 44)
          .background(HexStyle.panel, in: Circle())
      }.accessibilityLabel("Back").accessibilityIdentifier("captures.back")
      Text("Captures").font(.system(size: 28, weight: .semibold))
      Spacer()
    }.padding(.horizontal, 20).padding(.top, 16)
      .background(HexStyle.background)
  }
}

@MainActor private struct CaptureDetailScreen: View {
  let registry: PieceRegistry
  @State var capture: CaptureRow
  @State private var player: AVPlayer?
  @State private var deleteConfirmation = false
  @State private var deleteFailed = false
  @State private var connectionRetry = 0
  @State private var queue = CaptureUploadQueue.shared
  private var deletionAllowed: Bool {
    capture.local.upload == nil ? queue.issues[capture.id] != nil : !capture.uploadActive
  }
  @Environment(\.dismiss) private var dismiss
  var body: some View {
    ScrollView {
      VStack(spacing: 20) {
        if let route = capture.route {
          GlobeScreen(registry: registry, route: route).frame(height: 360)
            .clipShape(RoundedRectangle(cornerRadius: 18))
            .accessibilityIdentifier("capture.trainedScan")
        } else if let player {
          VideoPlayer(player: player).frame(height: 300)
            .clipShape(RoundedRectangle(cornerRadius: 18))
            .accessibilityLabel("Scan replay")
        } else {
          CaptureThumbnail(video: capture.video).frame(height: 300)
            .clipShape(RoundedRectangle(cornerRadius: 18))
        }
        Text(capture.name).font(.title2.weight(.semibold))
        Text(capture.subtitle).font(.system(.caption, design: .monospaced))
          .foregroundStyle(HexStyle.muted)
        CaptureStatusPill(label: capture.status).accessibilityIdentifier("capture.status")
        if capture.serverUnavailable {
          Button("Retry connection") {
            connectionRetry += 1
          }.frame(minHeight: 44)
        }
        if let error = capture.remote?.fullError ?? capture.local.upload?.lastError {
          Text(error).font(.footnote).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        if let route = capture.route {
          NavigationLink {
            GlobeScreen(registry: registry, route: route)
          } label: {
            Label("Open on globe", systemImage: "globe.americas")
          }
          .frame(minHeight: 44).accessibilityIdentifier("capture.openGlobe")
        } else {
          Label("Not mapped yet", systemImage: "globe.americas")
            .foregroundStyle(HexStyle.muted).frame(minHeight: 44)
        }
        if capture.local.upload?.jobID == nil {
          if APICredentials.token.isEmpty {
            NavigationLink("Sign in") { SettingsScreen(registry: registry) }.frame(minHeight: 44)
          } else {
            Button("Retry upload", systemImage: "arrow.clockwise") {
              Task { await CaptureUploadQueue.shared.enqueue(capture.local) }
            }.frame(minHeight: 44).disabled(capture.uploadActive)
          }
        }
        HStack(spacing: 32) {
          ShareLink(item: capture.video) { Label("Share", systemImage: "square.and.arrow.up") }
            .frame(minHeight: 44).accessibilityIdentifier("capture.share")
          Button(role: .destructive) {
            deleteConfirmation = true
          } label: {
            Label("Delete", systemImage: "trash")
          }.frame(minHeight: 44).accessibilityIdentifier("capture.delete")
            .disabled(!deletionAllowed)
        }
        if deleteFailed { Label("Delete failed", systemImage: "exclamationmark.circle") }
      }.padding(20)
    }
    .background(HexStyle.background).foregroundStyle(HexStyle.text)
    .navigationTitle("Scan").navigationBarTitleDisplayMode(.inline)
    .toolbar(.visible, for: .navigationBar)
    .accessibilityIdentifier("capture.detail")
    .confirmationDialog(
      "Delete local scan?", isPresented: $deleteConfirmation, titleVisibility: .visible
    ) {
      Button("Delete local scan", role: .destructive) {
        // An active uploader owns these files. Deletion is available once it stops.
        do {
          if FileManager.default.fileExists(atPath: capture.local.uploadDirectory.path) {
            try FileManager.default.removeItem(at: capture.local.uploadDirectory)
          }
          try FileManager.default.removeItem(at: capture.local.directory)
          dismiss()
        } catch { deleteFailed = true }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("The uploaded copy stays on the server.")
    }
    .task(id: connectionRetry) {
      player = AVPlayer(url: capture.video)
      capture.serverUnavailable = false
      while !Task.isCancelled {
        var updated = capture
        await updated.refresh(configuration: registry.configuration)
        capture = updated
        if capture.serverUnavailable { return }
        do { try await Task.sleep(for: .seconds(3)) } catch { return }
      }
    }
    .onDisappear { player?.pause() }
  }
}

@MainActor private struct CaptureThumbnail: View {
  let video: URL
  @State private var thumbnail: UIImage?
  var body: some View {
    ZStack {
      HexStyle.panel
      if let thumbnail {
        Image(uiImage: thumbnail).resizable().scaledToFill()
      } else {
        Image(systemName: "video").foregroundStyle(HexStyle.muted)
      }
    }.clipped().accessibilityHidden(true)
      .task(id: video) {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: video))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 600, height: 600)
        if let result = try? await generator.image(at: .zero), !Task.isCancelled {
          thumbnail = UIImage(cgImage: result.image)
        }
      }
  }
}

private struct CaptureStatusPill: View {
  let label: String
  private var color: Color {
    if label.hasPrefix("Failed") { return .red }
    if label == "Ready" { return Color(red: 185 / 255, green: 196 / 255, blue: 106 / 255) }
    return label.hasPrefix("Uploading") ? HexStyle.teal : HexStyle.amber
  }
  var body: some View {
    Text(label).font(.system(size: 12, weight: .semibold, design: .monospaced))
      .padding(.horizontal, 10).padding(.vertical, 5)
      .foregroundStyle(color).background(color.opacity(0.15), in: Capsule())
      .accessibilityLabel(label)
  }
}

private struct CaptureRow: Identifiable, Sendable {
  var local: LocalCapture
  let manifest: CaptureManifest?
  let details: CaptureDetails?
  let date: Date
  let durationSeconds: Double?
  var remote: CatalogCaptureState?
  var serverUnavailable = false
  var id: String { local.id }
  static var root: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Captures")
  }
  init(_ local: LocalCapture) {
    self.local = local
    details = local.details
    manifest = try? JSONDecoder().decode(
      CaptureManifest.self,
      from: Data(contentsOf: local.directory.appendingPathComponent("manifest.json")))
    durationSeconds = manifest.flatMap {
      Self.lastTimestamp(in: local.directory.appendingPathComponent($0.frameLog))
    }
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    date =
      manifest.flatMap {
        formatter.date(from: $0.startedAt) ?? ISO8601DateFormatter().date(from: $0.startedAt)
      }
      ?? (try? local.directory.resourceValues(forKeys: [.creationDateKey]).creationDate)
      ?? .distantPast
  }
  var name: String {
    if let site = details?.siteName, !site.isEmpty { return site }
    if let name = local.upload?.name, name != "Capture", !name.isEmpty { return name }
    return date.formatted(date: .abbreviated, time: .omitted)
  }
  var video: URL { local.directory.appendingPathComponent(manifest?.video ?? "rgb.mov") }
  var subtitle: String {
    let day =
      Calendar.current.isDateInToday(date)
      ? "Today" : date.formatted(.dateTime.month(.abbreviated).day())
    return "\(day) · \(duration)"
  }
  var duration: String {
    guard let durationSeconds else { return "—" }
    let seconds = Int(max(0, durationSeconds))
    return String(format: "%02d:%02d", seconds / 60, seconds % 60)
  }
  private static func lastTimestamp(in log: URL) -> Double? {
    guard let handle = try? FileHandle(forReadingFrom: log) else { return nil }
    defer { try? handle.close() }
    do {
      let size = try handle.seekToEnd()
      try handle.seek(toOffset: size > 16_384 ? size - 16_384 : 0)
      let data = try handle.readToEnd() ?? Data()
      guard let last = data.split(separator: 10).last else { return nil }
      return try JSONDecoder().decode(CaptureFrame.self, from: Data(last)).timestamp
    } catch { return nil }
  }
  var status: String {
    CaptureStatus.label(upload: local.upload, remote: remote, unavailable: serverUnavailable)
  }

  var uploadActive: Bool {
    local.upload.map { $0.jobID == nil && $0.lastError == nil } ?? false
  }
  var route: GlobeRoute? {
    CaptureStatus.trainedRoute(upload: local.upload, remote: remote)
  }
  mutating func refresh(configuration: ServiceConfiguration) async {
    if let upload = try? UploadState.load(
      from: local.uploadDirectory.appendingPathComponent("upload.json"))
    {
      local = LocalCapture(directory: local.directory, upload: upload)
    }
    guard let base = configuration.apiRoot,
      let id = local.upload?.captureID.flatMap(UUID.init(uuidString:)), local.upload?.jobID != nil
    else { return }
    do {
      remote = try await CatalogCaptureState.fetch(captureID: id, baseURL: base)
      serverUnavailable = false
    } catch { if !Task.isCancelled { serverUnavailable = true } }
  }
}

@MainActor struct GlobeScreen: View {
  let registry: PieceRegistry
  let route: GlobeRoute?
  @AppStorage("hex.testflight.debugEnabled") private var testing = false
  init(registry: PieceRegistry, route: GlobeRoute? = nil) {
    self.registry = registry
    self.route = route
  }
  private var debugChoices: Bool {
    #if DEBUG
      true
    #else
      PieceRegistry.menuAvailable(
        debug: false, receipt: Bundle.main.appStoreReceiptURL?.lastPathComponent, enabled: testing)
    #endif
  }
  var body: some View {
    let choice =
      UIFixtureLaunch.active ? "fake" : (debugChoices ? registry.choice("globe") : "real")
    Group {
      if let source = try? registry.globeSources.resolve(choice) {
        VStack(spacing: 0) {
          if choice == "fake" { Text("Simulated").font(.caption) }
          GlobeView(
            source: source, route: route, webBaseURL: registry.configuration.webBaseURL,
            fixture: choice == "fake")
        }
      } else {
        Label("Server unavailable", systemImage: "wifi.exclamationmark")
      }
    }
    .background(HexStyle.background).accessibilityIdentifier("globe.screen")
    .toolbar(.visible, for: .navigationBar)
  }
}

@MainActor struct SettingsScreen: View {
  let registry: PieceRegistry
  @State private var token = ""
  @State private var saved = false
  @State private var saveFailed = false
  @FocusState private var editingToken: Bool
  @State private var showTesting = false
  @AppStorage("hex.testflight.debugEnabled") private var testing = false
  init(registry: PieceRegistry) { self.registry = registry }
  private var testingAvailable: Bool {
    #if DEBUG
      true
    #else
      Bundle.main.appStoreReceiptURL?.lastPathComponent == "sandboxReceipt"
    #endif
  }
  var body: some View {
    Form {
      Section { LiveSplatSettingsRow(preference: registry.liveSplatPreference) }
      Section("Server") {
        Text(registry.configuration.baseURL?.absoluteString ?? "Not configured")
          .font(.footnote).foregroundStyle(HexStyle.muted).textSelection(.enabled)
          .accessibilityIdentifier("settings.server")
        SecureField("Write token", text: $token)
          .focused($editingToken).submitLabel(.done).onSubmit { editingToken = false }
          .textContentType(.password).textInputAutocapitalization(.never).autocorrectionDisabled()
          .accessibilityIdentifier("settings.token")
        Button(saved ? "Saved" : "Sign in") {
          editingToken = false
          do {
            try APICredentials.setWriteToken(token)
            saved = true
            saveFailed = false
            Task { await CaptureUploadQueue.shared.resumeQueued() }
          } catch { saveFailed = true }
        }.frame(minHeight: 44).accessibilityIdentifier("settings.signIn")
          .disabled(token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        if !token.isEmpty {
          Button("Sign out") {
            do {
              try APICredentials.setWriteToken("")
              token = ""
              saved = false
              saveFailed = false
            } catch { saveFailed = true }
          }.frame(minHeight: 44).accessibilityIdentifier("settings.signOut")
        }
        if saveFailed { Label("Sign-in failed", systemImage: "exclamationmark.circle") }
      }
      #if DEBUG
        Section {
          Text("Providers")
            .accessibilityIdentifier("settings.providers")
            .accessibilityValue(
              PieceRegistry.options.map { "\($0.key):\(registry.choice($0.key))" }.joined(
                separator: ","))
        }
      #endif
      if testingAvailable {
        Section {
          Button("Testing", systemImage: "wrench.and.screwdriver") {
            testing = true
            showTesting = true
          }.frame(minHeight: 44).accessibilityIdentifier("settings.testing")
        }
      }
    }
    .scrollContentBackground(.hidden).background(HexStyle.background)
    .navigationTitle("Settings").accessibilityIdentifier("settings.screen")
    .toolbar(.visible, for: .navigationBar)
    .onAppear { token = APICredentials.writeToken }
    .onChange(of: token) { _, _ in saved = false }
    .sheet(isPresented: $showTesting) {
      PieceDebugMenu(registry: registry) { showTesting = false }
    }
  }
}
