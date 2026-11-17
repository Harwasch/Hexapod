import Foundation
import Globe
import LiveSplat
import SwiftUI

@MainActor struct HomeScreen: View {
  @State private var registry = PieceRegistry()
  var body: some View {
    NavigationStack {
      List {
        NavigationLink("Scan", systemImage: "viewfinder") { CameraHost(registry: registry) }
        NavigationLink("Captures", systemImage: "square.stack") {
          CapturesScreen(registry: registry)
        }
        NavigationLink("Globe", systemImage: "globe.americas") {
          PropertyScreen(registry: registry)
        }
        NavigationLink("Settings", systemImage: "gearshape") {
          Form { LiveSplatSettingsRow(preference: registry.liveSplatPreference) }
            .navigationTitle("Settings")
        }
      }.navigationTitle("Hex")
    }
  }
}

@MainActor private struct CapturesScreen: View {
  let registry: PieceRegistry
  @State private var captures: [LocalCapture] = []
  @State private var failed = false
  var body: some View {
    List {
      NavigationLink("Globe", systemImage: "globe.americas") { PropertyScreen(registry: registry) }
      if failed { Image(systemName: "exclamationmark.circle").accessibilityLabel("Unavailable") }
      ForEach(captures) { capture in
        NavigationLink {
          VStack {
            Text(capture.status)
            if let site = capture.upload?.siteID {
              PropertyScreen(
                registry: registry,
                route: GlobeRoute(
                  siteID: site, scanID: capture.upload?.captureID.flatMap(UUID.init(uuidString:))))
            } else {
              PropertyScreen(registry: registry)
            }
          }
        } label: {
          HStack {
            Text(capture.upload?.name ?? "Capture").lineLimit(1)
            Spacer()
            Text(capture.status).font(.caption)
          }
        }
      }
    }.navigationTitle("Captures")
      .task {
        do {
          let root = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Captures")
          captures = try LocalCapture.list(at: root)
        } catch { failed = true }
      }
  }
}

@MainActor private struct PropertyScreen: View {
  let registry: PieceRegistry
  var route: GlobeRoute?
  @AppStorage("hex.testflight.debugEnabled") private var testing = false
  private var useTestChoices: Bool {
    #if DEBUG
      true
    #else
      PieceRegistry.menuAvailable(
        debug: false, receipt: Bundle.main.appStoreReceiptURL?.lastPathComponent, enabled: testing)
    #endif
  }
  var body: some View {
    let choice =
      useTestChoices
      ? registry.choice("globe") : (registry.configuration.apiRoot == nil ? "fake" : "real")
    if let source = try? registry.globeSources.resolve(choice) {
      VStack(spacing: 0) {
        if choice == "fake" { Text("Simulated").font(.caption) }
        GlobeView(source: source, route: route, webBaseURL: registry.configuration.webBaseURL)
      }
    } else {
      Image(systemName: "exclamationmark.circle").accessibilityLabel("Unavailable")
    }
  }
}
