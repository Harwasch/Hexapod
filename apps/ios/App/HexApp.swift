import Globe
import SwiftUI
import UIKit

@main
struct HexApp: App {
  @UIApplicationDelegateAdaptor(UploadLifecycle.self) private var uploadLifecycle
  @Environment(\.scenePhase) private var scenePhase
  var body: some Scene {
    WindowGroup {
      HomeScreen()
        .preferredColorScheme(.dark)
        // Retain WebKit early; the globe chooses HTTPS or an explicit fixture on opening.
        .task {
          _ = CesiumGlobeSession.shared
          if scenePhase == .active { await CaptureUploadQueue.shared.resumeQueued() }
        }
        .onChange(of: scenePhase) { _, phase in
          if phase == .active { Task { await CaptureUploadQueue.shared.resumeQueued() } }
        }
    }
  }
}

@MainActor final class UploadLifecycle: NSObject, UIApplicationDelegate {
  func application(
    _ application: UIApplication,
    handleEventsForBackgroundURLSession identifier: String,
    completionHandler: @escaping () -> Void
  ) {
    let completion = UploadCompletion(completionHandler)
    Task {
      await CaptureUploadQueue.shared.reconnect(identifier: identifier) { completion.finish() }
    }
  }
}

@MainActor private final class UploadCompletion {
  private let handler: () -> Void
  init(_ handler: @escaping () -> Void) { self.handler = handler }
  func finish() { handler() }
}
