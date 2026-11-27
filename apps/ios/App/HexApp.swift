import Globe
import SwiftUI
import UIKit

@main
struct HexApp: App {
  @UIApplicationDelegateAdaptor(UploadLifecycle.self) private var uploadLifecycle
  @Environment(\.scenePhase) private var scenePhase
  init() {
    if LaunchMode.current.isUITest {
      // UIKit needs its presentation animations to open the globe's Menu.
      // The SwiftUI transaction below still quiets fixture view transitions.
      do { try LaunchMode.current.prepareCaptureStorage() } catch {
        assertionFailure("Could not reset UI-test capture storage")
      }
    }
  }

  var body: some Scene {
    WindowGroup {
      HomeScreen()
        .preferredColorScheme(.dark)
        .transaction { transaction in
          if LaunchMode.current.isUITest {
            transaction.animation = nil
            transaction.disablesAnimations = true
          }
        }
        // Retain WebKit early on normal launches; fixture launches wait for the globe.
        .task {
          if !LaunchMode.current.isUITest { _ = CesiumGlobeSession.shared }
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
