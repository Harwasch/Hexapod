import SwiftUI

@main
struct HexApp: App {
  var body: some Scene {
    WindowGroup {
      CameraScreen()
        .preferredColorScheme(.dark)
    }
  }
}
