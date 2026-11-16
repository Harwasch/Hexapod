import SwiftUI

@main
struct HexApp: App {
  var body: some Scene {
    WindowGroup {
      CameraHost()
        .preferredColorScheme(.dark)
    }
  }
}
