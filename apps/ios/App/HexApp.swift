import SwiftUI

@main
struct HexApp: App {
  var body: some Scene {
    WindowGroup {
      HomeScreen()
        .preferredColorScheme(.dark)
    }
  }
}
