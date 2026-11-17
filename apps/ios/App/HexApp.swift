import Globe
import SwiftUI

@main
struct HexApp: App {
  var body: some Scene {
    WindowGroup {
      HomeScreen()
        .preferredColorScheme(.dark)
        // Warm the bundled Cesium globe so opening it is quick.
        .task { _ = CesiumGlobeSession.shared }
    }
  }
}
