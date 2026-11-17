import Globe
import SwiftUI

@main
struct HexApp: App {
  var body: some Scene {
    WindowGroup {
      TabView {
        CameraScreen().tabItem { Label("Camera", systemImage: "camera") }
        NavigationStack {
          GlobeView(
            source: APIGlobeDataSource(baseURL: URL(string: "https://twin-api.fly.dev/api/v1")!),
            webBaseURL: URL(string: "https://twin-web-f57.pages.dev")
          )
          #if DEBUG
            .toolbar { NavigationLink("Cesium fixture") { GlobeDebugView() } }
          #endif
        }.tabItem { Label("Globe", systemImage: "globe") }
      }
      .preferredColorScheme(.dark)
      .task { _ = CesiumGlobeSession.shared }
    }
  }
}
