// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Voice",
    platforms: [.iOS("26.0"), .macOS(.v15)],
    products: [.library(name: "Voice", targets: ["Voice"])],
    targets: [
        .target(name: "Voice"),
        .testTarget(name: "VoiceTests", dependencies: ["Voice"])
    ]
)
