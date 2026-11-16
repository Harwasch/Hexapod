// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Perception",
    platforms: [.iOS("26.0"), .macOS(.v15)],
    products: [.library(name: "Perception", targets: ["Perception"])],
    targets: [
        .target(name: "Perception"),
        .testTarget(name: "PerceptionTests", dependencies: ["Perception"])
    ]
)
