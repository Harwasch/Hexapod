// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ARAnswer",
    platforms: [.iOS("26.0"), .macOS(.v15)],
    products: [.library(name: "ARAnswer", targets: ["ARAnswer"])],
    targets: [
        .target(name: "ARAnswer"),
        .testTarget(name: "ARAnswerTests", dependencies: ["ARAnswer"])
    ]
)
