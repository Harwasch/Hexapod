// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Cloud",
    platforms: [.iOS("26.0"), .macOS(.v13)],
    products: [.library(name: "Cloud", targets: ["Cloud"])],
    dependencies: [.package(path: "../EntitySchema")],
    targets: [
        .target(name: "Cloud", dependencies: ["EntitySchema"]),
        .testTarget(name: "CloudTests", dependencies: ["Cloud", "EntitySchema"]),
    ]
)
