// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Router",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "Router", targets: ["Router"])],
  dependencies: [
    .package(path: "../EntitySchema"), .package(path: "../Voice"), .package(path: "../Perception"),
  ],
  targets: [
    .target(name: "Router", dependencies: ["EntitySchema"]),
    .testTarget(name: "RouterTests", dependencies: ["Router", "Voice", "Perception"]),
  ]
)
