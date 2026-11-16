// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Perception",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "Perception", targets: ["Perception"])],
  dependencies: [.package(path: "../EntitySchema")],
  targets: [
    .target(name: "Perception", dependencies: ["EntitySchema"]),
    .testTarget(name: "PerceptionTests", dependencies: ["Perception"]),
  ]
)
