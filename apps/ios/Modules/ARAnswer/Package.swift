// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "ARAnswer",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "ARAnswer", targets: ["ARAnswer"])],
  dependencies: [
    .package(path: "../EntitySchema"),
    .package(path: "../Voice"),
    .package(path: "../Router"),
    .package(path: "../Perception"),
  ],
  targets: [
    .target(
      name: "ARAnswer", dependencies: ["EntitySchema", "Router"], resources: [.process("Shaders")]),
    .testTarget(
      name: "ARAnswerTests", dependencies: ["ARAnswer", "Voice", "Router", "Perception"]),
  ]
)
