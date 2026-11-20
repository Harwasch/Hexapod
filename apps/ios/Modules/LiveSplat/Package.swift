// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "LiveSplat",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "LiveSplat", targets: ["LiveSplat"])],
  targets: [
    .target(name: "LiveSplat", resources: [.process("Shaders")]),
    .testTarget(name: "LiveSplatTests", dependencies: ["LiveSplat"]),
  ]
)
