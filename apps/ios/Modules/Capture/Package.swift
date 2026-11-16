// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Capture",
  platforms: [.iOS("26.0"), .macOS(.v13)],
  products: [.library(name: "Capture", targets: ["Capture"])],
  targets: [.target(name: "Capture"), .testTarget(name: "CaptureTests", dependencies: ["Capture"])]
)
