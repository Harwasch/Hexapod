// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Upload",
  platforms: [.iOS("26.0"), .macOS(.v13)],
  products: [.library(name: "Upload", targets: ["Upload"])],
  dependencies: [.package(path: "../Capture")],
  targets: [
    .target(name: "Upload", dependencies: ["Capture"]),
    .testTarget(name: "UploadTests", dependencies: ["Upload", "Capture"]),
  ]
)
