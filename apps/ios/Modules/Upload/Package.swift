// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Upload",
  platforms: [.iOS("26.0"), .macOS(.v13)],
  products: [.library(name: "Upload", targets: ["Upload"])],
  targets: [.target(name: "Upload"), .testTarget(name: "UploadTests", dependencies: ["Upload"])]
)
