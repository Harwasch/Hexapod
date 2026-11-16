// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Globe",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "Globe", targets: ["Globe"])],
  dependencies: [.package(path: "../EntitySchema"), .package(path: "../ARAnswer")],
  targets: [
    .target(name: "Globe", dependencies: ["EntitySchema", "ARAnswer"]),
    .testTarget(name: "GlobeTests", dependencies: ["Globe"]),
  ]
)
