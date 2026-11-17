// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Globe",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "Globe", targets: ["Globe"])],
  dependencies: [.package(path: "../EntitySchema")],
  targets: [
    .target(
      name: "Globe", dependencies: ["EntitySchema"], resources: [.copy("Resources/GlobeWeb")]),
    .testTarget(name: "GlobeTests", dependencies: ["Globe"], resources: [.copy("Fixtures")]),
  ]
)
