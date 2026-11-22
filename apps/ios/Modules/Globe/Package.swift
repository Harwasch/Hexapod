// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Globe",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [.library(name: "Globe", targets: ["Globe"])],
  dependencies: [.package(path: "../Networking"), .package(path: "../EntitySchema"), .package(path: "../ARAnswer")],
  targets: [
    .target(
      name: "Globe", dependencies: ["Networking", "EntitySchema", "ARAnswer"],
      resources: [.copy("Resources/GlobeWeb")]),
    .testTarget(name: "GlobeTests", dependencies: ["Globe"], resources: [.copy("Fixtures")]),
  ]
)
