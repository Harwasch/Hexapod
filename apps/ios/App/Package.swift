// swift-tools-version: 6.0
#if canImport(PackageDescription)
  import PackageDescription
  let package = Package(
    name: "AppSupport",
    platforms: [.iOS("26.0"), .macOS(.v15)],
    products: [.library(name: "AppSupport", targets: ["AppSupport"])],
    dependencies: [
      .package(path: "../Modules/Capture"), .package(path: "../Modules/Voice"),
      .package(path: "../Modules/Router"), .package(path: "../Modules/Perception"),
      .package(path: "../Modules/ARAnswer"), .package(path: "../Modules/Cloud"),
      .package(path: "../Modules/Globe"), .package(path: "../Modules/LiveSplat"),
      .package(path: "../Modules/Upload"),
    ],
    targets: [
      .target(
        name: "AppSupport",
        dependencies: [
          "Capture", "Voice", "Router", "Perception", "ARAnswer", "Cloud", "Globe", "LiveSplat",
          "Upload",
        ],
        path: "Support"),
      .testTarget(name: "AppSupportTests", dependencies: ["AppSupport"], path: "Tests"),
    ])
#endif
