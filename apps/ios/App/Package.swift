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
      .package(path: "../Modules/ARAnswer"),
    ],
    targets: [
      .target(
        name: "AppSupport", dependencies: ["Capture", "Voice", "Router", "Perception", "ARAnswer"],
        path: "Support"),
      .testTarget(name: "AppSupportTests", dependencies: ["AppSupport"], path: "Tests"),
    ])
#endif
