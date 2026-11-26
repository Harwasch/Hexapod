// swift-tools-version: 6.0
import PackageDescription

var dependencies: [Package.Dependency] = []
var trainerDependencies: [Target.Dependency] = ["PhoneTrainPrep"]
// Binary XCFrameworks cannot run on Linux. Keep its tests entirely local.
#if os(macOS)
  dependencies.append(.package(url: "https://github.com/Plinth-it/msplat-ios", exact: "3.0.0"))
  trainerDependencies.append(.product(name: "Msplat", package: "msplat-ios"))
#endif

let package = Package(
  name: "PhoneTrain",
  platforms: [.iOS("26.0"), .macOS(.v15)],
  products: [
    .library(name: "PhoneTrainPrep", targets: ["PhoneTrainPrep"]),
    .library(name: "PhoneTrain", targets: ["PhoneTrain"]),
  ],
  dependencies: dependencies,
  targets: [
    .target(name: "PhoneTrainPrep"),
    .target(name: "PhoneTrain", dependencies: trainerDependencies),
    .testTarget(name: "PhoneTrainPrepTests", dependencies: ["PhoneTrainPrep", "PhoneTrain"]),
  ]
)
