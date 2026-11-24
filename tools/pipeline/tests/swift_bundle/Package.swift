// swift-tools-version: 6.0
import Foundation
import PackageDescription

// pytest stages this package beside an exact copy of the App's BackgroundWriter.swift.
// SwiftPM targets must stay inside their package; no source symlinks are needed.
let repository = ProcessInfo.processInfo.environment["HEXAPOD_REPO_ROOT"]!
let package = Package(
  name: "CaptureBundleIntegration",
  dependencies: [.package(path: repository + "/apps/ios/Modules/Capture")],
  targets: [
    .target(name: "RecorderSupport", dependencies: ["Capture"]),
    .testTarget(name: "RecorderSupportTests", dependencies: ["RecorderSupport", "Capture"]),
  ])
