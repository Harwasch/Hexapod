// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "EntitySchema",
    platforms: [.iOS("26.0"), .macOS(.v15)],
    products: [.library(name: "EntitySchema", targets: ["EntitySchema"])],
    targets: [
        .target(name: "EntitySchema"),
        .testTarget(name: "EntitySchemaTests", dependencies: ["EntitySchema"], resources: [.copy("Fixtures")])
    ]
)
