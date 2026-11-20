// swift-tools-version:5.10
// The Syrinx app for the Mac: runs syrinx-server --standalone, downloads models, says how
// other apps connect and shows the statistics. build-app.sh makes Syrinx.app from it.
import PackageDescription

let package = Package(
    name: "SyrinxManager",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "SyrinxManager",
            path: "Sources/SyrinxManager"
        ),
        .testTarget(
            name: "SyrinxManagerTests",
            dependencies: ["SyrinxManager"],
            path: "Tests/SyrinxManagerTests"
        ),
    ]
)
