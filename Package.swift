// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LiToStudio",
    platforms: [.macOS(.v15)],
    dependencies: [
        // Apple's MLX, Swift bindings — the runtime for the whole LiTo engine.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        // True gaussian-splat rendering (Metal) for the viewer — pure Swift, on-device.
        // Pinned to main: the SplatChunk/SH3 API postdates the last tagged release (1.0.1).
        .package(url: "https://github.com/scier/MetalSplatter.git",
                 revision: "2b965de1934de38dda1c71cf90bf798aa948a14c"),
    ],
    targets: [
        // The macOS app (SwiftUI), driving the native LiToKit engine in-process.
        .executableTarget(
            name: "LiToStudio",
            dependencies: [
                "LiToKit",
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ],
            path: "Sources/LiToStudio",
            // Info.plist is consumed by the Xcode app target (project.yml), not SwiftPM.
            exclude: ["Info.plist"]
        ),
        // One-time, dev-only weight converter: torch-zip (.ckpt/.pth) → .safetensors.
        // Foundation-only on purpose — no MLX dependency, so it builds/runs fast.
        .executableTarget(
            name: "LiToConvert",
            path: "Sources/LiToConvert"
        ),
        // The native inference engine (DINOv2 → DiT → voxel VAE → gaussian decoder).
        .target(
            name: "LiToKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/LiToKit"
        ),
        // Dev-only smoke/parity harness for LiToKit (not shipped).
        .executableTarget(
            name: "LiToSmoke",
            dependencies: [
                "LiToKit",
                .product(name: "SplatIO", package: "MetalSplatter"),
            ],
            path: "Sources/LiToSmoke"
        ),
    ]
)
