import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

/// Stage-0 capability check: confirms MLX runs on this Mac and that the ops the
/// LiTo port depends on — **conv3d**, **fast scaled-dot-product attention**, and
/// **safetensors loading** — are all available in this mlx-swift build.
public enum Smoke {
    public static func run(weights: URL?) {
        Device.withDefaultDevice(Device(.gpu)) {
            print("MLX default device: \(Device.defaultDevice())")

            // 1) basic GPU matmul
            let a = MLXRandom.normal([512, 512])
            let b = MLXRandom.normal([512, 512])
            let c = matmul(a, b)
            c.eval()
            print("✓ matmul  \(a.shape)·\(b.shape) -> \(c.shape)  mean=\(c.mean().item(Float.self))")

            // 2) conv3d (MLX uses NDHWC layout) — needed for the TRELLIS voxel VAE (Stage 4)
            let conv = Conv3d(inputChannels: 8, outputChannels: 16, kernelSize: 3, padding: 1)
            let x = MLXRandom.normal([1, 16, 16, 16, 8])
            let y = conv(x)
            y.eval()
            print("✓ conv3d  in \(x.shape) -> out \(y.shape)")

            // 3) fast SDPA — the attention primitive for DINOv2 + DiT + gaussian decoder
            let q = MLXRandom.normal([1, 8, 64, 64])
            let k = MLXRandom.normal([1, 8, 64, 64])
            let v = MLXRandom.normal([1, 8, 64, 64])
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                      scale: 1.0 / 8.0, mask: .none)
            o.eval()
            print("✓ sdpa    q\(q.shape) -> \(o.shape)")

            // 4) load the converted safetensors and confirm key tensors are present/shaped
            if let w = weights {
                do {
                    let arrays = try loadArrays(url: w)
                    let probes = [
                        "velocity_estimator_ema.module.pos_mtx",
                        "patch_encoder.dinov2_model.model.patch_embed.proj.weight",
                        "pretrained_tokenizer.voxel_decoder.net.init_query",
                    ]
                    print("✓ safetensors  \(arrays.count) tensors")
                    for p in probes { print("    \(p) -> \(arrays[p]?.shape.description ?? "MISSING")") }
                } catch {
                    print("✗ safetensors load FAILED: \(error)")
                }
            }
            print("Stage 0 smoke complete.")
        }
    }
}
