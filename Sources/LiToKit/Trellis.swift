import Foundation
import MLX
import MLXNN

/// Stage 4b — the TRELLIS `SparseStructureDecoder` (conv3d), a port of
/// `third_party/TRELLIS/.../sparse_structure_vae.py`. Upsamples the 16³ ss-latent
/// (8 ch) to a 64³ occupancy-logit grid: input conv 8→512, 2 mid ResBlocks,
/// `[RB512, RB512, Up→128, RB128, RB128, Up→32, RB32, RB32]`, out norm+silu+conv→1.
/// Everything in NDHWC; ChannelLayerNorm = LayerNorm over the (last) channel axis.
public struct TrellisDecoder {
    let w: Weights
    public init(_ weights: Weights) { self.w = weights }

    private func conv3d(_ x: MLXArray, _ name: String) -> MLXArray {
        let weight = w(name + ".weight", as: .float32).transposed(0, 2, 3, 4, 1)   // OIDHW → ODHWI
        return MLX.conv3d(x, weight, stride: 1, padding: 1) + w(name + ".bias", as: .float32)
    }
    private func cln(_ x: MLXArray, _ name: String) -> MLXArray {
        Nn.layerNorm(x, w(name + ".weight", as: .float32), w(name + ".bias", as: .float32), eps: 1e-5)
    }
    private func resBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = conv3d(silu(cln(x, p + ".norm1")), p + ".conv1")
        h = conv3d(silu(cln(h, p + ".norm2")), p + ".conv2")
        return h + x                                          // skip = Identity (channels unchanged)
    }
    private func pixelShuffle3d(_ x: MLXArray, _ s: Int) -> MLXArray {
        let b = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), co = x.dim(4) / (s * s * s)
        return x.reshaped([b, D, H, W, co, s, s, s])
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped([b, D * s, H * s, W * s, co])
    }

    /// ss-latent (b,16,16,16,8) NDHWC → occupancy logits (b,64,64,64,1).
    public func callAsFunction(ssLatent: MLXArray) -> MLXArray {
        var h = conv3d(ssLatent, "input_layer")              // (b,16,16,16,512)
        for i in 0 ..< 2 { h = resBlock(h, "middle_block.\(i)") }
        let upsampleAt: Set<Int> = [2, 5]
        for i in 0 ..< 8 {
            h = upsampleAt.contains(i) ? pixelShuffle3d(conv3d(h, "blocks.\(i).conv"), 2)
                                       : resBlock(h, "blocks.\(i)")
        }
        return conv3d(silu(cln(h, "out_layer.0")), "out_layer.2")   // (b,64,64,64,1)
    }

    /// Occupied voxel centers in [-1,1] from logits ≥ `threshold` (default 0 = sigmoid ≥ 0.5,
    /// the reference), in torch `nonzero` order over the permuted (x,y,z) grid. Raising the
    /// threshold prunes low-confidence "ghost" voxels.
    ///
    /// `minComponentFraction` > 0 additionally drops disconnected occupancy islands smaller
    /// than that fraction of the largest connected component (6-connectivity) — detached
    /// floater blobs the threshold alone lets through. 0 disables.
    public func initCoords(logit: MLXArray, threshold: Float = 0,
                           minComponentFraction: Float = 0) -> (coords: MLXArray, count: Int) {
        let R = logit.dim(1)
        let flat = logit.reshaped([R * R * R]).asType(.float32).asArray(Float.self)   // idx = z*R² + y*R + x
        var occ = [Bool](repeating: false, count: R * R * R)
        for i in occ.indices { occ[i] = flat[i] >= threshold }
        if minComponentFraction > 0 { pruneComponents(&occ, R, minComponentFraction) }

        let cw = Float(2.0) / Float(R)
        var out = [Float](); out.reserveCapacity(8192 * 3); var n = 0
        for x in 0 ..< R { for y in 0 ..< R { for z in 0 ..< R {
            if occ[z * R * R + y * R + x] {
                out.append((Float(x) + 0.5) * cw - 1)
                out.append((Float(y) + 0.5) * cw - 1)
                out.append((Float(z) + 0.5) * cw - 1)
                n += 1
            }
        }}}
        return (MLXArray(out).reshaped([n, 3]), n)
    }

    /// Label 6-connected components in the occupancy grid and clear all components smaller
    /// than `fraction` of the largest one.
    private func pruneComponents(_ occ: inout [Bool], _ R: Int, _ fraction: Float) {
        var label = [Int32](repeating: -1, count: occ.count)
        var sizes = [Int]()
        var stack = [Int]()
        for start in occ.indices where occ[start] && label[start] < 0 {
            let id = Int32(sizes.count); var size = 0
            stack.removeAll(keepingCapacity: true); stack.append(start); label[start] = id
            while let idx = stack.popLast() {
                size += 1
                let z = idx / (R * R), y = (idx / R) % R, x = idx % R
                func visit(_ nx: Int, _ ny: Int, _ nz: Int) {
                    guard nx >= 0, nx < R, ny >= 0, ny < R, nz >= 0, nz < R else { return }
                    let ni = nz * R * R + ny * R + nx
                    if occ[ni] && label[ni] < 0 { label[ni] = id; stack.append(ni) }
                }
                visit(x - 1, y, z); visit(x + 1, y, z)
                visit(x, y - 1, z); visit(x, y + 1, z)
                visit(x, y, z - 1); visit(x, y, z + 1)
            }
            sizes.append(size)
        }
        guard let largest = sizes.max(), largest > 0 else { return }
        let minKeep = Int(Float(largest) * fraction)
        for i in occ.indices where occ[i] && sizes[Int(label[i])] < minKeep { occ[i] = false }
    }
}
