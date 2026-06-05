import Foundation
import MLX
import MLXNN

/// Stage 4a — the voxel decoder (`SSLatentDecoder` + its `VectorDecoder` perceiver),
/// a port of `lito.models.point_decoder.SSLatentDecoder`. Maps the shape latent
/// (b, 8192, 32) → a 16³ "sparse-structure latent" (b, 16, 16, 16, 8) in NDHWC.
///
/// 16³ learned queries (+ zyx Fourier positional encoding) cross-attend the latent,
/// 4 perceiver blocks (cross-attn → mlp → 2×[self-attn, mlp]), then a final LN+Linear.
public struct VoxelDecoder {
    let w: Weights.Scoped
    static let R = 16, dim = 512, heads = 8, headDim = 128, blocks = 4, numSA = 2

    public init(_ weights: Weights) { self.w = weights.prefixed("pretrained_tokenizer.voxel_decoder.") }

    /// (b, 8192, 32) → ss-latent (b, 16, 16, 16, 8) NDHWC. (b = 1.)
    public func callAsFunction(latent: MLXArray) -> MLXArray {
        let R = VoxelDecoder.R, net = w.prefixed("net.")
        let kv = Nn.linear(latent, w("input_linear.weight"), w("input_linear.bias"))   // (1,8192,512)

        // learned init queries + zyx Fourier positional encoding over the 16³ grid
        let coords = gridCoords()                                                       // (16,16,16,3) in [-1,1]
        let freq = net("zyx_pos_encoder.freq_bands")                                    // (128,)
        let pe = coords.expandedDimensions(axis: -1) * freq                             // (16,16,16,3,128)
        let nf = freq.dim(0) * 3
        let posEnc = concatenated([coords, sin(pe).reshaped([R, R, R, nf]), cos(pe).reshaped([R, R, R, nf])], axis: -1)
        var lat = (net("init_query") + Nn.linear(posEnc, net("init_query_linear.weight"), net("init_query_linear.bias")))
            .reshaped([1, R * R * R, VoxelDecoder.dim])                                 // (1,4096,512)

        // perceiver blocks (keep_block_bug=False → latents carry over)
        let h = VoxelDecoder.heads, hd = VoxelDecoder.headDim
        for i in 0 ..< VoxelDecoder.blocks {
            let bl = net.prefixed("encoder.blocks.\(i).")
            lat = lat + Attn.crossAttn(lat, kv, bl.prefixed("ca_layer."), heads: h, headDim: hd)
            lat = lat + Attn.mlp(Nn.layerNorm(lat, bl("ca_ln.weight"), bl("ca_ln.bias"), eps: 1e-6), bl.prefixed("ca_mlp."))
            for s in 0 ..< VoxelDecoder.numSA {
                lat = lat + Attn.selfAttn(Nn.layerNorm(lat, bl("ln1_layers.\(s).weight"), bl("ln1_layers.\(s).bias"), eps: 1e-6),
                                          bl.prefixed("sa_layers.\(s)."), heads: h, headDim: hd)
                lat = lat + Attn.mlp(Nn.layerNorm(lat, bl("ln2_layers.\(s).weight"), bl("ln2_layers.\(s).bias"), eps: 1e-6),
                                     bl.prefixed("mlp_layers.\(s)."))
            }
        }

        // (1,4096,512) → (1,16,16,16,512); final LN + Linear → (1,16,16,16,8)
        var out = lat.reshaped([1, R, R, R, VoxelDecoder.dim])
        out = Nn.layerNorm(out, w("final_layer.norm_final.weight"), w("final_layer.norm_final.bias"), eps: 1e-6)
        return Nn.linear(out, w("final_layer.linear.weight"), w("final_layer.linear.bias"))   // (1,16,16,16,8) NDHWC
    }

    /// The 16³ voxel-center grid in [-1,1], meshgrid order (z, y, x) → (16,16,16,3).
    private func gridCoords() -> MLXArray {
        let R = VoxelDecoder.R
        var vals = [Float](repeating: 0, count: R)
        for i in 0 ..< R { vals[i] = (Float(i) + 0.5) * (2.0 / Float(R)) - 1.0 }
        let a = MLXArray(vals)                                                            // (16,)
        let one = MLXArray.ones([R, R, R])
        let qz = a.reshaped([R, 1, 1]) * one, hy = a.reshaped([1, R, 1]) * one, wx = a.reshaped([1, 1, R]) * one
        return stacked([qz, hy, wx], axis: -1)                                            // (16,16,16,3)
    }
}
