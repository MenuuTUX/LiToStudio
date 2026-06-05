import Foundation
import MLX
import MLXFast

/// Generic attention sub-layers shared across the LiTo perceivers/decoders.
/// They mirror `lito.models.layers.{SelfAttentionLayer,CrossAttentionLayer}`:
/// fused qkv (or q + kv), optional RMSNorm on q/k, fast SDPA, output projection.
/// The caller passes a weight scope ending at the layer (e.g. `…sa_layers.0.`).
enum Attn {
    static func selfAttn(_ x: MLXArray, _ w: Weights.Scoped, heads: Int, headDim: Int) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), dQKV = heads * headDim
        let p = Nn.linear(x, w("linear_qkv.weight"), w("linear_qkv.bias")).split(parts: 3, axis: -1)
        let q = Nn.rmsNorm(p[0], w("rmsnorm_q.scale"))
        let k = Nn.rmsNorm(p[1], w("rmsnorm_k.scale"))
        func H(_ a: MLXArray) -> MLXArray { a.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(queries: H(q), keys: H(k), values: H(p[2]),
                                                  scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return Nn.linear(o.transposed(0, 2, 1, 3).reshaped([b, n, dQKV]),
                         w("linear_out.weight"), w("linear_out.bias"))
    }

    static func crossAttn(_ qIn: MLXArray, _ kvIn: MLXArray, _ w: Weights.Scoped,
                          heads: Int, headDim: Int) -> MLXArray {
        let b = qIn.dim(0), n = qIn.dim(1), m = kvIn.dim(1), dQKV = heads * headDim
        let q0 = Nn.layerNorm(qIn, w("layernorm_q.weight"), w("layernorm_q.bias"), eps: 1e-5)
        let kv0 = Nn.layerNorm(kvIn, w("layernorm_kv.weight"), w("layernorm_kv.bias"), eps: 1e-5)
        let q = Nn.rmsNorm(Nn.linear(q0, w("linear_q.weight"), w("linear_q.bias")), w("rmsnorm_q.scale"))
        let kv = Nn.linear(kv0, w("linear_kv.weight"), w("linear_kv.bias")).split(parts: 2, axis: -1)
        let k = Nn.rmsNorm(kv[0], w("rmsnorm_k.scale"))
        func Hq(_ a: MLXArray) -> MLXArray { a.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3) }
        func Hk(_ a: MLXArray) -> MLXArray { a.reshaped([b, m, heads, headDim]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(queries: Hq(q), keys: Hk(k), values: Hk(kv[1]),
                                                  scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return Nn.linear(o.transposed(0, 2, 1, 3).reshaped([b, n, dQKV]),
                         w("linear_out.weight"), w("linear_out.bias"))
    }

    /// timm-style MLP (fc1 → gelu-tanh → fc2), used by the perceiver blocks.
    static func mlp(_ x: MLXArray, _ w: Weights.Scoped) -> MLXArray {
        Nn.linear(Nn.geluTanh(Nn.linear(x, w("fc1.weight"), w("fc1.bias"))), w("fc2.weight"), w("fc2.bias"))
    }
}
