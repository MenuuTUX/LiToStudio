import Foundation
import MLX
import MLXFast

/// Stage 2 — the LiTo image-conditioning encoder, a port of
/// `src/lito/models/dino.py::SpatialDinov2` (the `dinov2_vitl14_reg_rgba` variant).
///
/// Two branches, concatenated feature-wise into 2048-dim cond tokens:
///   • a frozen **DINOv2 ViT-L/14 with registers** over the normalized premult-RGB
///     (layer −1, NO final norm, then a parameterless LayerNorm over the concat), and
///   • a learnable **4-channel Conv2d** (normalized premult-RGB + raw alpha) producing
///     1369 patch tokens, prepended with `learnable_paddings` for the cls+register slots.
///
/// Token order is `[cls, reg×4, patch×1369]` → 1374 tokens. Input is `cond_rgba`
/// (b, 518, 518, 4) in [0,1]; output is `(b, 1374, 2048)`.
public struct Dinov2Encoder {
    let dino: Weights.Scoped          // patch_encoder.dinov2_model.model.*
    let pe: Weights.Scoped            // patch_encoder.*

    static let D = 1024, heads = 16, headDim = 64, depth = 24, patch = 14
    static let scale: Float = 1.0 / 8.0          // 1/sqrt(head_dim=64)

    public init(_ w: Weights) {
        self.dino = w.prefixed("patch_encoder.dinov2_model.model.")
        self.pe = w.prefixed("patch_encoder.")
    }

    public func callAsFunction(condRGBA: MLXArray) -> MLXArray {
        let D = Dinov2Encoder.D
        var x4 = condRGBA
        if x4.ndim == 3 { x4 = x4.expandedDimensions(axis: 0) }   // (b,H,W,4)
        let b = x4.dim(0), H = x4.dim(1), W = x4.dim(2)

        let rgb = x4[0 ..< b, 0 ..< H, 0 ..< W, 0 ..< 3]
        let alpha = x4[0 ..< b, 0 ..< H, 0 ..< W, 3 ..< 4]
        let premult = rgb * alpha

        // ImageNet normalization (shared by both branches' RGB input)
        let mean = MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped([1, 1, 1, 3])
        let std = MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped([1, 1, 1, 3])
        // compute in the weights' dtype (fp16 in half-precision mode)
        let wdt = dino("cls_token").dtype
        let dinoIn = ((premult - mean) / std).asType(wdt)        // (b,H,W,3) NHWC

        // ── DINOv2 backbone ────────────────────────────────────────────────
        // patch embed: torch conv weight [O,I,kH,kW] → MLX [O,kH,kW,I]
        let peW = dino("patch_embed.proj.weight").transposed(0, 2, 3, 1)
        var emb = conv2d(dinoIn, peW, stride: 14, padding: 0)     // (b,37,37,1024)
        emb = emb + dino("patch_embed.proj.bias")
        let gh = emb.dim(1), gw = emb.dim(2), nPatch = gh * gw
        let patches = emb.reshaped([b, nPatch, D])               // (b,1369,1024)

        // cls + interpolated pos-embed (identity at the 518² training resolution).
        // DINO runs at batch=1 (cls/reg/pad keep their leading dim 1), so no broadcast needed.
        let cls = dino("cls_token")                              // (1,1,1024)
        var seq = concatenated([cls, patches], axis: 1)          // (b,1370,1024)
        seq = seq + dino("pos_embed")
        // insert register tokens after cls (they receive no positional embedding)
        let reg = dino("register_tokens")                        // (1,4,1024)
        let n0 = seq.dim(1)
        seq = concatenated([seq[0 ..< b, 0 ..< 1, 0 ..< D], reg, seq[0 ..< b, 1 ..< n0, 0 ..< D]], axis: 1)

        // 24 transformer blocks (pre-norm + layerscale)
        for i in 0 ..< Dinov2Encoder.depth {
            let bl = dino.prefixed("blocks.\(i).")
            let a = attention(Nn.layerNorm(seq, bl("norm1.weight"), bl("norm1.bias"), eps: 1e-6), bl)
            seq = seq + a * bl("ls1.gamma")
            var m = Nn.layerNorm(seq, bl("norm2.weight"), bl("norm2.bias"), eps: 1e-6)
            m = Nn.linear(m, bl("mlp.fc1.weight"), bl("mlp.fc1.bias"))
            m = Nn.gelu(m)
            m = Nn.linear(m, bl("mlp.fc2.weight"), bl("mlp.fc2.bias"))
            seq = seq + m * bl("ls2.gamma")
        }
        // layer −1, no final norm; tokens already ordered [cls, reg, patch]
        // parameterless LayerNorm over the concat (F.layer_norm default eps 1e-5)
        let dinoFeat = Nn.layerNorm(seq, eps: 1e-5)              // (b,1374,1024)

        // ── learnable 4-channel conv branch ────────────────────────────────
        let learnIn = concatenated([dinoIn, alpha.asType(wdt)], axis: -1)   // (b,H,W,4) [rgb_norm, alpha]
        let lW = pe("learnable_model.weight").transposed(0, 2, 3, 1)   // (1024,14,14,4)
        var lt = conv2d(learnIn, lW, stride: 14, padding: 0)     // (b,37,37,1024)
        lt = lt + pe("learnable_model.bias")
        let learnPatch = lt.reshaped([b, nPatch, D])             // (b,1369,1024)
        let pad = pe("learnable_paddings").reshaped([1, 5, D])   // (1,5,1024), batch=1
        let learnTok = concatenated([pad, learnPatch], axis: 1)  // (b,1374,1024)

        return concatenated([dinoFeat, learnTok], axis: -1)      // (b,1374,2048)
    }

    private func attention(_ x: MLXArray, _ bl: Weights.Scoped) -> MLXArray {
        let b = x.dim(0), N = x.dim(1)
        let h = Dinov2Encoder.heads, hd = Dinov2Encoder.headDim, D = Dinov2Encoder.D
        let qkv = Nn.linear(x, bl("attn.qkv.weight"), bl("attn.qkv.bias"))   // (b,N,3D)
        let parts = qkv.split(parts: 3, axis: -1)                            // [q,k,v] (b,N,D)
        func toHeads(_ a: MLXArray) -> MLXArray { a.reshaped([b, N, h, hd]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(
            queries: toHeads(parts[0]), keys: toHeads(parts[1]), values: toHeads(parts[2]),
            scale: Dinov2Encoder.scale, mask: .none)            // (b,h,N,hd)
        let merged = o.transposed(0, 2, 1, 3).reshaped([b, N, D])
        return Nn.linear(merged, bl("attn.proj.weight"), bl("attn.proj.bias"))
    }
}
