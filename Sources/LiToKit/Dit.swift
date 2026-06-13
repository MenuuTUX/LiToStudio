import Foundation
import MLX
import MLXNN
import MLXFast

/// Stage 3 — the flow-matching DiT (velocity model), a port of the MLX-Python
/// `DiffusionTransformer` (`src/lito/mlx/models/dit.py`) using the EMA weights.
///
/// How multiple conditioning views are combined during sampling (see `DiT.sample`).
public enum MultiViewMode: String, CaseIterable, Sendable {
    case multidiffusion, stochastic, concat
}

/// 28 PixArt-style adaLN blocks (self-attn + cross-attn + SwiGLU MLP), RMSNorm on
/// q/k, FourierEmbed timestep conditioning. Outputs the velocity dx/dt directly
/// (linear flow path), integrated by Heun in `sample`.
public struct DiT {
    let w: Weights.Scoped
    static let dH = 1152, heads = 16, hd = 72, blocks = 28
    static let attnScale = Float(1.0 / 72.0.squareRoot())

    public init(_ weights: Weights) { self.w = weights.prefixed("velocity_estimator_ema.module.") }

    /// Velocity forward. `tokens` (b, 8192, 32), `t` (b,), `cond` (b, m, 2048) → (b, 8192, 32).
    /// Computes in the weights' dtype (fp16 in half-precision mode) — inputs are cast at entry.
    public func callAsFunction(tokens xIn: MLXArray, t tIn: MLXArray, cond condIn: MLXArray) -> MLXArray {
        let wdt = w("z_proj.weight").dtype
        let x = xIn.asType(wdt), t = tIn.asType(wdt), cond = condIn.asType(wdt)
        let b = x.dim(0)

        // timestep embedding: FourierEmbed → t_proj(Linear,SiLU,Linear); t0 = t0_proj(SiLU,Linear)
        let freq = w("t_embedder.freq_bands")                       // (32,)
        let pe = t.reshaped([b, 1, 1]) * freq                       // (b,1,32)
        var temb = concatenated([sin(pe).reshaped([b, 32]), cos(pe).reshaped([b, 32])], axis: -1)  // (b,64)
        temb = silu(Nn.linear(temb, w("t_proj.0.weight"), w("t_proj.0.bias")))
        temb = Nn.linear(temb, w("t_proj.2.weight"), w("t_proj.2.bias"))           // (b,1152)
        var t0 = silu(temb)
        t0 = Nn.linear(t0, w("t0_proj.1.weight"), w("t0_proj.1.bias"))             // (b,6912)

        // conditioning embedder (Mlp with gelu-tanh)
        var y = Nn.linear(cond, w("cond_embedder.y_proj.fc1.weight"), w("cond_embedder.y_proj.fc1.bias"))
        y = Nn.geluTanh(y)
        y = Nn.linear(y, w("cond_embedder.y_proj.fc2.weight"), w("cond_embedder.y_proj.fc2.bias"))  // (b,m,1152)

        // latent projection + learned positional embedding
        var lat = Nn.linear(x, w("z_proj.weight"), w("z_proj.bias"))               // (b,8192,1152)
        lat = Nn.layerNorm(lat, w("z_proj_ln.weight"), w("z_proj_ln.bias"), eps: 1e-6)
        let pos = Nn.linear(w("pos_mtx"), w("pos_proj.weight"), w("pos_proj.bias"))  // (8192,1152)
        lat = lat + pos.expandedDimensions(axis: 0)

        for i in 0 ..< DiT.blocks { lat = block(i, lat, y, t0) }

        // final layer with adaLN from temb
        var ada = silu(Nn.linear(temb, w("final_layer.adaLN_modulation.0.weight"), w("final_layer.adaLN_modulation.0.bias")))
        ada = Nn.linear(ada, w("final_layer.adaLN_modulation.2.weight"), w("final_layer.adaLN_modulation.2.bias"))  // (b,2304)
        let sh = ada.split(parts: 2, axis: -1)
        var fx = Nn.layerNorm(lat, eps: 1e-6)                                       // affine=False
        fx = fx * (1 + sh[1].expandedDimensions(axis: 1)) + sh[0].expandedDimensions(axis: 1)
        return Nn.linear(fx, w("final_layer.linear.weight"), w("final_layer.linear.bias"))   // (b,8192,32)
    }

    private func block(_ i: Int, _ x0: MLXArray, _ y: MLXArray, _ t0: MLXArray) -> MLXArray {
        let bl = w.prefixed("blocks.\(i).")
        let b = x0.dim(0), dH = DiT.dH
        let mod = bl("scale_shift_table").expandedDimensions(axis: 0) + t0.reshaped([b, 6, dH])  // (b,6,1152)
        func part(_ k: Int) -> MLXArray { mod[0 ..< b, k ..< (k + 1), 0 ..< dH] }   // (b,1,1152)

        var x = x0
        // self-attention with adaLN modulation
        var h = Nn.layerNorm(x, eps: 1e-6)                       // norm1 (affine=False)
        h = h * (1 + part(1)) + part(0)                          // pixart_modulate(shift0, scale1)
        x = x + part(2) * selfAttn(h, bl)                        // gate2
        // cross-attention (post-norm)
        x = x + crossAttn(x, y, bl)
        x = Nn.layerNorm(x, eps: 1e-6)                           // norm2 (affine=False)
        // SwiGLU MLP with adaLN modulation
        x = x + part(5) * swiglu(x * (1 + part(4)) + part(3), bl)  // gate5, modulate(shift3,scale4)
        return x
    }

    private func selfAttn(_ x: MLXArray, _ bl: Weights.Scoped) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        let qkv = Nn.linear(x, bl("attn.linear_qkv.weight"), bl("attn.linear_qkv.bias"))  // (b,n,3456)
        let p = qkv.split(parts: 3, axis: -1)
        let q = Nn.rmsNorm(p[0], bl("attn.rmsnorm_q.scale"))
        let k = Nn.rmsNorm(p[1], bl("attn.rmsnorm_k.scale"))
        func H(_ a: MLXArray) -> MLXArray { a.reshaped([b, n, DiT.heads, DiT.hd]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(queries: H(q), keys: H(k), values: H(p[2]),
                                                  scale: DiT.attnScale, mask: .none)
        return Nn.linear(o.transposed(0, 2, 1, 3).reshaped([b, n, DiT.dH]),
                         bl("attn.linear_out.weight"), bl("attn.linear_out.bias"))
    }

    private func crossAttn(_ x: MLXArray, _ y: MLXArray, _ bl: Weights.Scoped) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), m = y.dim(1)
        let q0 = Nn.layerNorm(x, bl("cross_attn.layernorm_q.weight"), bl("cross_attn.layernorm_q.bias"), eps: 1e-5)
        let kv0 = Nn.layerNorm(y, bl("cross_attn.layernorm_kv.weight"), bl("cross_attn.layernorm_kv.bias"), eps: 1e-5)
        let q = Nn.rmsNorm(Nn.linear(q0, bl("cross_attn.linear_q.weight"), bl("cross_attn.linear_q.bias")),
                           bl("cross_attn.rmsnorm_q.scale"))
        let kv = Nn.linear(kv0, bl("cross_attn.linear_kv.weight"), bl("cross_attn.linear_kv.bias")).split(parts: 2, axis: -1)
        let k = Nn.rmsNorm(kv[0], bl("cross_attn.rmsnorm_k.scale"))
        func Hq(_ a: MLXArray) -> MLXArray { a.reshaped([b, n, DiT.heads, DiT.hd]).transposed(0, 2, 1, 3) }
        func Hk(_ a: MLXArray) -> MLXArray { a.reshaped([b, m, DiT.heads, DiT.hd]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(queries: Hq(q), keys: Hk(k), values: Hk(kv[1]),
                                                  scale: DiT.attnScale, mask: .none)
        return Nn.linear(o.transposed(0, 2, 1, 3).reshaped([b, n, DiT.dH]),
                         bl("cross_attn.linear_out.weight"), bl("cross_attn.linear_out.bias"))
    }

    private func swiglu(_ x: MLXArray, _ bl: Weights.Scoped) -> MLXArray {
        let g = silu(Nn.linear(x, bl("mlp.w1.weight"))) * Nn.linear(x, bl("mlp.w3.weight"))
        return Nn.linear(g, bl("mlp.w2.weight"), bl("mlp.w2.bias"))
    }

    /// Heun ODE integration. `ts` has N points (N−1 steps); the model is the velocity field.
    ///
    /// `cfgScale > 1` enables classifier-free guidance: each step evaluates a conditional
    /// and an unconditional velocity and returns `uncond + cfg·(cond − uncond)`. The
    /// unconditional branch mirrors the reference `ConditionEmbedder.token_drop` — every
    /// cond-token position is replaced by the learned `cond_embedder.y_embedding`. Without
    /// this the shape stays a mushy blob that ignores the image; `cfg≈3` is the reference
    /// default. `cfgScale = 1` (the default) keeps the bare conditional field for parity.
    /// `onStepSample` (optional) receives a cheap prediction of the *final* sample after
    /// each step — `x_t + (1−t)·v` reuses the corrector velocity, so it costs no extra
    /// DiT evaluations. Used for live "shape forming" previews; quality improves as t→1.
    public func sample(x0: MLXArray, ts: [Float], cond: MLXArray, cfgScale: Float = 1.0,
                       onStep: ((Int, Int) -> Void)? = nil,
                       onStepSample: ((Int, Int, MLXArray) -> Void)? = nil,
                       shouldStop: (() -> Bool)? = nil) -> MLXArray {
        sample(x0: x0, ts: ts, conds: [cond], cfgScale: cfgScale,
               onStep: onStep, onStepSample: onStepSample, shouldStop: shouldStop)
    }

    /// Multi-image conditioned sampling (same subject, different angles — the TRELLIS
    /// `run_multi_image` idea). Conditioning is pose-free, so this is purely a
    /// sampling-time change:
    ///   • `.multidiffusion` — average the conditional velocities over all views each
    ///     half-step. CFG is applied once to the mean (linearity makes that identical to
    ///     averaging per-view guided velocities), so the uncond eval stays single: cost
    ///     is N+1 forwards per half-step vs 2 for one view. Best geometry.
    ///   • `.stochastic` — round-robin one view per half-step. Single-view cost, noisier.
    ///   • `.concat` — cross-attend all views' tokens at once (m = N·1374). Single pass;
    ///     off-distribution for the single-image-trained DiT but cheap, and the right
    ///     mode once the fine-tune trains with multi-view concat conditioning.
    /// `shouldStop` is polled before every Heun step; when true the loop exits early
    /// and the *current* (partially integrated) sample is returned — callers decide
    /// whether to discard it (immediate cancel) or never set it mid-candidate
    /// (finish-candidate cancel).
    public func sample(x0: MLXArray, ts: [Float], conds condsIn: [MLXArray], cfgScale: Float = 1.0,
                       mode: MultiViewMode = .multidiffusion,
                       onStep: ((Int, Int) -> Void)? = nil,
                       onStepSample: ((Int, Int, MLXArray) -> Void)? = nil,
                       shouldStop: (() -> Bool)? = nil) -> MLXArray {
        precondition(!condsIn.isEmpty, "need at least one conditioning view")
        let conds = mode == .concat && condsIn.count > 1
            ? [concatenated(condsIn, axis: 1)] : condsIn
        let b = x0.dim(0)
        let useCFG = cfgScale > 1.0
        // Uncond tokens are all the same learned embedding, so cross-attention over them
        // is invariant to token count — any m works; use the first view's.
        let negCond: MLXArray? = useCFG
            ? broadcast(w("cond_embedder.y_embedding").reshaped([1, 1, conds[0].dim(2)]),
                        to: [conds[0].dim(0), conds[0].dim(1), conds[0].dim(2)]).asType(conds[0].dtype)
            : nil
        // Evaluate each forward pass before building the next so only ONE DiT
        // forward's activations are live at a time. Without this the lazy graph for
        // all forwards in a Heun step (N cond + uncond × predictor + corrector) is
        // held until the step's final eval — multiples of the no-CFG peak, which
        // OOM-kills a 16 GB machine mid-sampling.
        func vel(_ tv: Float, _ x: MLXArray, step: Int, half: Int) -> MLXArray {
            let t = MLXArray([Float](repeating: tv, count: b))
            var vc: MLXArray
            if mode == .stochastic && conds.count > 1 {
                let pick = (step * 2 + half) % conds.count
                vc = self(tokens: x, t: t, cond: conds[pick]).asType(.float32)
                vc.eval()
            } else {
                vc = self(tokens: x, t: t, cond: conds[0]).asType(.float32)
                vc.eval()
                for c in conds.dropFirst() {
                    let v = self(tokens: x, t: t, cond: c).asType(.float32)
                    v.eval()
                    vc = vc + v
                    vc.eval()
                }
                if conds.count > 1 { vc = vc / Float(conds.count); vc.eval() }
            }
            guard let neg = negCond else { return vc }
            let vu = self(tokens: x, t: t, cond: neg).asType(.float32)
            vu.eval()
            let g = vu + cfgScale * (vc - vu)
            g.eval()
            return g
        }
        var x = x0.asType(.float32)
        let n = ts.count - 1
        for i in 0 ..< n {
            if shouldStop?() == true { break }
            let h = ts[i + 1] - ts[i]
            let d0 = vel(ts[i], x, step: i, half: 0)
            let xMid = x + h * d0; xMid.eval()
            let d1 = vel(ts[i + 1], xMid, step: i, half: 1)
            x = x + (h * 0.5) * (d0 + d1)
            x.eval()
            Memory.clearCache()
            if let cb = onStepSample {
                let xPred = x + (1 - ts[i + 1]) * d1
                xPred.eval()
                cb(i + 1, n, xPred)
                Memory.clearCache()
            }
            onStep?(i + 1, n)
        }
        return x
    }
}
