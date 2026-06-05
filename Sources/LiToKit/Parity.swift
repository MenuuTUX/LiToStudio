import Foundation
import MLX

/// Dev-time parity checks: run a Swift stage and compare to the Python golden
/// tensors captured from the reference pipeline.
public enum Parity {
    /// Stage 2 gate: DINOv2 cond tokens vs the golden, fed the identical `cond_rgba`.
    public static func dino(weights: URL, golden: URL) throws {
        let t0 = Date()
        let w = try Weights(url: weights)
        let g = try loadArrays(url: golden)
        guard let condRGBA = g["cond_rgba"], let ref = g["cond_tokens_f32"] else {
            print("✗ golden missing cond_rgba / cond_tokens_f32"); return
        }
        print("loaded weights + golden in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s; cond_rgba \(condRGBA.shape)")
        let enc = Dinov2Encoder(w)
        let out = enc(condRGBA: condRGBA)
        out.eval()
        compare("DINO cond", out, ref)
    }

    /// Stage 3 gate: DiT velocity (single forward) + 1-step Heun result, same x0/cond.
    public static func dit(weights: URL, golden: URL) throws {
        let t0 = Date()
        let w = try Weights(url: weights)
        let g = try loadArrays(url: golden)
        guard let cond = g["cond_tokens"], let x0 = g["x0"], let tt = g["t0"],
              let v0ref = g["v0"], let ts = g["ts"], let sampledRef = g["sampled"] else {
            print("✗ dit golden missing keys"); return
        }
        print("loaded weights + golden in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
        let dit = DiT(w)
        let v0 = dit(tokens: x0, t: tt, cond: cond); v0.eval()
        compare("DiT velocity (1 fwd)", v0, v0ref)
        let sampled = dit.sample(x0: x0, ts: ts.asArray(Float.self), cond: cond); sampled.eval()
        compare("DiT sampled (Heun)", sampled, sampledRef)
    }

    /// Stage 4 gate: voxel decoder ss-latent + TRELLIS occupancy + init-coord count.
    public static func voxel(weights: URL, trellis: URL, golden: URL) throws {
        let w = try Weights(url: weights)
        let tw = try Weights(url: trellis)
        let g = try loadArrays(url: golden)
        guard let latent = g["latent"], let ssRef0 = g["ss_latent"],
              let logitRef0 = g["occ_logit"], let icRef = g["init_coord"] else {
            print("✗ voxel golden missing keys"); return
        }
        let ss = VoxelDecoder(w)(latent: latent); ss.eval()                 // (1,16,16,16,8)
        compare("ss_latent", ss, ssRef0.transposed(0, 2, 3, 4, 1))          // golden NCDHW → NDHWC
        let td = TrellisDecoder(tw)
        let logit = td(ssLatent: ss); logit.eval()                          // (1,64,64,64,1)
        compare("occ_logit", logit, logitRef0.transposed(0, 2, 3, 4, 1))
        let (coords, n) = td.initCoords(logit: logit)
        let refN = icRef.dim(0)
        print("  init voxels: swift=\(n)  ref=\(refN)  (Δ=\(n - refN))")
        if n == refN { compare("init_coord", coords, icRef) }
    }

    /// Stage 5 gate: final gaussians (xyz/scaling/quat/opacity/rgb_sh) from latent+init_coord.
    public static func gs(weights: URL, golden: URL) throws {
        let w = try Weights(url: weights)
        let g = try loadArrays(url: golden)
        guard let latent = g["latent"], let ic = g["init_coord"] else { print("✗ gs golden missing keys"); return }
        let dec = GaussianDecoder(w)
        if let catRef = g["iq_cat"] {                                 // init_query sub-stage localization
            let s = dec.forwardInitStages(initCoord: ic); s.forEach { $0.eval() }
            compare("gs.iq_cat", s[0], catRef)
            if let r = g["iq_post_linear"] { compare("gs.iq_post_linear", s[1], r) }
            if let r = g["iq_post_mlp"] { compare("gs.iq_post_mlp", s[2], r) }
        }
        if let iq = g["init_query"] {                                 // per-block divergence localization
            let dbg = dec.forwardDebug(latent: latent, initCoord: ic)
            dbg.forEach { $0.eval() }
            compare("gs.init_query", dbg[0], iq)
            for i in 0 ..< 6 { if let ref = g["blk_\(i)"] { compare("gs.blk_\(i)", dbg[i + 1], ref) } }
        }
        if let sRef = g["shape_out"], let cRef = g["color_out"] {     // isolate perceiver vs decode_gs
            let (s, c) = dec.forwardRaw(latent: latent, initCoord: ic); s.eval(); c.eval()
            compare("gs.shape_out(raw)", s, sRef)
            compare("gs.color_out(raw)", c, cRef)
        }
        let out = dec(latent: latent, initCoord: ic)
        for kk in ["xyz_w", "scaling", "quaternion", "opacity", "rgb_sh"] {
            guard let o = out[kk], let ref = g[kk] else { print("  \(kk): MISSING"); continue }
            o.eval(); compare("gs.\(kk)", o, ref)
        }
    }

    /// Cosine similarity + abs-error report; a >0.999 cosine is our parity bar.
    public static func compare(_ tag: String, _ a: MLXArray, _ b: MLXArray) {
        let af = a.reshaped([-1]).asType(.float32)
        let bf = b.reshaped([-1]).asType(.float32)
        let cos = ((af * bf).sum() / (sqrt((af * af).sum()) * sqrt((bf * bf).sum()))).item(Float.self)
        let maxAbs = abs(af - bf).max().item(Float.self)
        let meanAbs = abs(af - bf).mean().item(Float.self)
        let refMax = abs(bf).max().item(Float.self)
        print(String(format: "[%@] %@  cosine=%.6f  max|Δ|=%.5f  mean|Δ|=%.6f  (ref max|x|=%.3f)",
                     tag, a.shape.description, cos, maxAbs, meanAbs, refMax))
        print(cos > 0.999 ? "  ✓ PARITY PASS (cosine > 0.999)" : "  ✗ PARITY FAIL")
    }
}
