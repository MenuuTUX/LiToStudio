import Foundation
import MLX
import MLXNN
import MLXFast

/// Stage 5 — `GaussianDecoderXv`, the final decoder that turns the shape latent +
/// occupied-voxel init coords into 3D gaussians. Port of `mlx/models/gaussian_decoder.py`.
///
/// Query = init coords (xyz + Fourier enc) → point MLP → 6 perceiver blocks
/// (global cross-attn over the latent + **localized_voxel** windowed self-attn) →
/// shape/color output MLPs → `decode_gs` activations. 64 gaussians per voxel.
/// Everything is **packed** (b=1): no padding/masks needed for the global parts.
public struct GaussianDecoder {
    let w: Weights.Scoped
    static let pdim = 512, heads = 8, dh = 64, blocks = 6, numSA = 2, k = 64
    static let scale = Float(1.0 / 8.0)            // dim_head(64)^-0.5
    static let cellWidth: Float = 0.25

    public init(_ weights: Weights) { self.w = weights.prefixed("pretrained_tokenizer.gs_decoder.") }

    /// `latent` (8192,32) packed, `initCoord` (Nvox,3) → gaussians dict (each (Nvox,64,·)).
    public func callAsFunction(latent latentIn: MLXArray, initCoord: MLXArray) -> [String: MLXArray] {
        let (shapeOut, colorOut) = forwardRaw(latent: latentIn, initCoord: initCoord)
        return decodeGS(shapeOut, colorOut, initCoord, initCoord.dim(0))
    }

    /// Raw output-MLP results (shape (N,640), color (N,3136)) — before `decode_gs`. For parity isolation.
    /// Runs in the weights' dtype (fp16 in half-precision mode). The Fourier xyz encoding
    /// stays float32 — sin/cos of coord·2¹² is meaningless at fp16 resolution — and the
    /// query stream is cast to the compute dtype after the point MLP.
    public func forwardRaw(latent latentIn: MLXArray, initCoord initCoordIn: MLXArray) -> (MLXArray, MLXArray) {
        let wdt = w("point_linear.weight").dtype
        var latent = latentIn.ndim == 3 ? latentIn.reshaped([latentIn.dim(1), latentIn.dim(2)]) : latentIn
        latent = latent.asType(wdt)
        let initCoord = initCoordIn.asType(.float32)
        let N = initCoord.dim(0)

        // voxelizations for the two windowed-attn shifts (0 and half-cell), reused across blocks
        let coordsHost = initCoord.asType(.float32).asArray(Float.self)
        let voxByLayer = [VoxelInfo(coordsHost, N, GaussianDecoder.cellWidth, 0),
                          VoxelInfo(coordsHost, N, GaussianDecoder.cellWidth, 0.5 * GaussianDecoder.cellWidth)]

        // init query: [xyz(3), Fourier(no input)(192)] → point_linear → point MLP
        let freq = w("xyz_encoding.freq_bands").asType(.float16).asType(.float32)   // match fp16-loaded reference                       // (32,)
        let pe = initCoord.expandedDimensions(axis: -1) * freq        // (N,3,32)
        let enc = concatenated([sin(pe).reshaped([N, 96]), cos(pe).reshaped([N, 96])], axis: -1)
        var q = Nn.linear(concatenated([initCoord, enc], axis: -1), w("point_linear.weight"), w("point_linear.bias"))
        q = outputMLP(q, "point_mlp").asType(wdt)                     // (N,512)

        // perceiver: 6 blocks (cross-attn + mlp + 2×[voxel self-attn + mlp])
        for b in 0 ..< GaussianDecoder.blocks {
            let bl = w.prefixed("perceiver.blocks.\(b).")
            let kv = Nn.linear(latent, bl("kv_linear.weight"))        // (8192,32), no bias
            q = q + crossAttn(q, kv, bl.prefixed("ca_layer."))
            q = q + swiglu(Nn.layerNorm(q, bl("ca_ln.weight"), bl("ca_ln.bias"), eps: 1e-6), bl, "ca_mlp")
            for s in 0 ..< GaussianDecoder.numSA {
                q = q + selfAttnVoxel(Nn.layerNorm(q, bl("ln1_layers.\(s).weight"), bl("ln1_layers.\(s).bias"), eps: 1e-6),
                                      bl.prefixed("sa_layers.\(s)."), voxByLayer[s])
                q = q + swiglu(Nn.layerNorm(q, bl("ln2_layers.\(s).weight"), bl("ln2_layers.\(s).bias"), eps: 1e-6), bl, "mlp_layers.\(s)")
            }
        }

        _ = N
        return (outputMLP(q, "gs_output_shape_mlp"), outputMLP(q, "gs_output_color_mlp"))
    }

    /// Debug: [init_query, block0_out, …, block5_out] — to localize a divergence.
    public func forwardDebug(latent latentIn: MLXArray, initCoord: MLXArray) -> [MLXArray] {
        let latent = latentIn.ndim == 3 ? latentIn.reshaped([latentIn.dim(1), latentIn.dim(2)]) : latentIn
        let N = initCoord.dim(0)
        let coordsHost = initCoord.asType(.float32).asArray(Float.self)
        let voxByLayer = [VoxelInfo(coordsHost, N, GaussianDecoder.cellWidth, 0),
                          VoxelInfo(coordsHost, N, GaussianDecoder.cellWidth, 0.5 * GaussianDecoder.cellWidth)]
        let freq = w("xyz_encoding.freq_bands").asType(.float16).asType(.float32)   // match fp16-loaded reference
        let pe = initCoord.expandedDimensions(axis: -1) * freq
        let enc = concatenated([sin(pe).reshaped([N, 96]), cos(pe).reshaped([N, 96])], axis: -1)
        var q = Nn.linear(concatenated([initCoord, enc], axis: -1), w("point_linear.weight"), w("point_linear.bias"))
        q = outputMLP(q, "point_mlp")
        var dbg = [q]
        for b in 0 ..< GaussianDecoder.blocks {
            let bl = w.prefixed("perceiver.blocks.\(b).")
            let kv = Nn.linear(latent, bl("kv_linear.weight"))
            q = q + crossAttn(q, kv, bl.prefixed("ca_layer."))
            q = q + swiglu(Nn.layerNorm(q, bl("ca_ln.weight"), bl("ca_ln.bias"), eps: 1e-6), bl, "ca_mlp")
            for s in 0 ..< GaussianDecoder.numSA {
                q = q + selfAttnVoxel(Nn.layerNorm(q, bl("ln1_layers.\(s).weight"), bl("ln1_layers.\(s).bias"), eps: 1e-6),
                                      bl.prefixed("sa_layers.\(s)."), voxByLayer[s])
                q = q + swiglu(Nn.layerNorm(q, bl("ln2_layers.\(s).weight"), bl("ln2_layers.\(s).bias"), eps: 1e-6), bl, "mlp_layers.\(s)")
            }
            dbg.append(q)
        }
        return dbg
    }

    /// Debug: [concat(xyz,enc), post_point_linear, post_point_mlp(=init_query)].
    public func forwardInitStages(initCoord: MLXArray) -> [MLXArray] {
        let N = initCoord.dim(0)
        let freq = w("xyz_encoding.freq_bands").asType(.float16).asType(.float32)   // match fp16-loaded reference
        let pe = initCoord.expandedDimensions(axis: -1) * freq
        let enc = concatenated([sin(pe).reshaped([N, 96]), cos(pe).reshaped([N, 96])], axis: -1)
        let cat = concatenated([initCoord, enc], axis: -1)
        let lin = Nn.linear(cat, w("point_linear.weight"), w("point_linear.bias"))
        return [cat, lin, outputMLP(lin, "point_mlp")]
    }

    // fused-w12 SwiGLU: w3(silu(w1·x) * w2·x), w12 = [w1; w2]; no bias
    private func swiglu(_ x: MLXArray, _ s: Weights.Scoped, _ name: String) -> MLXArray {
        let ab = Nn.linear(x, s("\(name).w12.weight")).split(parts: 2, axis: -1)
        return Nn.linear(silu(ab[0]) * ab[1], s("\(name).w3.weight"))
    }

    // (LayerNorm → SwiGLU) → (norm_final → linear); non-residual
    private func outputMLP(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = swiglu(Nn.layerNorm(x, w("\(name).0.weight"), w("\(name).0.bias"), eps: 1e-6), w, "\(name).1")
        let z = Nn.layerNorm(y, w("\(name).2.norm_final.weight"), w("\(name).2.norm_final.bias"), eps: 1e-6)
        return Nn.linear(z, w("\(name).2.linear.weight"), w("\(name).2.linear.bias"))
    }

    // global cross-attention (b=1: no padding); q (N,512) attends kv (M,32)
    private func crossAttn(_ q: MLXArray, _ kv: MLXArray, _ s: Weights.Scoped) -> MLXArray {
        let N = q.dim(0), M = kv.dim(0)
        let q0 = Nn.layerNorm(q, s("layernorm_q.weight"), s("layernorm_q.bias"), eps: 1e-5)
        let kv0 = Nn.layerNorm(kv, s("layernorm_kv.weight"), s("layernorm_kv.bias"), eps: 1e-5)
        let qp = Nn.rmsNorm(Nn.linear(q0, s("linear_q.weight"), s("linear_q.bias")), s("rmsnorm_q.scale"))
        let kvp = Nn.linear(kv0, s("linear_kv.weight"), s("linear_kv.bias")).split(parts: 2, axis: -1)
        let kp = Nn.rmsNorm(kvp[0], s("rmsnorm_k.scale"))
        func H(_ a: MLXArray, _ n: Int) -> MLXArray { a.reshaped([1, n, GaussianDecoder.heads, GaussianDecoder.dh]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(queries: H(qp, N), keys: H(kp, M), values: H(kvp[1], M),
                                                  scale: GaussianDecoder.scale, mask: .none)
        return Nn.linear(o.transposed(0, 2, 1, 3).reshaped([N, GaussianDecoder.pdim]), s("linear_out.weight"), s("linear_out.bias"))
    }

    // localized_voxel windowed self-attention
    private func selfAttnVoxel(_ x: MLXArray, _ s: Weights.Scoped, _ vox: VoxelInfo) -> MLXArray {
        let N = x.dim(0), h = GaussianDecoder.heads, hd = GaussianDecoder.dh
        let qkv = Nn.linear(x, s("linear_qkv.weight"), s("linear_qkv.bias")).split(parts: 3, axis: -1)
        let q = Nn.rmsNorm(qkv[0], s("rmsnorm_q.scale")).reshaped([N, h, hd])
        let k = Nn.rmsNorm(qkv[1], s("rmsnorm_k.scale")).reshaped([N, h, hd])
        let v = qkv[2].reshaped([N, h, hd])
        let out = vox.attend(q, k, v, scale: GaussianDecoder.scale)
        return Nn.linear(out.reshaped([N, GaussianDecoder.pdim]), s("linear_out.weight"), s("linear_out.bias"))
    }

    // decode_gs activations + per-voxel xyz offset
    private func decodeGS(_ shapeOut: MLXArray, _ colorOut: MLXArray, _ initCoord: MLXArray, _ N: Int) -> [String: MLXArray] {
        let k = GaussianDecoder.k
        let s = shapeOut.reshaped([N, k, 10]), c = colorOut.reshaped([N, k, 49])
        let xyzRaw = s[0 ..< N, 0 ..< k, 0 ..< 3]
        let quatRaw = s[0 ..< N, 0 ..< k, 3 ..< 7]
        let scaleRaw = s[0 ..< N, 0 ..< k, 7 ..< 10]
        let opRaw = c[0 ..< N, 0 ..< k, 0 ..< 1]
        let shRaw = c[0 ..< N, 0 ..< k, 1 ..< 49]

        let xyz = (sigmoid(xyzRaw) * 2 - 1) * 0.05 + initCoord.reshaped([N, 1, 3])   // ± region_scaling + center
        let quat = quatRaw * rsqrt((quatRaw * quatRaw).sum(axis: -1, keepDims: true))  // normalize
        var scaling = sigmoid(scaleRaw) * 0.01                                          // sigmoid·scaling_scalar
        scaling = sqrt(scaling * scaling + Float(0.001 * 0.001))                        // ⊕ min_scaling
        let opacity = sigmoid(opRaw * 1.0 + 0.1)                                        // sigmoid(scale·x+bias)
        let rgbSh = shRaw.reshaped([N, k, 16, 3])
        return ["xyz_w": xyz, "scaling": scaling, "quaternion": quat, "opacity": opacity, "rgb_sh": rgbSh]
    }
}

/// Host-computed voxel windowing for one (cell_width, shift): sort points by cell,
/// then gather/pad/SDPA/scatter so each point attends only within its voxel cell.
struct VoxelInfo {
    let forwardIdxs, backwardIdxs, padGather, unpadGather: MLXArray   // int32
    let maskAdd: MLXArray                                             // (numCells,1,1,maxLen) f32
    let numCells, maxLen: Int

    init(_ coords: [Float], _ n: Int, _ cw: Float, _ shift: Float) {
        // cell key per point (i/x fastest), offset to keep non-negative
        var keys = [Int64](repeating: 0, count: n)
        for p in 0 ..< n {
            let i = Int(floor((coords[p * 3] + shift) / cw)) + 256
            let j = Int(floor((coords[p * 3 + 1] + shift) / cw)) + 256
            let kk = Int(floor((coords[p * 3 + 2] + shift) / cw)) + 256
            keys[p] = (Int64(kk) * 512 + Int64(j)) * 512 + Int64(i)
        }
        let order = (0 ..< n).sorted { keys[$0] != keys[$1] ? keys[$0] < keys[$1] : $0 < $1 }

        var cellStart = [Int](), cellLen = [Int](), sIdx = 0
        while sIdx < n {
            let key = keys[order[sIdx]]; var e = sIdx
            while e < n && keys[order[e]] == key { e += 1 }
            cellStart.append(sIdx); cellLen.append(e - sIdx); sIdx = e
        }
        let nc = cellStart.count, ml = cellLen.max() ?? 0
        self.numCells = nc; self.maxLen = ml

        var fwd = [Int32](repeating: 0, count: n), back = [Int32](repeating: 0, count: n)
        var unpad = [Int32](repeating: 0, count: n)
        var pad = [Int32](repeating: 0, count: nc * ml)
        var mask = [Float](repeating: -1e9, count: nc * ml)
        for s in 0 ..< n { fwd[s] = Int32(order[s]); back[order[s]] = Int32(s) }
        for c in 0 ..< nc {
            for p in 0 ..< cellLen[c] {
                pad[c * ml + p] = Int32(cellStart[c] + p)
                unpad[cellStart[c] + p] = Int32(c * ml + p)
                mask[c * ml + p] = 0
            }
        }
        self.forwardIdxs = MLXArray(fwd); self.backwardIdxs = MLXArray(back)
        self.padGather = MLXArray(pad); self.unpadGather = MLXArray(unpad)
        self.maskAdd = MLXArray(mask).reshaped([max(nc, 1), 1, 1, max(ml, 1)])
    }

    /// q,k,v (N,h,dh) → windowed attention output (N,h,dh).
    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        if numCells == 0 { return q }
        let h = q.dim(1), hd = q.dim(2)
        func pad(_ a: MLXArray) -> MLXArray {
            take(take(a, forwardIdxs, axis: 0), padGather, axis: 0)
                .reshaped([numCells, maxLen, h, hd]).transposed(0, 2, 1, 3)   // (nc,h,ml,hd)
        }
        let o = MLXFast.scaledDotProductAttention(queries: pad(q), keys: pad(k), values: pad(v),
                                                  scale: scale, mask: .array(maskAdd.asType(q.dtype)))
        let flat = o.transposed(0, 2, 1, 3).reshaped([numCells * maxLen, h, hd])    // (nc·ml,h,hd)
        return take(take(flat, unpadGather, axis: 0), backwardIdxs, axis: 0)        // unpad → unsort
    }
}
