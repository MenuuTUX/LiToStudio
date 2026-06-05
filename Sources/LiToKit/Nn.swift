import Foundation
import MLX
import MLXNN

/// Small functional building blocks. We load weights by exact name and apply them
/// explicitly (rather than via MLXNN.Module) so the port maps 1:1 to the torch code
/// and weight wiring is unambiguous.
enum Nn {
    /// `y = x · Wᵀ + b` for a torch `Linear` (weight stored as [out, in]).
    static func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
        var y = matmul(x, w.transposed())
        if let b { y = y + b }
        return y
    }

    /// LayerNorm over the last axis. `w`/`b` nil ⇒ the parameterless variant
    /// (torch `F.layer_norm(x, x.shape[-1:])`, default eps 1e-5).
    /// Statistics run in float32 regardless of input dtype (fp16 variance under/overflows),
    /// mirroring torch autocast, then cast back to the input dtype.
    static func layerNorm(_ x: MLXArray, _ w: MLXArray? = nil, _ b: MLXArray? = nil,
                          eps: Float) -> MLXArray {
        let xf = x.dtype == .float32 ? x : x.asType(.float32)
        let mu = xf.mean(axis: -1, keepDims: true)
        let xc = xf - mu
        let v = (xc * xc).mean(axis: -1, keepDims: true)
        var y = xc * rsqrt(v + eps)
        if let w { y = y * w.asType(.float32) }
        if let b { y = y + b.asType(.float32) }
        return y.asType(x.dtype)
    }

    /// Exact GELU (erf-based), matching torch `nn.GELU()`.
    static func gelu(_ x: MLXArray) -> MLXArray { MLXNN.gelu(x) }

    /// GELU with the tanh approximation (the DiT's `Mlp` uses this).
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        let c: Float = 0.7978845608028654          // sqrt(2/π)
        return 0.5 * x * (1 + tanh(c * (x + 0.044715 * (x * x * x))))
    }

    /// RMSNorm over the last axis: `x · rsqrt(mean(x²)+eps) · scale` (DiT default eps 1e-8).
    /// Statistics in float32 (see layerNorm), result back in the input dtype.
    static func rmsNorm(_ x: MLXArray, _ scale: MLXArray, eps: Float = 1e-8) -> MLXArray {
        let xf = x.dtype == .float32 ? x : x.asType(.float32)
        return (xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps) * scale.asType(.float32))
            .asType(x.dtype)
    }
}
