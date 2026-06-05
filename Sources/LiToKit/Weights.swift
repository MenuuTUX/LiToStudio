import Foundation
import MLX

/// A safetensors-backed weight store. Tensors are looked up by their exact
/// (torch) parameter name; the converter preserved those names verbatim.
public final class Weights {
    public let arrays: [String: MLXArray]

    public init(url: URL) throws { self.arrays = try loadArrays(url: url) }
    public init(_ arrays: [String: MLXArray]) { self.arrays = arrays }

    /// Load with a one-time dtype cast (e.g. `.float16` — halves resident memory and runs
    /// matmuls at half precision, the official LiTo demo's configuration on Apple Silicon).
    /// Tensors whose name starts with any of `keepFP32Prefixes` stay float32 — for small,
    /// precision-sensitive heads like the occupancy voxel decoder.
    public init(url: URL, castTo dtype: DType, keepFP32Prefixes: [String] = []) throws {
        let raw = try loadArrays(url: url)
        var out = [String: MLXArray](minimumCapacity: raw.count)
        for (k, v) in raw {
            let keep = keepFP32Prefixes.contains { k.hasPrefix($0) }
            out[k] = keep ? v : v.asType(dtype)
        }
        // Materialize the cast copies now so the mmapped fp32 originals can be evicted.
        eval(Array(out.values))
        self.arrays = out
    }

    public func has(_ name: String) -> Bool { arrays[name] != nil }

    /// Fetch a tensor by name (optionally cast). Traps on a missing key — a missing
    /// weight is a porting bug we want surfaced loudly, not silently zero-filled.
    public func callAsFunction(_ name: String, as dtype: DType? = nil) -> MLXArray {
        guard let a = arrays[name] else {
            fatalError("Weights: missing tensor '\(name)'")
        }
        return dtype.map { a.asType($0) } ?? a
    }

    /// A view whose lookups are prefixed (e.g. `w.prefixed("patch_encoder.")`).
    public func prefixed(_ prefix: String) -> Scoped { Scoped(self, prefix) }

    public struct Scoped {
        let w: Weights; let prefix: String
        init(_ w: Weights, _ p: String) { self.w = w; self.prefix = p }
        public func callAsFunction(_ name: String, as dtype: DType? = nil) -> MLXArray { w(prefix + name, as: dtype) }
        public func has(_ name: String) -> Bool { w.has(prefix + name) }
        public func prefixed(_ p: String) -> Scoped { Scoped(w, prefix + p) }
    }
}
