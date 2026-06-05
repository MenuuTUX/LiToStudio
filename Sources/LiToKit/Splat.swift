import Foundation
import MLX

/// Stage 6 — gaussians → a colored point-cloud `.ply` (what the SceneKit viewer renders).
/// Bakes SH degree-0 → RGB (`0.5 + 0.282·sh0`), drops near-transparent gaussians, and
/// writes a binary little-endian `x,y,z,red,green,blue` cloud. Mirrors `splat_to_pc.py`.
public enum Splat {
    static let shC0: Float = 0.28209479177387814

    @discardableResult
    public static func writePointCloud(_ gs: [String: MLXArray], to url: URL,
                                       opacityThreshold: Float = 0.10, maxPoints: Int = 500_000) throws -> Int {
        guard let xyzW = gs["xyz_w"], let rgbSh = gs["rgb_sh"], let opacity = gs["opacity"] else {
            throw Err.missing
        }
        let N = xyzW.dim(0), k = xyzW.dim(1)
        let xyz = xyzW.reshaped([N * k, 3])
        let sh0 = rgbSh[0 ..< N, 0 ..< k, 0 ..< 1, 0 ..< 3].reshaped([N * k, 3])   // degree-0 coeff
        let op = opacity.reshaped([N * k])

        let xyzA = xyz.asType(.float32).asArray(Float.self)     // (M·3)
        let shA = sh0.asType(.float32).asArray(Float.self)      // (M·3)
        let opA = op.asType(.float32).asArray(Float.self)       // (M)
        let M = opA.count

        // keep opaque gaussians (opacity is already post-sigmoid in [0,1])
        var keep = [Int](); keep.reserveCapacity(M)
        for i in 0 ..< M where opA[i] > opacityThreshold { keep.append(i) }
        if keep.count > maxPoints {                            // deterministic subsample for the viewer
            var rng = SplitMix64(seed: 0)
            for i in stride(from: keep.count - 1, through: 1, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1)); keep.swapAt(i, j)
            }
            keep = Array(keep.prefix(maxPoints))
        }

        var body = Data(capacity: keep.count * 15)
        for i in keep {
            var x = xyzA[i * 3], y = xyzA[i * 3 + 1], z = xyzA[i * 3 + 2]
            withUnsafeBytes(of: &x) { body.append(contentsOf: $0) }
            withUnsafeBytes(of: &y) { body.append(contentsOf: $0) }
            withUnsafeBytes(of: &z) { body.append(contentsOf: $0) }
            for c in 0 ..< 3 {
                let v = max(0, min(1, 0.5 + Splat.shC0 * shA[i * 3 + c]))
                body.append(UInt8(v * 255))
            }
        }
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(keep.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        var out = Data(header.utf8); out.append(body)
        try out.write(to: url)
        return keep.count
    }

    /// Write the full gaussians as a standard 3DGS (INRIA-format) binary PLY:
    /// x,y,z, nx,ny,nz(0), f_dc_0..2, f_rest_0..44 (channel-major SH), opacity (logit),
    /// scale_0..2 (log), rot_0..3 — byte-compatible with `plibs.gs_utils.save_ply` and
    /// loadable by any gaussian-splat viewer. Coordinates stay in LiTo's native z-up frame
    /// (same as the reference exporter); viewers handle orientation.
    /// `opacityThreshold` drops fully-dead gaussians; keep it small — a real splat renderer
    /// blends faint gaussians correctly, unlike the baked point cloud.
    @discardableResult
    public static func writeGaussians(_ gs: [String: MLXArray], to url: URL,
                                      opacityThreshold: Float = 0.01) throws -> Int {
        guard let xyzW = gs["xyz_w"], let rgbSh = gs["rgb_sh"], let opacity = gs["opacity"],
              let scaling = gs["scaling"], let quat = gs["quaternion"] else { throw Err.missing }
        let N = xyzW.dim(0), k = xyzW.dim(1), M = N * k
        let shCoeffs = rgbSh.dim(2)                                   // 16 for degree 3
        let xyzA = xyzW.reshaped([M, 3]).asType(.float32).asArray(Float.self)
        let shA = rgbSh.reshaped([M, shCoeffs * 3]).asType(.float32).asArray(Float.self)  // (sh, 3) row-major
        let opA = opacity.reshaped([M]).asType(.float32).asArray(Float.self)
        let scA = scaling.reshaped([M, 3]).asType(.float32).asArray(Float.self)
        let qA = quat.reshaped([M, 4]).asType(.float32).asArray(Float.self)

        var keep = [Int](); keep.reserveCapacity(M)
        for i in 0 ..< M where opA[i] > opacityThreshold { keep.append(i) }

        let rest = shCoeffs - 1
        let floatsPerVertex = 3 + 3 + 3 + rest * 3 + 1 + 3 + 4
        var body = [Float](); body.reserveCapacity(keep.count * floatsPerVertex)
        for i in keep {
            body.append(xyzA[i * 3]); body.append(xyzA[i * 3 + 1]); body.append(xyzA[i * 3 + 2])
            body.append(0); body.append(0); body.append(0)                       // normals
            let sh = i * shCoeffs * 3
            body.append(shA[sh]); body.append(shA[sh + 1]); body.append(shA[sh + 2])   // f_dc rgb
            for c in 0 ..< 3 {                                                    // f_rest channel-major
                for j in 1 ..< shCoeffs { body.append(shA[sh + j * 3 + c]) }
            }
            let o = min(max(opA[i], 1e-6), 1 - 1e-6)
            body.append(log(o / (1 - o)))                                         // opacity logit
            for c in 0 ..< 3 { body.append(log(max(scA[i * 3 + c], 1e-9))) }       // log scale
            // LiTo quaternions are xyzw (scalar-last); 3DGS PLY rot_0..3 is wxyz — reorder.
            body.append(qA[i * 4 + 3])
            body.append(qA[i * 4]); body.append(qA[i * 4 + 1]); body.append(qA[i * 4 + 2])
        }

        var header = """
        ply
        format binary_little_endian 1.0
        element vertex \(keep.count)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz

        """
        for j in 0 ..< 3 { header += "property float f_dc_\(j)\n" }
        for j in 0 ..< rest * 3 { header += "property float f_rest_\(j)\n" }
        header += "property float opacity\n"
        for j in 0 ..< 3 { header += "property float scale_\(j)\n" }
        for j in 0 ..< 4 { header += "property float rot_\(j)\n" }
        header += "end_header\n"

        var out = Data(header.utf8)
        body.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        try out.write(to: url)
        return keep.count
    }

    enum Err: Error { case missing }
}

/// Tiny deterministic RNG for reproducible subsampling.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
