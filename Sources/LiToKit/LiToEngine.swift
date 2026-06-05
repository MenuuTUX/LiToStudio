import Foundation
import MLX
import MLXRandom

/// The in-process LiTo image→3D engine: orchestrates all five native stages and writes
/// a colored point-cloud `.ply`. Replaces the Python subprocess entirely.
///
///   image → [Vision preprocess] cond_rgba → [DINOv2] cond → [DiT Heun-ODE] latent
///         → [voxel decoder + conv3d VAE] occupied-voxel coords → [gaussian decoder] gaussians → .ply
public final class LiToEngine {
    private let dino: Dinov2Encoder
    private let dit: DiT
    private let voxel: VoxelDecoder
    private let trellis: TrellisDecoder
    private let gauss: GaussianDecoder

    private static let latentMean: Float = 0.0661
    private static let latentStd: Float = 1.64639997
    private static let tEps: Float = 1e-4

    public enum EngineError: Error, CustomStringConvertible {
        case emptyShape
        public var description: String {
            "The model found no object in this image — try a clearer subject on a clean background."
        }
    }

    /// `weightsDir` holds `lito.safetensors` + `ss_dec_conv3d_16l8_fp16.safetensors`.
    /// `halfPrecision` (default on, matching the official LiTo demo on Apple Silicon) runs
    /// DINOv2/DiT/gaussian-decoder in float16 — half the memory, ~2× the throughput, and on
    /// 16 GB machines it avoids swap entirely. The occupancy voxel decoder stays float32.
    /// Set env `LITO_FP32=1` (or pass false) to force full precision everywhere.
    public init(weightsDir: URL, halfPrecision: Bool = ProcessInfo.processInfo.environment["LITO_FP32"] != "1") throws {
        let tw = try Weights(url: weightsDir.appending(path: "ss_dec_conv3d_16l8_fp16.safetensors"))
        let lito = weightsDir.appending(path: "lito.safetensors")
        let w = halfPrecision
            ? try Weights(url: lito, castTo: .float16,
                          keepFP32Prefixes: ["pretrained_tokenizer.voxel_decoder."])
            : try Weights(url: lito)
        dino = Dinov2Encoder(w)
        dit = DiT(w); voxel = VoxelDecoder(w); gauss = GaussianDecoder(w)
        trellis = TrellisDecoder(tw)
    }

    /// Run the full pipeline. `steps` = number of ODE timesteps (Heun: steps−1 corrector steps).
    /// `progress(fraction, stage)` is called as it advances. Returns the gaussian/point count.
    /// If `preprocessedRGBA` is set, it points to a pre-made RGBA PNG (from RMBG 2.0)
    /// and the native Vision masking is skipped.
    /// `occupancyThreshold` (logit; 0 = sigmoid 0.5, reference) raises the bar for a voxel to
    /// count as occupied — higher prunes low-confidence "ghost" geometry. `opacityThreshold`
    /// drops gaussians below that opacity from the cloud (prunes floaters).
    /// `minComponentFraction` drops disconnected occupancy islands smaller than that fraction
    /// of the largest connected component (floater blobs). `seed` makes runs reproducible.
    /// `outSplatPLY` additionally writes the full gaussians as a standard 3DGS splat PLY —
    /// the high-fidelity artifact (the point cloud is just a preview format).
    /// `onStepPreview` is called after each sampling step with the step index.
    /// `onStepCloud` (optional) additionally receives an *intermediate occupancy cloud* a
    /// few times per candidate — world-space voxel centers (count·3 floats, z-up) decoded
    /// from the running sample's final-state prediction. Drives the live "dots assembling
    /// into the shape" preview; costs one voxel+occupancy decode per emission.
    @discardableResult
    public func generate(imageURL: URL, steps: Int, outPLY: URL,
                         preprocessedRGBA: URL? = nil,
                         cfgScale: Float = 3.0,
                         occupancyThreshold: Float = 0,
                         opacityThreshold: Float = 0.10,
                         minComponentFraction: Float = 0.01,
                         seed: UInt64? = nil,
                         seedCandidates: Int = 1,
                         outSplatPLY: URL? = nil,
                         progress: @escaping (Double, String) -> Void,
                         onStepPreview: ((Int, Int) -> Void)? = nil,
                         onStepCloud: (([Float], Int, Int) -> Void)? = nil) throws -> Int {
        let res = 518

        progress(0.04, "Preprocessing image (\(res)²)")
        let condRGBA: MLXArray
        if let rgba = preprocessedRGBA {
            condRGBA = try Preprocess.condRGBA(rgbaURL: rgba, resolution: res)
        } else {
            condRGBA = try Preprocess.condRGBA(imageURL: imageURL, resolution: res)
        }

        progress(0.12, "Encoding image (DINOv2)")
        let cond = dino(condRGBA: condRGBA)
        cond.eval()

        var ts = [Float](repeating: 0, count: max(steps, 2))
        for i in ts.indices { ts[i] = Self.tEps + (1 - Self.tEps) * Float(i) / Float(ts.count - 1) }

        // Sample 1..N seeds; pick the candidate whose occupancy silhouette best matches the
        // conditioning mask (the paper shows the sampler converges by ~25 steps — remaining
        // variance is seed luck, which this search turns into a quality knob).
        let nCand = max(1, seedCandidates)
        let baseSeed = seed ?? UInt64.random(in: 0 ..< .max)
        var latent = MLXArray(0), coords = MLXArray(0), n = 0
        var bestIoU: Float = -1
        for c in 0 ..< nCand {
            MLXRandom.seed(baseSeed &+ UInt64(c) &* 7919)
            let x0 = MLXRandom.normal([1, 8192, 32])
            let candLabel = nCand > 1 ? " (candidate \(c + 1)/\(nCand))" : ""
            let f0 = 0.18 + 0.56 * Double(c) / Double(nCand)
            let fw = 0.56 / Double(nCand)
            // Live cloud preview: decode the predicted-final latent's occupancy every few
            // steps. minComponentFraction 0 keeps the stray specks — the coalescing fuzz
            // is the point of the visual. Early steps may legitimately decode to nothing.
            let cloudEvery = max(2, (ts.count - 1) / 5)
            let onStepSample: ((Int, Int, MLXArray) -> Void)? = onStepCloud == nil ? nil : { done, total, xPred in
                guard done % cloudEvery == 0 || done == total else { return }
                let lat = xPred * Self.latentStd + Self.latentMean
                let ss = self.voxel(latent: lat); ss.eval()
                let logit = self.trellis(ssLatent: ss); logit.eval()
                let (pc, pn) = self.trellis.initCoords(logit: logit, threshold: occupancyThreshold,
                                                       minComponentFraction: 0)
                if pn > 0 {
                    onStepCloud?(pc.asType(.float32).asArray(Float.self), done, total)
                }
                Memory.clearCache()
            }
            let sampled = dit.sample(x0: x0, ts: ts, cond: cond, cfgScale: cfgScale, onStep: { done, total in
                progress(f0 + fw * Double(done) / Double(total), "Sampling shape — step \(done)/\(total)\(candLabel)")
                onStepPreview?(done, total)
            }, onStepSample: onStepSample)
            let candLatent = sampled * Self.latentStd + Self.latentMean
            candLatent.eval(); Memory.clearCache()

            let ss = voxel(latent: candLatent); ss.eval()
            let logit = trellis(ssLatent: ss); logit.eval(); Memory.clearCache()
            let (candCoords, candN) = trellis.initCoords(logit: logit, threshold: occupancyThreshold,
                                                         minComponentFraction: minComponentFraction)
            if nCand == 1 {
                latent = candLatent; coords = candCoords; n = candN
                break
            }
            guard candN > 0 else { continue }
            let iou = Self.silhouetteIoU(coords: candCoords, count: candN, condRGBA: condRGBA)
            progress(f0 + fw, String(format: "Candidate %d/%d — silhouette IoU %.3f", c + 1, nCand, iou))
            if iou > bestIoU {
                bestIoU = iou
                latent = candLatent; coords = candCoords; n = candN
            }
        }
        guard n > 0 else { throw EngineError.emptyShape }
        progress(0.78, nCand > 1 ? String(format: "Best seed IoU %.3f — decoding", bestIoU) : "Decoding structure")

        progress(0.88, "Decoding \(n) gaussians")
        let gs = gauss(latent: latent, initCoord: coords)
        gs.values.forEach { $0.eval() }; Memory.clearCache()

        progress(0.96, "Writing splat")
        if let splatURL = outSplatPLY {
            try Splat.writeGaussians(gs, to: splatURL)
        }
        let count = try Splat.writePointCloud(gs, to: outPLY, opacityThreshold: opacityThreshold)
        progress(1.0, "Done")
        return count
    }

    /// IoU between the occupancy silhouette projected into the conditioning view and the
    /// conditioning alpha mask. The conditioning camera looks along +x with v=z up, u=y
    /// (determined empirically via `LiToSmoke score`: IoU 0.64 vs ≤0.61 for all others).
    static func silhouetteIoU(coords: MLXArray, count: Int, condRGBA: MLXArray) -> Float {
        let S = 64
        let res = condRGBA.dim(0)
        let alpha: [Float] = condRGBA[0 ..< res, 0 ..< res, 3 ..< 4].reshaped([res * res]).asArray(Float.self)
        var mask = [Bool](repeating: false, count: S * S)
        for y in 0 ..< S { for x in 0 ..< S {
            if alpha[(y * res / S) * res + x * res / S] > 0.5 { mask[y * S + x] = true }
        }}
        let c = coords.asType(.float32).asArray(Float.self)
        var sil = [Bool](repeating: false, count: S * S)
        for i in 0 ..< count {
            let u = c[i * 3 + 1], v = c[i * 3 + 2]                  // (y, z) plane
            let xi = Int((u + 1.04) / 2.08 * Float(S)), yi = Int((1.04 - v) / 2.08 * Float(S))
            if xi >= 0, xi < S, yi >= 0, yi < S { sil[yi * S + xi] = true }
        }
        var inter = 0, uni = 0
        for j in 0 ..< S * S {
            if sil[j] && mask[j] { inter += 1 }
            if sil[j] || mask[j] { uni += 1 }
        }
        return Float(inter) / Float(max(uni, 1))
    }
}
