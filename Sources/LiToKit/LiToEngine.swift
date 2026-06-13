import Foundation
import MLX
import MLXRandom

/// The in-process LiTo image→3D engine: orchestrates all five native stages and writes
/// a colored point-cloud `.ply`. Replaces the Python subprocess entirely.
///
///   image(s) → [Vision preprocess] cond_rgba → [DINOv2] cond → [DiT Heun-ODE] latent
///            → [voxel decoder + conv3d VAE] occupied-voxel coords → [gaussian decoder] gaussians → .ply
///
/// One image is the reference path. Several images of the *same subject in the same
/// pose* (front/¾/side/back — e.g. an AI-generated turnaround) condition one shape
/// together via `MultiViewMode`; every view also scores seed candidates and gets a
/// yaw estimate so later stages (texture backprojection, per-view refinement) know
/// where each photo sits around the model.
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
        case noViews
        case cancelled
        public var description: String {
            switch self {
            case .emptyShape:
                return "The model found no object in this image — try a clearer subject on a clean background."
            case .noViews:
                return "No conditioning images were provided."
            case .cancelled:
                return "Generation cancelled."
            }
        }
    }

    /// What a generation produced beyond the PLY files.
    public struct GenResult {
        public let pointCount: Int
        /// Estimated camera azimuth per input view, radians around +z; 0 = the
        /// conditioning convention (camera at +x looking −x). Single-view runs are
        /// pinned to 0 — the model faces the conditioning camera by construction.
        public let viewYaws: [Float]
        /// Silhouette IoU per view at its yaw (−1 when scoring was skipped).
        public let viewIoUs: [Float]
        /// The base seed that produced the kept candidate — reruns reproduce it.
        public let seedUsed: UInt64
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

    /// Single-image reference path — see the multi-view overload for the parameters.
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
        try generate(imageURLs: [imageURL], steps: steps, outPLY: outPLY,
                     preprocessedRGBAs: [preprocessedRGBA],
                     cfgScale: cfgScale,
                     occupancyThreshold: occupancyThreshold,
                     opacityThreshold: opacityThreshold,
                     minComponentFraction: minComponentFraction,
                     seed: seed, seedCandidates: seedCandidates,
                     outSplatPLY: outSplatPLY, progress: progress,
                     onStepPreview: onStepPreview, onStepCloud: onStepCloud).pointCount
    }

    /// Run the full pipeline. `steps` = number of ODE timesteps (Heun: steps−1 corrector steps).
    /// `progress(fraction, stage)` is called as it advances.
    /// `preprocessedRGBAs` aligns with `imageURLs`; a non-nil entry points to a pre-made
    /// RGBA PNG (from RMBG 2.0) and native Vision masking is skipped for that view.
    /// `multiViewMode` picks how several views condition the DiT (ignored for one view).
    /// `occupancyThreshold` (logit; 0 = sigmoid 0.5, reference) raises the bar for a voxel to
    /// count as occupied — higher prunes low-confidence "ghost" geometry. `opacityThreshold`
    /// drops gaussians below that opacity from the cloud (prunes floaters).
    /// `minComponentFraction` drops disconnected occupancy islands smaller than that fraction
    /// of the largest connected component (floater blobs). `seed` makes runs reproducible.
    /// Seed candidates are scored by silhouette IoU averaged over *all* views (each at its
    /// best-matching yaw) — the multi-view consistency signal is much stronger than one view.
    /// `outSplatPLY` additionally writes the full gaussians as a standard 3DGS splat PLY —
    /// the high-fidelity artifact (the point cloud is just a preview format).
    /// `onStepPreview` is called after each sampling step with the step index.
    /// `onStepCloud` (optional) additionally receives an *intermediate occupancy cloud* a
    /// few times per candidate — world-space voxel centers (count·3 floats, z-up) decoded
    /// from the running sample's final-state prediction. Drives the live "dots assembling
    /// into the shape" preview; costs one voxel+occupancy decode per emission.
    @discardableResult
    public func generate(imageURLs: [URL], steps: Int, outPLY: URL,
                         preprocessedRGBAs: [URL?]? = nil,
                         cfgScale: Float = 3.0,
                         multiViewMode: MultiViewMode = .multidiffusion,
                         occupancyThreshold: Float = 0,
                         opacityThreshold: Float = 0.10,
                         minComponentFraction: Float = 0.01,
                         seed: UInt64? = nil,
                         seedCandidates: Int = 1,
                         outSplatPLY: URL? = nil,
                         onEvent: ((EngineEvent) -> Void)? = nil,
                         cancel: GenCancelToken? = nil,
                         progress: @escaping (Double, String) -> Void,
                         onStepPreview: ((Int, Int) -> Void)? = nil,
                         onStepCloud: (([Float], Int, Int) -> Void)? = nil) throws -> GenResult {
        let res = 518
        let nViews = imageURLs.count
        guard nViews > 0 else { throw EngineError.noViews }

        // ── Per-view conditioning: preprocess + DINOv2 encode ──
        var conds = [MLXArray]()
        var masks = [[Bool]]()                  // 64² alpha silhouettes for scoring
        let S = 64
        for (v, url) in imageURLs.enumerated() {
            // Any cancel during preprocessing is immediate — there is no candidate
            // to finish yet.
            if cancel?.isRequested == true { throw EngineError.cancelled }
            let tag = nViews > 1 ? " — view \(v + 1)/\(nViews)" : ""
            progress(0.02 + 0.10 * Double(v) / Double(nViews), "Preprocessing image (\(res)²)\(tag)")
            onEvent?(.viewPreprocessing(view: v))
            let condRGBA: MLXArray
            if let rgba = preprocessedRGBAs?[v] {
                condRGBA = try Preprocess.condRGBA(rgbaURL: rgba, resolution: res)
            } else {
                condRGBA = try Preprocess.condRGBA(imageURL: url, resolution: res)
            }
            masks.append(Self.alphaMask(condRGBA: condRGBA, S: S))
            progress(0.04 + 0.12 * Double(v + 1) / Double(nViews), "Encoding image (DINOv2)\(tag)")
            onEvent?(.viewEncoding(view: v))
            let c = dino(condRGBA: condRGBA)
            c.eval()
            conds.append(c)
            onEvent?(.viewEncoded(view: v, tokens: c.dim(1), dim: c.dim(2)))
            Memory.clearCache()
        }

        var ts = [Float](repeating: 0, count: max(steps, 2))
        for i in ts.indices { ts[i] = Self.tEps + (1 - Self.tEps) * Float(i) / Float(ts.count - 1) }

        // Sample 1..N seeds; pick the candidate whose occupancy silhouette best matches the
        // conditioning mask(s) (the paper shows the sampler converges by ~25 steps — remaining
        // variance is seed luck, which this search turns into a quality knob).
        let nCand = max(1, seedCandidates)
        let baseSeed = seed ?? UInt64.random(in: 0 ..< .max)
        var latent = MLXArray(0), coords = MLXArray(0), n = 0
        var bestScore: Float = -1
        var viewYaws = [Float](repeating: 0, count: nViews)
        var viewIoUs = [Float](repeating: -1, count: nViews)
        let needScore = nCand > 1 || nViews > 1
        for c in 0 ..< nCand {
            MLXRandom.seed(baseSeed &+ UInt64(c) &* 7919)
            let x0 = MLXRandom.normal([1, 8192, 32])
            let candLabel = nCand > 1 ? " (candidate \(c + 1)/\(nCand))" : ""
            let viewLabel = nViews > 1 ? " · \(nViews) views (\(multiViewMode.rawValue))" : ""
            let f0 = 0.18 + 0.56 * Double(c) / Double(nCand)
            let fw = 0.56 / Double(nCand)
            // Live cloud preview: decode the predicted-final latent's occupancy as the
            // sampler runs. minComponentFraction 0 keeps the stray specks — the
            // coalescing fuzz is the point of the visual. Early steps may legitimately
            // decode to nothing (the dots genuinely grow in as the shape forms).
            // Cadence: step 1 always, every 2nd step while coalescing, every step in
            // the final 30 % — each emission costs one voxel+occupancy decode.
            let onStepSample: ((Int, Int, MLXArray) -> Void)? = onStepCloud == nil ? nil : { done, total, xPred in
                let finalStretch = done >= Int(Double(total) * 0.7)
                guard done == 1 || done == total || finalStretch || done % 2 == 0 else { return }
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
            let sampled = dit.sample(x0: x0, ts: ts, conds: conds, cfgScale: cfgScale,
                                     mode: multiViewMode, onStep: { done, total in
                progress(f0 + fw * Double(done) / Double(total), "Sampling shape — step \(done)/\(total)\(candLabel)\(viewLabel)")
                onEvent?(.samplingStep(candidate: c + 1, candidates: nCand, step: done, total: total))
                onStepPreview?(done, total)
            }, onStepSample: onStepSample,
               shouldStop: cancel == nil ? nil : { cancel!.isImmediate })
            // Immediate stop mid-candidate: the half-integrated latent is not a
            // result — discard it rather than decode something that looks finished.
            if cancel?.isImmediate == true { throw EngineError.cancelled }
            let candLatent = sampled * Self.latentStd + Self.latentMean
            candLatent.eval(); Memory.clearCache()

            let ss = voxel(latent: candLatent); ss.eval()
            let logit = trellis(ssLatent: ss); logit.eval(); Memory.clearCache()
            let (candCoords, candN) = trellis.initCoords(logit: logit, threshold: occupancyThreshold,
                                                         minComponentFraction: minComponentFraction)
            if !needScore {
                onEvent?(.candidateDone(candidate: c + 1, candidates: nCand, meanIoU: nil))
                latent = candLatent; coords = candCoords; n = candN
                break
            }
            guard candN > 0 else {
                onEvent?(.candidateDone(candidate: c + 1, candidates: nCand, meanIoU: nil))
                if cancel?.isRequested == true { throw EngineError.cancelled }
                continue
            }

            // Score: mean silhouette IoU over views. A single view is pinned to yaw 0
            // (the model faces the conditioning camera by construction). With several
            // views nothing fixes the generated orientation — and no view is "the
            // front" a priori — so every view gets the 5°-grid yaw sweep; view order
            // doesn't matter.
            let pts = candCoords.asType(.float32).asArray(Float.self)
            var yaws = [Float](repeating: 0, count: nViews)
            var ious = [Float](repeating: 0, count: nViews)
            for v in 0 ..< nViews {
                if nViews == 1 {
                    ious[v] = Self.silhouetteIoU(points: pts, count: candN, mask: masks[v], S: S, yaw: 0)
                } else {
                    (yaws[v], ious[v]) = Self.bestYaw(points: pts, count: candN, mask: masks[v], S: S)
                }
            }
            let score = ious.reduce(0, +) / Float(nViews)
            if nViews > 1 {
                let detail = (0 ..< nViews).map { v in
                    String(format: "v%d %.0f° %.3f", v + 1, yaws[v] * 180 / .pi, ious[v])
                }.joined(separator: "  ")
                progress(f0 + fw, String(format: "Candidate %d/%d — mean IoU %.3f [%@]", c + 1, nCand, score, detail))
            } else {
                progress(f0 + fw, String(format: "Candidate %d/%d — silhouette IoU %.3f", c + 1, nCand, score))
            }
            onEvent?(.candidateDone(candidate: c + 1, candidates: nCand, meanIoU: score))
            if score > bestScore {
                bestScore = score
                latent = candLatent; coords = candCoords; n = candN
                viewYaws = yaws; viewIoUs = ious
            }
            // Finish-candidate stop: keep the best candidate so far and decode it
            // instead of starting the next seed.
            if cancel?.isRequested == true { break }
            if nCand == 1 { break }
        }
        guard n > 0 else { throw EngineError.emptyShape }
        progress(0.78, needScore ? String(format: "Best mean IoU %.3f — decoding", bestScore) : "Decoding structure")
        onEvent?(.decoding)

        progress(0.88, "Decoding \(n) gaussians")
        let gs = gauss(latent: latent, initCoord: coords)
        gs.values.forEach { $0.eval() }; Memory.clearCache()
        onEvent?(.decodedGaussians(count: n))

        progress(0.96, "Writing splat")
        onEvent?(.writingOutput)
        if let splatURL = outSplatPLY {
            try Splat.writeGaussians(gs, to: splatURL)
        }
        let count = try Splat.writePointCloud(gs, to: outPLY, opacityThreshold: opacityThreshold)
        progress(1.0, "Done")
        return GenResult(pointCount: count, viewYaws: viewYaws, viewIoUs: viewIoUs,
                         seedUsed: baseSeed)
    }

    // MARK: - silhouette scoring

    /// Downsampled boolean alpha silhouette of a cond view (S×S, row 0 = top).
    static func alphaMask(condRGBA: MLXArray, S: Int) -> [Bool] {
        let res = condRGBA.dim(0)
        let alpha: [Float] = condRGBA[0 ..< res, 0 ..< res, 3 ..< 4].reshaped([res * res]).asArray(Float.self)
        var mask = [Bool](repeating: false, count: S * S)
        for y in 0 ..< S { for x in 0 ..< S {
            if alpha[(y * res / S) * res + x * res / S] > 0.5 { mask[y * S + x] = true }
        }}
        return mask
    }

    /// IoU between the occupancy silhouette seen from a camera at azimuth `yaw` and a
    /// view's alpha mask. The conditioning camera (yaw 0) is orthographic along +x with
    /// v = z up, u = y (pinned empirically via `LiToSmoke score`); a camera at azimuth θ
    /// has right = (−sinθ, cosθ, 0), so u = y·cosθ − x·sinθ, v = z.
    static func silhouetteIoU(points: [Float], count: Int, mask: [Bool], S: Int, yaw: Float) -> Float {
        let cosT = cos(yaw), sinT = sin(yaw)
        var sil = [Bool](repeating: false, count: S * S)
        for i in 0 ..< count {
            let u = points[i * 3 + 1] * cosT - points[i * 3] * sinT
            let v = points[i * 3 + 2]
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

    /// Sweep the camera azimuth on a 5° grid and return the best-matching yaw + IoU.
    static func bestYaw(points: [Float], count: Int, mask: [Bool], S: Int) -> (yaw: Float, iou: Float) {
        var best: (yaw: Float, iou: Float) = (0, -1)
        var deg = 0
        while deg < 360 {
            let yaw = Float(deg) * .pi / 180
            let iou = silhouetteIoU(points: points, count: count, mask: mask, S: S, yaw: yaw)
            if iou > best.iou { best = (yaw, iou) }
            deg += 5
        }
        return best
    }
}
