import Foundation
import CoreGraphics
import ImageIO
import LiToKit

enum Config {
    /// Every place a weights directory may live, in resolution order. Re-evaluated on
    /// each access (cheap stats) so first-run setup can install into one of them and
    /// have the engine pick it up without relaunching.
    private static func weightsCandidates() -> [URL] {
        var out = [URL]()
        // 1. Explicit override — lets you point anywhere without rebuilding.
        if let env = ProcessInfo.processInfo.environment["LITO_WEIGHTS_DIR"], !env.isEmpty {
            out.append(URL(filePath: env))
        }
        // 2. weights/ sitting next to the executable or one level up (repo checkout
        //    via `swift run`, or a .app with weights copied alongside it).
        let exeDir = Bundle.main.bundleURL.deletingLastPathComponent()
        out.append(exeDir.appending(path: "weights"))
        out.append(exeDir.deletingLastPathComponent().appending(path: "weights"))
        // 3. Walk up from the executable looking for a repo-level weights/ dir.
        //    Covers `.build/<triple>/debug/` under SwiftPM and DerivedData layouts.
        var dir = Bundle.main.bundleURL
        for _ in 0..<8 {
            out.append(dir.appending(path: "weights"))
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }
        // 4. App's Resources — either the weights laid down directly, or a
        //    `weights/` symlink the Xcode build phase drops in (points at the checkout).
        if let res = Bundle.main.resourceURL {
            out.append(res)
            out.append(res.appending(path: "weights"))
        }
        // 5. Per-user install target — where first-run setup puts everything when
        //    no checkout-level weights dir exists (e.g. a bare downloaded .app).
        out.append(appSupportWeights)
        return out
    }

    private static func hasWeights(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appending(path: "lito.safetensors").path)
    }

    static var appSupportWeights: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "LiToStudio/weights")
    }

    static var weightsDir: URL {
        let candidates = weightsCandidates()
        for c in candidates where hasWeights(c) { return c }
        return candidates[0]
    }

    /// Where first-run setup installs: the override dir, an existing writable
    /// weights/ dir near the code (dev checkout), else per-user Application Support.
    static var installDir: URL {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["LITO_WEIGHTS_DIR"], !env.isEmpty {
            let dir = URL(filePath: env)
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }
        for c in weightsCandidates() where c.lastPathComponent == "weights" {
            var isDir: ObjCBool = false
            let resolved = c.resolvingSymlinksInPath()
            if fm.fileExists(atPath: resolved.path, isDirectory: &isDir), isDir.boolValue,
               fm.isWritableFile(atPath: resolved.path) {
                return resolved
            }
        }
        let dir = appSupportWeights
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// MLX refuses to start unless `mlx.metallib` sits next to the executable.
    /// run.sh and the Xcode post-build step normally handle that; after first-run
    /// setup (or for a bare .app) the app does it itself.
    @discardableResult
    static func ensureMetallibColocated() -> Bool {
        let fm = FileManager.default
        guard let exe = Bundle.main.executableURL?.resolvingSymlinksInPath() else { return false }
        let dst = exe.deletingLastPathComponent().appending(path: "mlx.metallib")
        if fm.fileExists(atPath: dst.path) { return true }
        let src = weightsDir.appending(path: "mlx.metallib")
        guard fm.fileExists(atPath: src.path) else { return false }
        do { try fm.copyItem(at: src, to: dst); return true } catch { return false }
    }
    static var outputDir: URL {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "LiToStudio/results")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    static var ready: Bool {
        FileManager.default.fileExists(atPath: weightsDir.appending(path: "lito.safetensors").path)
    }
    static var rmbgModelURL: URL { weightsDir.appending(path: "RMBG2.mlpackage") }
    /// Resolve the upscaler regardless of packaging (.mlpackage / .mlmodelc / .mlmodel).
    static var upscalerModelURL: URL? {
        for ext in ["mlpackage", "mlmodelc", "mlmodel"] {
            let u = weightsDir.appending(path: "RealESRGAN_x4.\(ext)")
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        return nil
    }
    static var rmbgReady: Bool { FileManager.default.fileExists(atPath: rmbgModelURL.path) }
    static var upscalerReady: Bool { upscalerModelURL != nil }
    /// Sapiens normal estimator (optional) — enables photo-measured mesh refinement.
    static var sapiensModelURL: URL? { SapiensNormal.locate(weightsDir: weightsDir) }

    /// Native SAM 3.1 CoreML packages — checked in the active weights dir AND the
    /// repo weights dir (the engine weights often live in App Support while the
    /// sam3-coreml download sits in the checkout's weights/).
    static var sam3CoreMLDir: URL? {
        if let d = Sam3CoreML.locate(weightsDir: weightsDir) { return d }
        if let backend = PythonBackend.backendDir {
            let repoWeights = backend.deletingLastPathComponent().deletingLastPathComponent()
                .appending(path: "weights")
            return Sam3CoreML.locate(weightsDir: repoWeights)
        }
        return nil
    }
}

enum RunEvent: Sendable {
    case progress(stage: String, fraction: Double)
    case line(String)
    case result(pointCloud: URL, splat: URL?, mesh: URL?)
    case failed(String)
    /// The run stopped on user request without producing a result (immediate stop).
    /// Finish-candidate stops still yield `.result` for the kept candidate.
    case cancelled
    case preview(label: String, imageURL: URL)
    /// Intermediate occupancy decode while sampling — flat [x,y,z]* world coords (z-up).
    case cloud(points: [Float], step: Int, total: Int)
    /// Structured progress-tree update (per-view branch or shared trunk).
    case stage(StageUpdate)
    /// The landmark package updated with REAL backend outputs (SAM3 / Sapiens2).
    case landmarks(LandmarkPackage)
    /// Sampling position — feeds the Stop button's "almost complete" decision.
    case sampling(candidate: Int, candidates: Int, step: Int, total: Int)
}

enum StageStatus: String, Sendable {
    case pending, running, done, failed, skipped, unavailable
}

/// One progress-tree node update. `view == nil` addresses the shared trunk
/// (conditioning merge → sampling candidates → decode → outputs); otherwise the
/// branch of that input view. Stages never silently disappear: absent models are
/// `unavailable`, not-run work is `skipped` with the reason in `detail`.
struct StageUpdate: Sendable {
    let view: Int?
    let stage: String          // stable id within its branch (e.g. "dino", "cand2")
    let label: String
    let status: StageStatus
    var thumbnail: URL? = nil
    var dims: String? = nil
    var detail: String? = nil
}

struct PipelineArgs: Sendable {
    /// One image = the reference path. Several images of the same subject from
    /// different angles condition one shape together. A single image that turns out
    /// to be a contact sheet (several figures on a uniform background) is split
    /// automatically when `splitSheet` is on.
    var imagePaths: [String]
    var steps: Int
    var cfgScale: Float = 3.0
    var multiView: MultiViewMode = .multidiffusion
    var splitSheet: Bool = true
    /// Replace generated splat/mesh colors with colors backprojected from the photos
    /// (the photos carry far more texture detail than the 64³-latent generation).
    /// Applies when ≥2 views are available.
    var photoTexture: Bool = true
    var useRMBG: Bool = true
    var useUpscaler: Bool = true
    var occupancyThreshold: Float = 0
    var opacityThreshold: Float = 0.10
    var seed: UInt64? = nil
    var seedCandidates: Int = 1
    var extractMesh: Bool = true
    /// The auto-detect analysis active when the run started (nil when auto was off) —
    /// persisted into the run metadata JSON.
    var analysis: ImageAnalyzer.Analysis? = nil
    /// Landmark conditioning package for the selection — exported as
    /// `<base>_landmarks.json`. NOT consumed by generation (no grounding backend, no
    /// auxiliary DiT conditioning channel); see LandmarkGrounding.swift.
    var landmarks: LandmarkPackage? = nil
    /// Optional user text guidance — recorded in run metadata only (the checkpoint
    /// has no text-conditioning pathway; see LITO_PROMPT_GUIDANCE_RESEARCH.md).
    var userPrompt: String = ""
}

/// Everything worth knowing about a finished run, written as `<base>_run.json` next
/// to the artifacts: the exact settings, the seed that produced the result, per-view
/// yaw/IoU estimates, and the auto-detect analysis (when auto was on).
struct RunMetadata: Codable {
    var schema = 1
    let createdAt: Date
    let inputs: [String]
    let steps: Int
    let cfgScale: Float
    let multiViewMode: String
    let occupancyThreshold: Float
    let opacityThreshold: Float
    let seedCandidates: Int
    let seedUsed: UInt64
    let useRMBG: Bool
    let useUpscaler: Bool
    let extractMesh: Bool
    let viewYaws: [Float]
    let viewIoUs: [Float]
    let pointCloud: String
    let splat: String
    let mesh: String?
    let analysis: ImageAnalyzer.Analysis?
    /// Exported landmark package file name (nil when no package was built).
    let landmarkPackageFile: String?
    /// Optional user text guidance — recorded only; not consumed by generation.
    let userPrompt: String?
    /// Per-view cutout artifact file names (nil where no cutout was produced).
    let cutouts: [String?]
    /// What actually ran for each model backend this run.
    let rmbgStatus: String
    let sam3Status: String
    let sapiensPoseStatus: String
}

nonisolated(unsafe) private var sharedEngine: LiToEngine?
nonisolated(unsafe) private var sharedRMBG: RMBG?
nonisolated(unsafe) private var sharedUpscaler: Upscaler?
nonisolated(unsafe) private var sharedSapiens: SapiensNormal?
private let engineLock = NSLock()

func runPipeline(_ args: PipelineArgs, cancel: GenCancelToken? = nil) -> AsyncStream<RunEvent> {
    AsyncStream(bufferingPolicy: .unbounded) { continuation in
        let thread = Thread {
            guard !args.imagePaths.isEmpty else {
                continuation.yield(.failed("No input image")); continuation.finish(); return
            }
            var inputURLs = args.imagePaths.map { URL(filePath: $0) }
            var tempFiles = [URL]()
            let stamp = Int(Date().timeIntervalSince1970)
            continuation.yield(.preview(label: "Input", imageURL: inputURLs[0]))

            func stage(_ view: Int?, _ id: String, _ label: String, _ status: StageStatus,
                       thumb: URL? = nil, dims: String? = nil, detail: String? = nil) {
                continuation.yield(.stage(StageUpdate(view: view, stage: id, label: label,
                                                      status: status, thumbnail: thumb,
                                                      dims: dims, detail: detail)))
            }

            // Phase 0: contact-sheet split — one dropped image that contains several
            // renders of the subject (AI-generated turnaround) becomes N view images.
            if args.splitSheet, inputURLs.count == 1 {
                if sharedRMBG == nil, args.useRMBG, Config.rmbgReady {
                    sharedRMBG = try? RMBG(modelURL: Config.rmbgModelURL)
                }
                let splitDir = FileManager.default.temporaryDirectory.appending(path: "lito_sheet_\(stamp)")
                if let views = try? SheetSplit.splitToFiles(
                    imageURL: inputURLs[0], rmbg: sharedRMBG, outDir: splitDir,
                    log: { continuation.yield(.line("[Sheet] \($0)")) }) {
                    inputURLs = views
                    tempFiles += views
                    tempFiles.append(splitDir)
                    for (i, v) in views.enumerated() {
                        continuation.yield(.preview(label: "View \(i + 1)", imageURL: v))
                    }
                    continuation.yield(.line("[Sheet] contact sheet split into \(views.count) views"))
                }
            }

            // Artifact base name — needed early so cutouts/masks/pose files land in
            // the results dir with the run's name.
            let stem = URL(filePath: args.imagePaths[0]).deletingPathExtension().lastPathComponent
            let safe = String(stem.prefix(40).map { $0.isLetter || $0.isNumber ? $0 : "_" })
            let base = "\(safe.isEmpty ? "img" : safe)_\(stamp)"

            // ── Progress-tree skeleton: every view's branch + the shared trunk, so
            // the UI shows the whole pipeline up front. Absent models are marked
            // unavailable (never silently done); disabled work is skipped.
            let nViews = inputURLs.count
            for v in 0 ..< nViews {
                let dims = imagePixelSize(inputURLs[v]).map { "\($0.0)×\($0.1)" }
                stage(v, "original", "Original", .done, thumb: inputURLs[v], dims: dims)
                if !args.useUpscaler {
                    stage(v, "upscale", "Upscale", .skipped, detail: "disabled in settings")
                } else if Config.upscalerModelURL == nil {
                    stage(v, "upscale", "Upscale", .unavailable, detail: "Real-ESRGAN model not installed")
                } else {
                    stage(v, "upscale", "Upscale", .pending)
                }
                if !args.useRMBG {
                    stage(v, "background", "Background removal", .skipped, detail: "disabled in settings")
                } else if Config.rmbgReady || RMBGWorkerBackend.isAvailable {
                    stage(v, "background", "Background removal", .pending)
                } else {
                    stage(v, "background", "Background removal", .unavailable,
                          detail: "RMBG-2.0 not installed — run tools/backend/convert_rmbg2.py (CoreML) or download briaai/RMBG-2.0 for the Python worker; Vision fallback during crop")
                }
                stage(v, "crop", "Crop · normalize", .pending)
                stage(v, "dino", "DINOv2 features", .pending)
                if SapiensPoseBackend.isAvailable {
                    stage(v, "sapiens", "Sapiens2 pose", .pending)
                } else {
                    stage(v, "sapiens", "Sapiens2 pose", .unavailable,
                          detail: SapiensPoseBackend.unavailableReason)
                }
                if Config.sam3CoreMLDir != nil || Sam3Backend.isAvailable {
                    stage(v, "sam3", "SAM3 landmarks", .pending)
                } else {
                    stage(v, "sam3", "SAM3 landmarks", .unavailable,
                          detail: "no SAM3 backend — install the CoreML packages (weights/sam3-coreml) or the gated facebook/sam3 worker")
                }
                stage(v, "token", "View token", .pending)
            }
            let mergeLabel = nViews > 1 ? "Multi-view conditioning" : "Conditioning"
            stage(nil, "merge", mergeLabel, .pending)
            stage(nil, "select", "Best candidate", .pending)
            stage(nil, "gauss", "Gaussian decode", .pending)
            if args.photoTexture, nViews > 1 {
                stage(nil, "texture", "Photo texture", .pending)
            } else {
                stage(nil, "texture", "Photo texture", .skipped,
                      detail: nViews > 1 ? "disabled in settings" : "needs ≥ 2 views")
            }
            stage(nil, "mesh", "Mesh extraction", args.extractMesh ? .pending : .skipped,
                  detail: args.extractMesh ? nil : "disabled in settings")
            stage(nil, "result", "Result", .pending)

            // Phases 0a–0c per view: low-light, upscale, background removal.
            var workingURLs = inputURLs
            var preRGBAs = [URL?](repeating: nil, count: nViews)
            for v in 0 ..< nViews {
                // Stop requested during preprocessing: nothing worth keeping yet.
                if cancel?.isRequested == true {
                    for tmp in tempFiles { try? FileManager.default.removeItem(at: tmp) }
                    continuation.yield(.cancelled)
                    continuation.finish()
                    return
                }
                let vtag = nViews > 1 ? " — view \(v + 1)/\(nViews)" : ""
                let vfile = nViews > 1 ? "v\(v + 1)_" : ""

                // Phase 0a0: Low-light normalization (auto — night shots starve DINOv2)
                if let cg = Preprocess.loadCGImageUpright(workingURLs[v]) {
                    let lum = Preprocess.meanLuminance(cg)
                    if lum < 0.27, let fixed = Preprocess.normalizeLowLight(cg) {
                        let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_lowlight_\(vfile)\(stamp).png")
                        if (try? writeCGImage(fixed, to: tmp)) != nil {
                            workingURLs[v] = tmp
                            tempFiles.append(tmp)
                            continuation.yield(.preview(label: "Low-light normalized", imageURL: tmp))
                            continuation.yield(.line(String(format: "[LowLight]%@ mean luminance %.2f → normalized", vtag, lum)))
                            stage(v, "original", "Original", .done,
                                  detail: String(format: "low-light normalized (ℓ = %.2f)", lum))
                        }
                    }
                }

                // Phase 0a: Upscale to the 2K subject target (Decision 006: "2K" =
                // estimated subject long side ≥ 2048 px — conditioning crops to the
                // subject, so canvas pixels don't count). Up to two Real-ESRGAN
                // passes (a cascade is real super-resolution, never plain resizing);
                // canvas capped at 4096 px for memory; alpha survives via the
                // alpha-preserving path.
                if args.useUpscaler, let upscalerURL = Config.upscalerModelURL {
                    do {
                        continuation.yield(.progress(stage: "Upscaling (Real-ESRGAN 4x)\(vtag)…",
                                                     fraction: 0.01 + 0.02 * Double(v) / Double(nViews)))
                        guard let cg = Preprocess.loadCGImageUpright(workingURLs[v]) else {
                            throw Upscaler.UpscalerError.image
                        }
                        let target = ImageAnalyzer.subjectTargetPx
                        let cap = ImageAnalyzer.canvasCapPx
                        let canvasLong = max(cg.width, cg.height)
                        // Subject estimate (alpha bbox / border-color heuristic);
                        // falls back to the canvas when the background is too busy.
                        let subjBefore = ImageAnalyzer.subjectBoxEstimate(cg)?.longSide ?? canvasLong

                        if subjBefore >= target, ImageAnalyzer.measuredSharpness(cg) >= 0.20 {
                            continuation.yield(.line("[Upscaler]\(vtag) skipped — subject ~\(subjBefore) px already ≥ \(target) and sharp"))
                            stage(v, "upscale", "Upscale", .skipped,
                                  dims: "\(cg.width)×\(cg.height)",
                                  detail: "subject ~\(subjBefore) px ≥ \(target) and sharp — no upscale needed")
                        } else if canvasLong >= cap {
                            continuation.yield(.line("[Upscaler]\(vtag) WARNING: canvas at the \(cap) px cap but subject only ~\(subjBefore) px — a tighter crop would condition better"))
                            stage(v, "upscale", "Upscale", .skipped,
                                  dims: "\(cg.width)×\(cg.height)",
                                  detail: "canvas at \(cap) px cap, subject ~\(subjBefore) px < \(target) — consider a tighter crop")
                        } else {
                            stage(v, "upscale", "Upscale", .running)
                            if sharedUpscaler == nil {
                                sharedUpscaler = try Upscaler(modelURL: upscalerURL)
                            }
                            var upscaled = try sharedUpscaler!.upscaleToMaxPreservingAlpha(cg, maxDim: cap)
                            var passes = 1
                            var subjAfter = subjBefore * max(upscaled.width, upscaled.height) / canvasLong
                            // Second pass only when one 4× still leaves the subject
                            // under 2K and the canvas has headroom (tiny sources).
                            if subjAfter < target, max(upscaled.width, upscaled.height) < cap {
                                upscaled = try sharedUpscaler!.upscaleToMaxPreservingAlpha(upscaled, maxDim: cap)
                                subjAfter = subjBefore * max(upscaled.width, upscaled.height) / canvasLong
                                passes = 2
                            }
                            let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_upscaled_\(vfile)\(stamp).png")
                            try writeCGImage(upscaled, to: tmp)
                            workingURLs[v] = tmp
                            tempFiles.append(tmp)
                            var note = "subject ~\(subjBefore) → ~\(subjAfter) px (\(passes) pass\(passes > 1 ? "es" : ""))"
                            if subjAfter < target {
                                note += " — source-limited, below the \(target) px target"
                                continuation.yield(.line("[Upscaler]\(vtag) WARNING: subject still ~\(subjAfter) px after \(passes) pass(es) — quality is limited by the source image"))
                            }
                            continuation.yield(.preview(label: "Upscaled (\(upscaled.width)×\(upscaled.height))", imageURL: tmp))
                            continuation.yield(.line("[Upscaler]\(vtag) \(cg.width)×\(cg.height) → \(upscaled.width)×\(upscaled.height), \(note)"))
                            stage(v, "upscale", "Upscale", .done, thumb: tmp,
                                  dims: "\(cg.width)×\(cg.height) → \(upscaled.width)×\(upscaled.height)",
                                  detail: note)
                        }
                    } catch {
                        continuation.yield(.line("[Upscaler]\(vtag) skipped: \(error)"))
                        stage(v, "upscale", "Upscale", .skipped, detail: "error: \(error)")
                    }
                }

            }

            // Phase 0b: Background removal — CoreML RMBG2 when installed, else the
            // Python RMBG worker (same briaai/RMBG-2.0 weights, PyTorch, one process
            // for the whole batch). Cutouts are persisted as run artifacts; photo
            // texture depends on them.
            var cutoutFiles = [String?](repeating: nil, count: nViews)
            var rmbgStatus = args.useRMBG ? "unavailable — RMBG-2.0 not installed" : "disabled in settings"
            func saveCutoutArtifact(_ v: Int, from url: URL) {
                let dst = Config.outputDir.appending(path: "\(base)_v\(v + 1)_cutout.png")
                try? FileManager.default.removeItem(at: dst)
                if (try? FileManager.default.copyItem(at: url, to: dst)) != nil {
                    cutoutFiles[v] = dst.lastPathComponent
                }
            }
            if args.useRMBG, Config.rmbgReady, cancel?.isRequested != true {
                rmbgStatus = "RMBG-2.0 (CoreML, weights/RMBG2.mlpackage)"
                for v in 0 ..< nViews {
                    if cancel?.isRequested == true { break }
                    let vtag = nViews > 1 ? " — view \(v + 1)/\(nViews)" : ""
                    let vfile = nViews > 1 ? "v\(v + 1)_" : ""
                    do {
                        continuation.yield(.progress(stage: "Background removal (RMBG 2.0)\(vtag)…",
                                                     fraction: 0.03 + 0.02 * Double(v) / Double(nViews)))
                        stage(v, "background", "Background removal", .running)
                        if sharedRMBG == nil {
                            sharedRMBG = try RMBG(modelURL: Config.rmbgModelURL)
                        }
                        guard let cg = Preprocess.loadCGImageUpright(workingURLs[v]) else {
                            throw RMBG.RMBGError.compile
                        }
                        var rgba = try sharedRMBG!.removeBackground(from: cg)
                        // Trim the cutout to Vision's person mask (dilated) — drops mirror
                        // frames / furniture that RMBG keeps because they touch the subject.
                        var trimmedNote: String?
                        if let trimmed = Preprocess.personTrim(rgba: rgba, original: cg) {
                            rgba = trimmed
                            trimmedNote = "trimmed to person mask"
                            continuation.yield(.line("[PersonTrim]\(vtag) cutout trimmed to person mask"))
                        }
                        let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_rmbg_\(vfile)\(stamp).png")
                        try writeCGImage(rgba, to: tmp)
                        preRGBAs[v] = tmp
                        tempFiles.append(tmp)
                        saveCutoutArtifact(v, from: tmp)
                        continuation.yield(.preview(label: nViews > 1 ? "View \(v + 1) cutout" : "Background Removed", imageURL: tmp))
                        continuation.yield(.line("[RMBG]\(vtag) background removed"))
                        stage(v, "background", "Background removal", .done, thumb: tmp,
                              dims: "\(rgba.width)×\(rgba.height)",
                              detail: [trimmedNote, cutoutFiles[v].map { "→ \($0)" }]
                                  .compactMap { $0 }.joined(separator: " · "))
                    } catch {
                        rmbgStatus = "CoreML error: \(error)"
                        continuation.yield(.line("[RMBG]\(vtag) skipped: \(error) — using Vision fallback"))
                        stage(v, "background", "Background removal", .skipped,
                              detail: "error — Apple Vision fallback during crop")
                    }
                }
            } else if args.useRMBG, RMBGWorkerBackend.isAvailable, cancel?.isRequested != true {
                for v in 0 ..< nViews { stage(v, "background", "Background removal", .running) }
                continuation.yield(.progress(stage: "Background removal (RMBG-2.0, Python backend)…",
                                             fraction: 0.035))
                do {
                    let dir = FileManager.default.temporaryDirectory.appending(path: "lito_rmbgw_\(stamp)")
                    let res = try RMBGWorkerBackend.cutouts(images: workingURLs, outDir: dir, cancel: cancel)
                    rmbgStatus = res.backend
                    tempFiles.append(dir)
                    for v in 0 ..< nViews {
                        guard let cut = res.cutouts[v] else {
                            stage(v, "background", "Background removal", .failed,
                                  detail: "no cutout produced")
                            continue
                        }
                        preRGBAs[v] = cut
                        saveCutoutArtifact(v, from: cut)
                        continuation.yield(.preview(label: nViews > 1 ? "View \(v + 1) cutout" : "Background Removed", imageURL: cut))
                        stage(v, "background", "Background removal", .done, thumb: cut,
                              detail: cutoutFiles[v].map { "→ \($0)" })
                    }
                    continuation.yield(.line("[RMBG] \(res.backend): \(res.cutouts.compactMap { $0 }.count)/\(nViews) cutouts"))
                } catch {
                    rmbgStatus = "Python worker failed: \(error)"
                    for v in 0 ..< nViews {
                        stage(v, "background", "Background removal", .failed, detail: "\(error)")
                    }
                    continuation.yield(.line("[RMBG] worker FAILED: \(error) — Vision fallback during crop"))
                }
            }

            // ── Grounding phase: real SAM3 detections + Sapiens2 pose. Workers are
            // separate processes that exit before MLX loads, so their memory never
            // overlaps sampling. The package keeps taxonomy priors only where no
            // backend actually ran; sheet-split runs skip grounding (the pre-split
            // package doesn't align with the split views).
            var finalLandmarks = args.landmarks
            let sam3CoreMLDir = Config.sam3CoreMLDir
            var sam3Status = sam3CoreMLDir != nil ? "available (CoreML)"
                : Sam3Backend.isAvailable ? "available (python worker)" : Sam3Backend.unavailableReason
            var sapiensStatus = SapiensPoseBackend.isAvailable ? "available" : SapiensPoseBackend.unavailableReason
            if let pkg = args.landmarks, pkg.views.count == nViews {
                let labels = pkg.views.map(\.viewLabel)
                var sam3Result: Sam3RunResult?
                if sam3CoreMLDir != nil || Sam3Backend.isAvailable, cancel?.isRequested != true {
                    for v in 0 ..< nViews { stage(v, "sam3", "SAM3 landmarks", .running) }
                    continuation.yield(.progress(stage: "Grounding landmarks (SAM3)…", fraction: 0.06))
                    do {
                        let masksDir = Config.outputDir.appending(path: "\(base)_masks")
                        let res: Sam3RunResult
                        if let dir = sam3CoreMLDir {
                            // Native path: scoped so the ~1.7 GB of CoreML models are
                            // released before the MLX engine loads.
                            res = try {
                                let sam = try Sam3CoreML(dir: dir)
                                // Cutouts give SAM3 the person silhouette: masks get
                                // cleaned of background speckle and the whole-person
                                // fallback is rejected.
                                return try sam.ground(images: workingURLs, labels: labels,
                                                      masksDir: masksDir, cutouts: preRGBAs,
                                                      cancel: cancel)
                            }()
                        } else {
                            res = try Sam3Backend.detect(images: workingURLs, labels: labels,
                                                         masksDir: masksDir, cancel: cancel)
                        }
                        sam3Result = res
                        sam3Status = res.backend
                        for v in 0 ..< nViews {
                            let det = v < res.perView.count
                                ? res.perView[v].filter { $0.status == "detected" }.count : 0
                            stage(v, "sam3", "SAM3 landmarks", .done,
                                  detail: "\(det)/\(LandmarkTaxonomy.coreSet.count) grounded")
                        }
                        continuation.yield(.line("[sam3] \(res.detectionCount) detections across \(nViews) view(s) → \(masksDir.lastPathComponent)/"))
                    } catch {
                        sam3Status = "failed: \(error)"
                        for v in 0 ..< nViews {
                            stage(v, "sam3", "SAM3 landmarks", .failed, detail: "\(error)")
                        }
                        continuation.yield(.line("[sam3] FAILED: \(error)"))
                    }
                }
                var poseRecords: [HumanPoseRecord?]?
                if SapiensPoseBackend.isAvailable, cancel?.isRequested != true {
                    for v in 0 ..< nViews { stage(v, "sapiens", "Sapiens2 pose", .running) }
                    continuation.yield(.progress(stage: "Extracting human pose (Sapiens2)…", fraction: 0.07))
                    do {
                        // Person-crop hints from the real cutout / subject estimate —
                        // the lean substitute for the upstream RTMDet detector.
                        var boxes = [[Double]?]()
                        for v in 0 ..< nViews {
                            if let cg = Preprocess.loadCGImageUpright(preRGBAs[v] ?? workingURLs[v]),
                               let box = ImageAnalyzer.subjectBoxEstimate(cg) {
                                boxes.append([Double(box.x) / Double(cg.width),
                                              Double(box.y) / Double(cg.height),
                                              Double(box.width) / Double(cg.width),
                                              Double(box.height) / Double(cg.height)])
                            } else {
                                boxes.append(nil)
                            }
                        }
                        let poseDir = Config.outputDir.appending(path: "\(base)_pose")
                        let recs = try SapiensPoseBackend.extract(images: workingURLs, labels: labels,
                                                                  subjectBoxes: boxes, outDir: poseDir,
                                                                  cancel: cancel)
                        poseRecords = recs
                        if let b = recs.compactMap({ $0 }).first?.backend { sapiensStatus = b }
                        for v in 0 ..< nViews {
                            if let r = recs[v] {
                                let hands = r.groups["hands"].map { "hands \($0.visible)/\($0.total)" }
                                stage(v, "sapiens", "Sapiens2 pose", .done,
                                      detail: ["\(r.keypointCount) keypoints", hands,
                                               r.raisedHand.map { "raised: \($0)" }]
                                          .compactMap { $0 }.joined(separator: " · "))
                            } else {
                                stage(v, "sapiens", "Sapiens2 pose", .failed, detail: "no pose result")
                            }
                        }
                        continuation.yield(.line("[sapiens2] pose extracted for \(recs.compactMap { $0 }.count)/\(nViews) view(s) → \(poseDir.lastPathComponent)/"))
                    } catch {
                        sapiensStatus = "failed: \(error)"
                        for v in 0 ..< nViews {
                            stage(v, "sapiens", "Sapiens2 pose", .failed, detail: "\(error)")
                        }
                        continuation.yield(.line("[sapiens2] FAILED: \(error)"))
                    }
                }
                if sam3Result != nil || poseRecords != nil {
                    finalLandmarks = pkg.applying(sam3: sam3Result, pose: poseRecords)
                }

                // Text guidance: segment the user's free-text phrase as a SAM 3.1
                // concept across views (real text → region). It is recorded + shown
                // but does NOT steer DiT geometry — the checkpoint has no text path.
                let phrase = args.userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !phrase.isEmpty, let dir = sam3CoreMLDir, ClipTokenizer.isAvailable,
                   cancel?.isRequested != true {
                    continuation.yield(.progress(stage: "Segmenting text concept “\(phrase)” (SAM3)…", fraction: 0.075))
                    do {
                        let ids = try ClipTokenizer.tokenize([phrase])
                        guard let tokenIds = ids.first else { throw PythonBackend.BackendError.badOutput("no tokens") }
                        let masksDir = Config.outputDir.appending(path: "\(base)_masks")
                        let obs = try {
                            let sam = try Sam3CoreML(dir: dir)
                            return try sam.groundConcept(phrase: phrase, tokenIds: tokenIds, id: "USER",
                                                         images: workingURLs, labels: labels,
                                                         masksDir: masksDir, cutouts: preRGBAs, cancel: cancel)
                        }()
                        let concept = LandmarkPackage.UserConcept(
                            phrase: phrase, backend: "SAM 3.1 CoreML",
                            perView: obs)
                        var pkg2 = finalLandmarks ?? pkg
                        pkg2.userConcept = concept
                        finalLandmarks = pkg2
                        continuation.yield(.line("[text-guide] “\(phrase)” → \(concept.detectionCount)/\(nViews) view(s)"))
                    } catch {
                        continuation.yield(.line("[text-guide] “\(phrase)” skipped: \(error)"))
                    }
                } else if !phrase.isEmpty {
                    continuation.yield(.line("[text-guide] “\(phrase)” recorded only (needs SAM 3.1 CoreML + backend venv to segment)"))
                }

                if finalLandmarks != nil, finalLandmarks?.backendAvailable == true || !phrase.isEmpty {
                    if let fl = finalLandmarks { continuation.yield(.landmarks(fl)) }
                }
            } else if args.landmarks != nil {
                continuation.yield(.line("[landmarks] view count changed after sheet split — grounding skipped for this run"))
            }

            // Phase 0c: Emit the condRGBA preview (the actual 518² input to DINOv2)
            if nViews == 1, let rgba = preRGBAs[0] {
                continuation.yield(.preview(label: "DINOv2 Input (518²)", imageURL: rgba))
            }

            // Phase 1+: LiTo engine
            let engine: LiToEngine
            do {
                engineLock.lock(); defer { engineLock.unlock() }
                if sharedEngine == nil {
                    if !Config.ensureMetallibColocated() {
                        continuation.yield(.line("[MLX] warning: mlx.metallib not colocated — GPU backend may fail to start"))
                    }
                    continuation.yield(.progress(stage: "Loading model (first run)…", fraction: 0.05))
                    sharedEngine = try LiToEngine(weightsDir: Config.weightsDir)
                }
                engine = sharedEngine!
            } catch {
                continuation.yield(.failed("Failed to load model: \(error)")); continuation.finish(); return
            }
            do {
                let out = Config.outputDir.appending(path: "\(base)_pc.ply")
                let outSplat = Config.outputDir.appending(path: "\(base)_gs.ply")
                stage(nil, "merge", mergeLabel, .running)
                let gen = try engine.generate(imageURLs: workingURLs, steps: args.steps,
                                              outPLY: out, preprocessedRGBAs: preRGBAs,
                                              cfgScale: args.cfgScale,
                                              multiViewMode: args.multiView,
                                              occupancyThreshold: args.occupancyThreshold,
                                              opacityThreshold: args.opacityThreshold,
                                              seed: args.seed,
                                              seedCandidates: args.seedCandidates,
                                              outSplatPLY: outSplat,
                                              onEvent: { ev in
                    switch ev {
                    case .viewPreprocessing(let v):
                        stage(v, "crop", "Crop · normalize", .running, detail: "518² conditioning crop")
                    case .viewEncoding(let v):
                        stage(v, "crop", "Crop · normalize", .done, dims: "518×518")
                        stage(v, "dino", "DINOv2 features", .running)
                    case .viewEncoded(let v, let tokens, let dim):
                        stage(v, "dino", "DINOv2 features", .done, detail: "\(tokens)×\(dim) tokens")
                        stage(v, "token", "View token", .done, detail: "conditioning ready")
                    case .samplingStep(let c, let nc, let s, let t):
                        if s == 1 { stage(nil, "merge", mergeLabel, .done) }
                        stage(nil, "cand\(c)", "Sampling — candidate \(c)/\(nc)", .running,
                              detail: "step \(s)/\(t)")
                        continuation.yield(.sampling(candidate: c, candidates: nc, step: s, total: t))
                    case .candidateDone(let c, let nc, let iou):
                        stage(nil, "cand\(c)", "Sampling — candidate \(c)/\(nc)", .done,
                              detail: iou.map { String(format: "mean IoU %.3f", $0) } ?? "unscored")
                    case .decoding:
                        stage(nil, "select", "Best candidate", .done)
                        stage(nil, "gauss", "Gaussian decode", .running)
                    case .decodedGaussians(let count):
                        stage(nil, "gauss", "Gaussian decode", .done, detail: "\(count) gaussians")
                    case .writingOutput:
                        break
                    }
                }, cancel: cancel, progress: { frac, stage in
                    continuation.yield(.progress(stage: stage, fraction: frac))
                }, onStepPreview: { done, total in
                    continuation.yield(.line("[DiT] step \(done)/\(total) complete"))
                }, onStepCloud: { points, step, total in
                    continuation.yield(.cloud(points: points, step: step, total: total))
                })
                continuation.yield(.line("Wrote \(gen.pointCount) colored points → \(out.lastPathComponent) + gaussian splat → \(outSplat.lastPathComponent)"))

                // HD photo texture: backproject every view's pixels onto the splat —
                // measured colors beat generated ones. Needs ≥2 views (with one view
                // the generated colors already came from exactly that angle).
                // Finish-candidate stop: the splat above is the kept result; skip the
                // post-processing extras and get the user to the viewport.
                let stoppedEarly = cancel?.isRequested == true
                if stoppedEarly {
                    continuation.yield(.line("[cancel] stopped after candidate — texture/mesh skipped"))
                    stage(nil, "texture", "Photo texture", .skipped, detail: "stopped by user")
                    stage(nil, "mesh", "Mesh extraction", .skipped, detail: "stopped by user")
                }
                var textureViews = [TextureProject.View]()
                if args.photoTexture, nViews > 1, !stoppedEarly {
                    for v in 0 ..< nViews {
                        guard let rgba = preRGBAs[v] else { continue }
                        textureViews.append(TextureProject.View(rgbaURL: rgba,
                                                                yaw: gen.viewYaws[v],
                                                                weight: max(0.05, gen.viewIoUs[v])))
                    }
                }
                if !textureViews.isEmpty {
                    continuation.yield(.progress(stage: "Backprojecting photo texture (\(textureViews.count) views)…", fraction: 0.975))
                    stage(nil, "texture", "Photo texture", .running, detail: "\(textureViews.count) views")
                    do {
                        let n = try TextureProject.recolorSplatPLY(at: outSplat, views: textureViews) {
                            continuation.yield(.line("[texture] \($0)"))
                        }
                        _ = try? TextureProject.recolorPointCloudPLY(at: out, views: textureViews)
                        continuation.yield(.line("[texture] recolored \(n) gaussians from \(textureViews.count) photos"))
                        stage(nil, "texture", "Photo texture", .done,
                              detail: "recolored \(n) gaussians from \(textureViews.count) photos")
                    } catch {
                        continuation.yield(.line("[texture] skipped: \(error)"))
                        stage(nil, "texture", "Photo texture", .skipped, detail: "error: \(error)")
                    }
                } else if args.photoTexture, nViews > 1, !stoppedEarly {
                    let fix = Config.rmbgReady || RMBGWorkerBackend.isAvailable
                        ? "background removal failed this run"
                        : "install RMBG-2.0 (accept license at huggingface.co/briaai/RMBG-2.0, `hf auth login`, `hf download briaai/RMBG-2.0`, run tools/backend/setup.sh)"
                    stage(nil, "texture", "Photo texture", .skipped, detail: "no cutouts — \(fix)")
                    continuation.yield(.line("[texture] WARNING: skipped, no cutouts — \(fix)"))
                }

                // Surface extraction: marching cubes over the gaussian density field.
                var outMesh: URL?
                if args.extractMesh, !stoppedEarly {
                    do {
                        continuation.yield(.progress(stage: "Extracting mesh (marching cubes)…", fraction: 0.99))
                        stage(nil, "mesh", "Mesh extraction", .running)
                        let meshURL = Config.outputDir.appending(path: "\(base)_mesh.ply")
                        let objURL = Config.outputDir.appending(path: "\(base)_mesh.obj")
                        var mesh = try MeshExtract.extract(gsPLY: outSplat) { msg in
                            continuation.yield(.line("[mesh] \(msg)"))
                        }
                        // Sapiens photo refinement: re-sculpt the camera-facing surface
                        // against measured normals + snap the outline to the silhouette.
                        if let sapiensURL = Config.sapiensModelURL, let rgba = preRGBAs[0] {
                            do {
                                continuation.yield(.progress(stage: "Refining mesh (Sapiens normals)…", fraction: 0.995))
                                let grid = 1024
                                let cond = try Preprocess.condRGBAPixels(rgbaURL: rgba, resolution: grid)
                                if sharedSapiens == nil {
                                    sharedSapiens = try SapiensNormal(modelURL: sapiensURL)
                                }
                                let pred = try sharedSapiens!.predict(condRGBA: cond, size: grid)
                                let stats = NormalRefine.refine(mesh: &mesh, condRGBA: cond, gridSize: grid,
                                                                pred: pred) { msg in
                                    continuation.yield(.line("[refine] \(msg)"))
                                }
                                if let reason = stats.aborted {
                                    continuation.yield(.line("[refine] skipped: \(reason)"))
                                }
                            } catch {
                                continuation.yield(.line("[refine] skipped: \(error)"))
                            }
                        }
                        if !textureViews.isEmpty {
                            let n = TextureProject.recolor(mesh: &mesh, views: textureViews) { msg in
                                continuation.yield(.line("[texture] \(msg)"))
                            }
                            continuation.yield(.line("[texture] recolored \(n) mesh vertices"))
                        }
                        try MeshExtract.writePLY(mesh, to: meshURL)
                        try MeshExtract.writeOBJ(mesh, to: objURL)
                        outMesh = meshURL
                        continuation.yield(.line("Mesh: \(mesh.vertexCount) verts, \(mesh.triangleCount) tris → \(meshURL.lastPathComponent) + .obj"))
                        stage(nil, "mesh", "Mesh extraction", .done,
                              detail: "\(mesh.vertexCount) verts, \(mesh.triangleCount) tris")
                    } catch {
                        continuation.yield(.line("[mesh] extraction skipped: \(error)"))
                        stage(nil, "mesh", "Mesh extraction", .skipped, detail: "error: \(error)")
                    }
                }
                // Landmark package: exported for inspection/future conditioning. The
                // cells are taxonomy priors + Vision pose features unless a grounding
                // backend produced real detections (none is installed today).
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                enc.dateEncodingStrategy = .iso8601
                var landmarkFile: String?
                if let pkg = finalLandmarks {
                    let lmURL = Config.outputDir.appending(path: "\(base)_landmarks.json")
                    if let data = try? enc.encode(pkg) {
                        try? data.write(to: lmURL)
                        landmarkFile = lmURL.lastPathComponent
                        continuation.yield(.line("[landmarks] exported \(lmURL.lastPathComponent) — "
                            + (pkg.backendAvailable ? "with real backend outputs"
                               : "taxonomy priors + pose features only (no grounding backend)")
                            + "; not consumed by generation"))
                    }
                }

                // Run metadata: the exact settings + seed + per-view estimates that
                // produced these artifacts, machine-readable for later comparison.
                let meta = RunMetadata(createdAt: Date(), inputs: args.imagePaths,
                                       steps: args.steps, cfgScale: args.cfgScale,
                                       multiViewMode: args.multiView.rawValue,
                                       occupancyThreshold: args.occupancyThreshold,
                                       opacityThreshold: args.opacityThreshold,
                                       seedCandidates: args.seedCandidates,
                                       seedUsed: gen.seedUsed,
                                       useRMBG: args.useRMBG, useUpscaler: args.useUpscaler,
                                       extractMesh: args.extractMesh,
                                       viewYaws: gen.viewYaws, viewIoUs: gen.viewIoUs,
                                       pointCloud: out.lastPathComponent,
                                       splat: outSplat.lastPathComponent,
                                       mesh: outMesh?.lastPathComponent,
                                       analysis: args.analysis,
                                       landmarkPackageFile: landmarkFile,
                                       userPrompt: args.userPrompt.isEmpty ? nil : args.userPrompt,
                                       cutouts: cutoutFiles,
                                       rmbgStatus: rmbgStatus,
                                       sam3Status: sam3Status,
                                       sapiensPoseStatus: sapiensStatus)
                let metaURL = Config.outputDir.appending(path: "\(base)_run.json")
                if let data = try? enc.encode(meta) {
                    try? data.write(to: metaURL)
                    continuation.yield(.line("[meta] wrote \(metaURL.lastPathComponent)"))
                }
                stage(nil, "result", "Result", .done,
                      detail: stoppedEarly ? "\(gen.pointCount) points (stopped after candidate)"
                                           : "\(gen.pointCount) points")
                continuation.yield(.result(pointCloud: out, splat: outSplat, mesh: outMesh))
            } catch LiToEngine.EngineError.cancelled {
                continuation.yield(.line("[cancel] generation stopped — no result produced"))
                continuation.yield(.cancelled)
            } catch {
                continuation.yield(.failed("\(error)"))
            }

            for tmp in tempFiles { try? FileManager.default.removeItem(at: tmp) }
            continuation.finish()
        }
        thread.stackSize = 32 << 20
        thread.start()
    }
}

/// Pixel size from the image header (no full decode), EXIF-orientation corrected.
private func imagePixelSize(_ url: URL) -> (Int, Int)? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
          let w = props[kCGImagePropertyPixelWidth] as? Int,
          let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
    if let o = props[kCGImagePropertyOrientation] as? UInt32, o >= 5 { return (h, w) }
    return (w, h)
}

private func writeCGImage(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        throw RMBG.RMBGError.compile
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw RMBG.RMBGError.compile }
}
