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
}

enum RunEvent: Sendable {
    case progress(stage: String, fraction: Double)
    case line(String)
    case result(pointCloud: URL, splat: URL?, mesh: URL?)
    case failed(String)
    case preview(label: String, imageURL: URL)
    /// Intermediate occupancy decode while sampling — flat [x,y,z]* world coords (z-up).
    case cloud(points: [Float], step: Int, total: Int)
}

struct PipelineArgs: Sendable {
    var imagePath: String
    var steps: Int
    var cfgScale: Float = 3.0
    var useRMBG: Bool = true
    var useUpscaler: Bool = true
    var occupancyThreshold: Float = 0
    var opacityThreshold: Float = 0.10
    var seed: UInt64? = nil
    var seedCandidates: Int = 1
    var extractMesh: Bool = true
}

nonisolated(unsafe) private var sharedEngine: LiToEngine?
nonisolated(unsafe) private var sharedRMBG: RMBG?
nonisolated(unsafe) private var sharedUpscaler: Upscaler?
nonisolated(unsafe) private var sharedSapiens: SapiensNormal?
private let engineLock = NSLock()

func runPipeline(_ args: PipelineArgs) -> AsyncStream<RunEvent> {
    AsyncStream(bufferingPolicy: .unbounded) { continuation in
        let thread = Thread {
            // Emit the original input as the first preview
            var workingImageURL = URL(filePath: args.imagePath)
            var tempFiles = [URL]()
            continuation.yield(.preview(label: "Input", imageURL: workingImageURL))

            // Phase 0a0: Low-light normalization (auto — night shots starve DINOv2)
            if let cg = Preprocess.loadCGImageUpright(workingImageURL) {
                let lum = Preprocess.meanLuminance(cg)
                if lum < 0.27, let fixed = Preprocess.normalizeLowLight(cg) {
                    let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_lowlight_\(Int(Date().timeIntervalSince1970)).png")
                    if (try? writeCGImage(fixed, to: tmp)) != nil {
                        workingImageURL = tmp
                        tempFiles.append(tmp)
                        continuation.yield(.preview(label: "Low-light normalized", imageURL: tmp))
                        continuation.yield(.line(String(format: "[LowLight] mean luminance %.2f → normalized", lum)))
                    }
                }
            }

            // Phase 0a: Upscale (if model available)

            if args.useUpscaler, let upscalerURL = Config.upscalerModelURL {
                do {
                    continuation.yield(.progress(stage: "Upscaling (Real-ESRGAN 4x)…", fraction: 0.01))
                    if sharedUpscaler == nil {
                        sharedUpscaler = try Upscaler(modelURL: upscalerURL)
                    }
                    guard let cg = Preprocess.loadCGImageUpright(workingImageURL) else {
                        throw Upscaler.UpscalerError.image
                    }
                    let upscaled = try sharedUpscaler!.upscaleToMax(cg, maxDim: 4096)
                    let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_upscaled_\(Int(Date().timeIntervalSince1970)).png")
                    try writeCGImage(upscaled, to: tmp)
                    workingImageURL = tmp
                    tempFiles.append(tmp)
                    continuation.yield(.preview(label: "Upscaled (\(upscaled.width)×\(upscaled.height))", imageURL: tmp))
                    continuation.yield(.line("[Upscaler] 4x → \(upscaled.width)×\(upscaled.height)"))
                } catch {
                    continuation.yield(.line("[Upscaler] skipped: \(error)"))
                }
            }

            // Phase 0b: Background removal (RMBG 2.0 via CoreML)
            var preprocessedRGBA: URL?
            if args.useRMBG && Config.rmbgReady {
                do {
                    continuation.yield(.progress(stage: "Background removal (RMBG 2.0)…", fraction: 0.03))
                    if sharedRMBG == nil {
                        sharedRMBG = try RMBG(modelURL: Config.rmbgModelURL)
                    }
                    guard let cg = Preprocess.loadCGImageUpright(workingImageURL) else {
                        throw RMBG.RMBGError.compile
                    }
                    var rgba = try sharedRMBG!.removeBackground(from: cg)
                    // Trim the cutout to Vision's person mask (dilated) — drops mirror
                    // frames / furniture that RMBG keeps because they touch the subject.
                    if let trimmed = Preprocess.personTrim(rgba: rgba, original: cg) {
                        rgba = trimmed
                        continuation.yield(.line("[PersonTrim] cutout trimmed to person mask"))
                    }
                    let tmp = FileManager.default.temporaryDirectory.appending(path: "lito_rmbg_\(Int(Date().timeIntervalSince1970)).png")
                    try writeCGImage(rgba, to: tmp)
                    preprocessedRGBA = tmp
                    tempFiles.append(tmp)
                    continuation.yield(.preview(label: "Background Removed", imageURL: tmp))
                    continuation.yield(.line("[RMBG] background removed"))
                } catch {
                    continuation.yield(.line("[RMBG] skipped: \(error) — using Vision fallback"))
                }
            }

            // Phase 0c: Emit the condRGBA preview (the actual 518² input to DINOv2)
            if let rgba = preprocessedRGBA {
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
                let stem = URL(filePath: args.imagePath).deletingPathExtension().lastPathComponent
                let safe = String(stem.prefix(40).map { $0.isLetter || $0.isNumber ? $0 : "_" })
                let base = "\(safe.isEmpty ? "img" : safe)_\(Int(Date().timeIntervalSince1970))"
                let out = Config.outputDir.appending(path: "\(base)_pc.ply")
                let outSplat = Config.outputDir.appending(path: "\(base)_gs.ply")
                let n = try engine.generate(imageURL: workingImageURL, steps: args.steps,
                                            outPLY: out, preprocessedRGBA: preprocessedRGBA,
                                            cfgScale: args.cfgScale,
                                            occupancyThreshold: args.occupancyThreshold,
                                            opacityThreshold: args.opacityThreshold,
                                            seed: args.seed,
                                            seedCandidates: args.seedCandidates,
                                            outSplatPLY: outSplat,
                                            progress: { frac, stage in
                    continuation.yield(.progress(stage: stage, fraction: frac))
                }, onStepPreview: { done, total in
                    continuation.yield(.line("[DiT] step \(done)/\(total) complete"))
                }, onStepCloud: { points, step, total in
                    continuation.yield(.cloud(points: points, step: step, total: total))
                })
                continuation.yield(.line("Wrote \(n) colored points → \(out.lastPathComponent) + gaussian splat → \(outSplat.lastPathComponent)"))

                // Surface extraction: marching cubes over the gaussian density field.
                var outMesh: URL?
                if args.extractMesh {
                    do {
                        continuation.yield(.progress(stage: "Extracting mesh (marching cubes)…", fraction: 0.99))
                        let meshURL = Config.outputDir.appending(path: "\(base)_mesh.ply")
                        let objURL = Config.outputDir.appending(path: "\(base)_mesh.obj")
                        var mesh = try MeshExtract.extract(gsPLY: outSplat) { msg in
                            continuation.yield(.line("[mesh] \(msg)"))
                        }
                        // Sapiens photo refinement: re-sculpt the camera-facing surface
                        // against measured normals + snap the outline to the silhouette.
                        if let sapiensURL = Config.sapiensModelURL, let rgba = preprocessedRGBA {
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
                        try MeshExtract.writePLY(mesh, to: meshURL)
                        try MeshExtract.writeOBJ(mesh, to: objURL)
                        outMesh = meshURL
                        continuation.yield(.line("Mesh: \(mesh.vertexCount) verts, \(mesh.triangleCount) tris → \(meshURL.lastPathComponent) + .obj"))
                    } catch {
                        continuation.yield(.line("[mesh] extraction skipped: \(error)"))
                    }
                }
                continuation.yield(.result(pointCloud: out, splat: outSplat, mesh: outMesh))
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

private func writeCGImage(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        throw RMBG.RMBGError.compile
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw RMBG.RMBGError.compile }
}
