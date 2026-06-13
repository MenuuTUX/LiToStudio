import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import LiToKit
import SplatIO
import simd

/// Write a CGImage (incl. RGBA) to a PNG.
func writePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        struct E: Error {}; throw E()
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { struct E: Error {}; throw E() }
}

func loadCG(_ path: String) -> CGImage? {
    Preprocess.loadCGImageUpright(URL(filePath: path))
}

// Dev harness.
//   LiToSmoke smoke  [weights.safetensors]          — MLX capability check (Stage 0)
//   LiToSmoke dino   <weights.safetensors> <golden> — DINOv2 parity vs Python (Stage 2)
//   LiToSmoke coreml <model.mlpackage|.mlmodel>     — print a CoreML model's input/output spec
func describeFeature(_ d: MLFeatureDescription) -> String {
    switch d.type {
    case .multiArray:
        if let c = d.multiArrayConstraint {
            return "MultiArray shape=\(c.shape.map { $0.intValue }) dtype=\(c.dataType.rawValue)"
        }
        return "MultiArray"
    case .image:
        if let c = d.imageConstraint {
            return "Image \(c.pixelsWide)x\(c.pixelsHigh) pixelFormat=\(c.pixelFormatType)"
        }
        return "Image"
    default:
        return "type(\(d.type.rawValue))"
    }
}
/// Shared photo(s) → splat path for the `engine` and `sculpt` commands: contact-sheet
/// split (a single image holding several renders becomes N views), RMBG cutouts (with
/// low-light + person-trim conditioning) when the model is present, then the full
/// native engine with multi-view conditioning. Returns the RGBA cutout URLs and the
/// per-view yaw/IoU estimates so later stages (Sapiens refinement, photo texture)
/// can reuse the exact conditioning views.
@discardableResult
func runEngineCLI(weightsDir: URL, imagePaths: [String], outPLY: URL,
                  steps: Int, cfg: Float, seed: UInt64?, bestOf: Int,
                  mode: MultiViewMode = .multidiffusion)
    throws -> (splat: URL, rgbas: [URL?], yaws: [Float], ious: [Float]) {
    let rmbgURL = weightsDir.appending(path: "RMBG2.mlpackage")
    let rmbg = FileManager.default.fileExists(atPath: rmbgURL.path)
        ? try? RMBG(modelURL: rmbgURL) : nil

    // One input that is actually a contact sheet → split into per-view crops.
    var paths = imagePaths
    if paths.count == 1 {
        let splitDir = outPLY.deletingPathExtension().appendingPathExtension("views")
        if let views = try? SheetSplit.splitToFiles(imageURL: URL(filePath: paths[0]), rmbg: rmbg,
                                                    outDir: splitDir, log: { print("  [sheet] \($0)") }),
           !views.isEmpty {
            paths = views.map(\.path)
            print("  [sheet] contact sheet → \(paths.count) views in \(splitDir.lastPathComponent)/")
        }
    }

    var engineInputs = paths.map { URL(filePath: $0) }
    var preRGBAs = [URL?](repeating: nil, count: paths.count)
    for v in 0 ..< paths.count {
        let vtag = paths.count > 1 ? " v\(v + 1)" : ""
        guard let rmbg, var cg = loadCG(engineInputs[v].path) else {
            if rmbg == nil { print("  [RMBG]\(vtag) model not found — using Vision foreground fallback") }
            continue
        }
        let lum = Preprocess.meanLuminance(cg)
        if lum < 0.27, let fixed = Preprocess.normalizeLowLight(cg) {
            cg = fixed
            let tmp = outPLY.deletingPathExtension().appendingPathExtension("lowlight\(vtag).png")
            try writePNG(fixed, to: tmp); engineInputs[v] = tmp
            print(String(format: "  [LowLight]%@ mean luminance %.2f → normalized", vtag, lum))
        }
        let t = Date()
        var rgba = try rmbg.removeBackground(from: cg)
        if let trimmed = Preprocess.personTrim(rgba: rgba, original: cg) {
            rgba = trimmed
            print("  [PersonTrim]\(vtag) cutout trimmed to person mask")
        }
        let tmp = outPLY.deletingPathExtension().appendingPathExtension("rmbg\(vtag).png")
        try writePNG(rgba, to: tmp); preRGBAs[v] = tmp
        print("  [RMBG]\(vtag) \(cg.width)×\(cg.height) masked in \(String(format: "%.1f", Date().timeIntervalSince(t)))s → \(tmp.lastPathComponent)")
    }
    let t0 = Date()
    let splatOut = outPLY.deletingPathExtension().appendingPathExtension("gs.ply")
    let gen = try LiToEngine(weightsDir: weightsDir).generate(
        imageURLs: engineInputs, steps: steps, outPLY: outPLY,
        preprocessedRGBAs: preRGBAs, cfgScale: cfg, multiViewMode: mode,
        seed: seed, seedCandidates: bestOf,
        outSplatPLY: splatOut) { f, s in
        print(String(format: "  [%3.0f%%] %@", f * 100, s))
    }
    if engineInputs.count > 1 {
        for v in 0 ..< engineInputs.count {
            print(String(format: "  [view %d] yaw %.0f° IoU %.3f", v + 1,
                         gen.viewYaws[v] * 180 / .pi, gen.viewIoUs[v]))
        }
    }
    print("✓ ENGINE DONE: \(gen.pointCount) colored points, \(engineInputs.count) view(s), cfg=\(cfg), steps=\(steps) in \(Int(Date().timeIntervalSince(t0)))s → \(outPLY.path) (+ \(splatOut.lastPathComponent))")
    return (splatOut, preRGBAs, gen.viewYaws, gen.viewIoUs)
}

/// Splat → (optionally Sapiens-refined, optionally photo-textured) mesh, PLY + OBJ.
/// `textureViews` (≥2) first recolors the splat PLY in place (so mesh vertex colors
/// inherit photo color), then refines the recolored mesh vertices directly.
func refineCLI(weightsDir: URL, gsPLY: URL, rgba: URL?, outBase: URL,
               meshRes: Int, iso: Float, grid: Int, selftest: Bool,
               textureViews: [TextureProject.View] = []) throws {
    let t0 = Date()
    if textureViews.count >= 2 {
        do {
            let n = try TextureProject.recolorSplatPLY(at: gsPLY, views: textureViews) { print("  [texture] \($0)") }
            print("  [texture] splat recolored from \(textureViews.count) photos (\(n) gaussians)")
        } catch {
            print("  [texture] splat recolor skipped: \(error)")
        }
    }
    var mesh = try MeshExtract.extract(gsPLY: gsPLY, resolution: meshRes, iso: iso) {
        print("  [mesh] \($0)")
    }
    let modelURL = SapiensNormal.locate(weightsDir: weightsDir)
    if let rgbaURL = rgba, selftest || modelURL != nil {
        let cond = try Preprocess.condRGBAPixels(rgbaURL: rgbaURL, resolution: grid)
        var pred: SapiensNormal.NormalMap?
        if !selftest, let modelURL {
            let t = Date()
            pred = try SapiensNormal(modelURL: modelURL).predict(condRGBA: cond, size: grid)
            let nValid = pred!.valid.reduce(0) { $0 + Int($1) }
            print("  [sapiens] normal map \(grid)² (\(nValid) px) in \(String(format: "%.1f", Date().timeIntervalSince(t)))s")
        }
        let stats = NormalRefine.refine(mesh: &mesh, condRGBA: cond, gridSize: grid,
                                        pred: pred) { print("  [refine] \($0)") }
        if let reason = stats.aborted { print("  [refine] SKIPPED: \(reason)") }
    } else if rgba == nil {
        print("  [refine] no RGBA cutout available — raw mesh only")
    } else {
        print("  [refine] no SapiensNormal model in weights dir — raw mesh (convert via docs/sapiens2_normal_coreml_colab.ipynb)")
    }
    if textureViews.count >= 2 {
        let n = TextureProject.recolor(mesh: &mesh, views: textureViews) { print("  [texture] \($0)") }
        print("  [texture] mesh recolored (\(n) vertices)")
    }
    let plyURL = outBase.pathExtension.isEmpty ? outBase.appendingPathExtension("ply") : outBase
    let objURL = plyURL.deletingPathExtension().appendingPathExtension("obj")
    try MeshExtract.writePLY(mesh, to: plyURL)
    try MeshExtract.writeOBJ(mesh, to: objURL)
    print("✓ SCULPT: \(mesh.vertexCount) verts, \(mesh.triangleCount) tris in \(Int(Date().timeIntervalSince(t0)))s")
    print("  → \(plyURL.lastPathComponent) + \(objURL.lastPathComponent)")
}

let args = CommandLine.arguments.filter { $0 != "--selftest" }
let selftestFlag = CommandLine.arguments.contains("--selftest")
let cmd = args.count > 1 ? args[1] : "smoke"

do {
    switch cmd {
    case "dino":
        guard args.count >= 4 else { print("usage: LiToSmoke dino <weights> <golden>"); exit(1) }
        try Parity.dino(weights: URL(filePath: args[2]), golden: URL(filePath: args[3]))
    case "dit":
        guard args.count >= 4 else { print("usage: LiToSmoke dit <weights> <golden>"); exit(1) }
        try Parity.dit(weights: URL(filePath: args[2]), golden: URL(filePath: args[3]))
    case "voxel":
        guard args.count >= 5 else { print("usage: LiToSmoke voxel <weights> <trellis_dec> <golden>"); exit(1) }
        try Parity.voxel(weights: URL(filePath: args[2]), trellis: URL(filePath: args[3]), golden: URL(filePath: args[4]))
    case "gs":
        guard args.count >= 4 else { print("usage: LiToSmoke gs <weights> <golden>"); exit(1) }
        try Parity.gs(weights: URL(filePath: args[2]), golden: URL(filePath: args[3]))
    case "rmbg":
        guard args.count >= 5 else { print("usage: LiToSmoke rmbg <weightsDir> <image> <out.png>"); exit(1) }
        let modelURL = URL(filePath: args[2]).appending(path: "RMBG2.mlpackage")
        guard let cg = loadCG(args[3]) else { print("✗ cannot load \(args[3])"); exit(1) }
        let t0 = Date()
        let rgba = try RMBG(modelURL: modelURL).removeBackground(from: cg)
        try writePNG(rgba, to: URL(filePath: args[4]))
        print("✓ RMBG \(cg.width)×\(cg.height) → \(rgba.width)×\(rgba.height) in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s → \(args[4])")
    case "engine":
        // <images> = one path, a comma-separated list of angles, or one contact sheet.
        guard args.count >= 5 else { print("usage: LiToSmoke engine <weightsDir> <images> <out.ply> [steps] [cfg] [seed] [bestOf]"); exit(1) }
        let steps = args.count > 5 ? (Int(args[5]) ?? 4) : 4
        let cfg = args.count > 6 ? (Float(args[6]) ?? 3.0) : 3.0
        let seed = args.count > 7 ? UInt64(args[7]) : nil
        let bestOf = args.count > 8 ? (Int(args[8]) ?? 1) : 1
        try runEngineCLI(weightsDir: URL(filePath: args[2]),
                         imagePaths: args[3].split(separator: ",").map(String.init),
                         outPLY: URL(filePath: args[4]), steps: steps, cfg: cfg,
                         seed: seed, bestOf: bestOf)
    case "sculpt":
        // Photo(s) → splat → Sapiens-refined, photo-textured mesh (the quality pipeline).
        // <images> = one path, a comma-separated list of angles, or one contact sheet.
        guard args.count >= 5 else {
            print("usage: LiToSmoke sculpt <weightsDir> <images> <out_base> [steps=25] [cfg=3.0] [seed] [bestOf=3] [meshRes=256]")
            exit(1)
        }
        let steps = args.count > 5 ? (Int(args[5]) ?? 25) : 25
        let cfg = args.count > 6 ? (Float(args[6]) ?? 3.0) : 3.0
        let seed = args.count > 7 ? UInt64(args[7]) : nil
        let bestOf = args.count > 8 ? (Int(args[8]) ?? 3) : 3
        let meshRes = args.count > 9 ? (Int(args[9]) ?? 256) : 256
        let weightsDir = URL(filePath: args[2])
        let base = URL(filePath: args[4])
        let pcURL = base.appendingPathExtension("pc.ply")
        let (splat, rgbas, yaws, ious) = try runEngineCLI(
            weightsDir: weightsDir, imagePaths: args[3].split(separator: ",").map(String.init),
            outPLY: pcURL, steps: steps, cfg: cfg, seed: seed, bestOf: bestOf)
        var textureViews = [TextureProject.View]()
        if rgbas.count > 1 {
            for v in 0 ..< rgbas.count {
                guard let rgba = rgbas[v] else { continue }
                textureViews.append(TextureProject.View(rgbaURL: rgba, yaw: yaws[v],
                                                        weight: max(0.05, ious[v])))
            }
        }
        try refineCLI(weightsDir: weightsDir, gsPLY: splat, rgba: rgbas.first ?? nil,
                      outBase: base.appendingPathExtension("mesh.ply"),
                      meshRes: meshRes, iso: 0.5, grid: 1024, selftest: selftestFlag,
                      textureViews: textureViews)
    case "analyze":
        // What auto-detect would pick for these image(s), with the working shown.
        guard args.count >= 3 else { print("usage: LiToSmoke analyze <image[,image2,...]>"); exit(1) }
        let urls = args[2].split(separator: ",").map { URL(filePath: String($0)) }
        let a = ImageAnalyzer.analyze(imageURLs: urls)
        print("measured: \(a.summary)")
        for v in a.views {
            let subj = v.subjectEstimateReliable
                ? String(format: "subj %dpx area %.0f%%", v.subjectLongSide, (v.maskAreaRatio ?? 0) * 100)
                : "subj n/a"
            print(String(format: "  view %d: %d×%d  %@  ε=%.2f(subj %.2f) σ=%.2f λ=%.2f ℓ=%.2f τ=%.2f  %@%@%@%@",
                         v.index + 1, v.width, v.height, subj, v.edgeDensity, v.subjectEdgeDensity,
                         v.contrast, v.sharpness, v.luminance, v.textureEntropy,
                         v.orientation == .unknown ? "" : "\(v.orientation.label)(est) ",
                         v.framing == .unknown ? "" : "\(v.framing.label) ",
                         v.premasked ? "(pre-masked) " : "",
                         v.needsUpscale ? "→ upscale" : "as-is"))
            if let raised = v.raisedHand { print("      raised hand: \(raised)") }
            print("      \(v.upscaleNote)")
        }
        print("settings (default → detected):")
        for n in a.notes {
            print("  \(n.name): \(n.defaultValue) → \(n.recommended)\(n.changed ? "  *" : "")")
            print("      \(n.reason)")
        }
    case "upscale":
        // Exercise the 2K-policy upscale path: alpha-preserving Real-ESRGAN with the
        // same cascade rule the app pipeline uses (second pass only when one 4× still
        // leaves the subject under the 2K target and the canvas has headroom).
        guard args.count >= 4 else {
            print("usage: LiToSmoke upscale <esrgan.mlmodel> <in.png> [out.png]"); exit(1)
        }
        let up = try Upscaler(modelURL: URL(filePath: args[2]))
        guard let cg = loadCG(args[3]) else { print("✗ can't read \(args[3])"); exit(1) }
        let canvas = max(cg.width, cg.height)
        let subj = ImageAnalyzer.subjectBoxEstimate(cg)?.longSide ?? canvas
        print("in : \(cg.width)×\(cg.height)  alphaInfo=\(cg.alphaInfo.rawValue)  subject≈\(subj)px")
        var out = try up.upscaleToMaxPreservingAlpha(cg, maxDim: ImageAnalyzer.canvasCapPx)
        var passes = 1
        var subjAfter = subj * max(out.width, out.height) / canvas
        if subjAfter < ImageAnalyzer.subjectTargetPx,
           max(out.width, out.height) < ImageAnalyzer.canvasCapPx {
            out = try up.upscaleToMaxPreservingAlpha(out, maxDim: ImageAnalyzer.canvasCapPx)
            subjAfter = subj * max(out.width, out.height) / canvas
            passes = 2
        }
        print("out: \(out.width)×\(out.height)  alphaInfo=\(out.alphaInfo.rawValue)  subject≈\(subjAfter)px  (\(passes) pass\(passes > 1 ? "es" : ""))")
        if args.count > 4 { try writePNG(out, to: URL(filePath: args[4])) }
    case "landmarks":
        // Build + print the landmark conditioning package exactly as the app does:
        // labels from filename/pose, Vision pose features, taxonomy priors. No
        // grounding backend exists, so the matrix carries expectations, never
        // detections — the output says so.
        guard args.count >= 3 else { print("usage: LiToSmoke landmarks <img[,img2,...]> [out.json]"); exit(1) }
        let urls = args[2].split(separator: ",").map { URL(filePath: String($0)) }
        let a = ImageAnalyzer.analyze(imageURLs: urls)
        var labels = [ViewLabel](), sources = [String](), pose = [PoseFeatures?]()
        for (i, u) in urls.enumerated() {
            let va = a.views.first(where: { $0.index == i })
            if let f = ViewLabel.fromFilename(u.deletingPathExtension().lastPathComponent) {
                labels.append(f); sources.append("filename")
            } else if let va, va.orientation != .unknown {
                labels.append(ViewLabel(orientation: va.orientation)); sources.append("pose-estimate")
            } else {
                labels.append(.unknown); sources.append("none")
            }
            pose.append(va?.poseFeatures)
        }
        let pkg = LandmarkPackage.build(imagePaths: urls.map(\.path), labels: labels,
                                        labelSources: sources, pose: pose)
        print("backend: \(pkg.backend)")
        print("consumed by generator: \(pkg.consumedByGenerator) — \(pkg.consumptionNote)")
        for v in pkg.views {
            let poseStr = v.poseFeatures.map { " pose: \($0.framing.rawValue)\($0.raisedHand.map { " raised:\($0)" } ?? "")" } ?? ""
            print("  V\(v.index + 1) \(v.viewLabel.rawValue) (\(v.labelSource))  expected \(v.expectedTokens.count) tokens  detections \(v.observations.count)\(poseStr)")
        }
        print("visibility matrix (● detected ○ expected – not · ? unknown):")
        for row in pkg.visibilityMatrix {
            let cells = row.perView.map { c in
                c.hasPrefix("detected") ? "●" : c == "expected" ? "○" : c == "not_expected" ? "–" : "?"
            }.joined(separator: " ")
            print(String(format: "  %@ %-22s %@", row.id, (row.token as NSString).utf8String!, cells))
        }
        if args.count > 3 {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            enc.dateEncodingStrategy = .iso8601
            try enc.encode(pkg).write(to: URL(filePath: args[3]))
            print("wrote \(args[3])")
        }
    case "sam3":
        // Native SAM 3.1 CoreML check. `norms` mode runs one prompt under every
        // input normalization (the conversion is undocumented — the right one is
        // whichever produces a confident, plausibly-placed detection); default mode
        // grounds all taxonomy prompts.
        //   LiToSmoke sam3 <sam3-coreml-dir> <img> [norms|all|<promptID>] [threshold]
        guard args.count >= 4 else {
            print("usage: LiToSmoke sam3 <sam3-coreml-dir> <img> [norms|all|L001..] [threshold]"); exit(1)
        }
        let dir = URL(filePath: args[2])
        guard let img = loadCG(args[3]) else { print("✗ can't read \(args[3])"); exit(1) }
        let mode = args.count > 4 ? args[4] : "all"
        let thr = args.count > 5 ? Float(args[5]) ?? 0.4 : 0.4
        if mode == "norms" {
            for norm in Sam3CoreML.Norm.allCases {
                let t0 = Date()
                let sam = try Sam3CoreML(dir: dir, norm: norm)
                let res = try sam.detect(image: img, threshold: 0, onlyPrompt: "L001")
                if let d = res.first?.detection {
                    print(String(format: "%-9@ face: conf %.3f  mask-box [%.2f %.2f %.2f %.2f]  (%.1fs)",
                                 norm.rawValue as NSString, d.confidence,
                                 d.box[0], d.box[1], d.box[2], d.box[3],
                                 -t0.timeIntervalSinceNow))
                } else {
                    print("\(norm.rawValue): face: NO mask above threshold")
                }
            }
        } else {
            let sam = try Sam3CoreML(dir: dir)
            print("norm: \(sam.norm.rawValue) (override: LITO_SAM3_NORM)")
            // Optional 5th arg = a cutout PNG → person silhouette for clean gating.
            var personMask: [Bool]?
            if args.count > 6, let cut = loadCG(args[6]) {
                personMask = Sam3CoreML.personMask288(fromCutout: cut)
                print("person mask: \(personMask != nil ? "from cutout" : "cutout had no alpha")")
            } else {
                personMask = Sam3CoreML.personMask288(fromCutout: img)
                print("person mask: \(personMask != nil ? "from image alpha" : "none (opaque image)")")
            }
            let t0 = Date()
            let res = try sam.detect(image: img, personMask: personMask, threshold: thr,
                                     onlyPrompt: mode == "all" ? nil : mode)
            print(String(format: "%.1fs for %d prompts", -t0.timeIntervalSinceNow, res.count))
            for (p, det) in res {
                if let d = det {
                    let maskURL = FileManager.default.temporaryDirectory.appending(path: "sam3_\(p.id).png")
                    let ovURL = FileManager.default.temporaryDirectory.appending(path: "sam3_\(p.id)_overlay.png")
                    try Sam3CoreML.writeMask(d, to: maskURL, width: img.width, height: img.height)
                    try? Sam3CoreML.writeOverlay(d, over: img, to: ovURL)
                    print(String(format: "  %@ %-22@ conf %.3f  cov %.0f%% person %.0f%%  overlay → %@",
                                 p.id, p.token as NSString, d.confidence,
                                 d.coverage * 100, d.personCoverage * 100, ovURL.path))
                } else {
                    print("  \(p.id) \(p.token): not detected")
                }
            }
        }
    case "sam3concept":
        // Free-text concept grounding through the full app path: CLIP tokenizer
        // worker → SAM 3.1 → region overlay. Verifies the text-guidance feature.
        //   LiToSmoke sam3concept <sam3-coreml-dir> <img> "<phrase>" [cutout]
        guard args.count >= 5 else {
            print("usage: LiToSmoke sam3concept <sam3-coreml-dir> <img> \"<phrase>\" [cutout]"); exit(1)
        }
        let dir = URL(filePath: args[2])
        let phrase = args[4]
        guard ClipTokenizer.isAvailable else { print("✗ backend venv missing (tokenizer)"); exit(1) }
        let ids = try ClipTokenizer.tokenize([phrase])
        let label = ViewLabel.fromFilename(URL(filePath: args[3]).deletingPathExtension().lastPathComponent) ?? .unknown
        let cutouts: [URL?] = args.count > 5 ? [URL(filePath: args[5])] : [nil]
        let sam = try Sam3CoreML(dir: dir)
        let obs = try sam.groundConcept(phrase: phrase, tokenIds: ids[0], id: "USER",
                                        images: [URL(filePath: args[3])], labels: [label],
                                        masksDir: FileManager.default.temporaryDirectory.appending(path: "sam3concept"),
                                        cutouts: cutouts)
        if let o = obs.first ?? nil {
            print(String(format: "“%@”: detected conf %.3f  person %.0f%%  overlay → %@",
                         phrase, o.confidence, (o.personCoverage ?? 0) * 100, o.overlayPath ?? "?"))
        } else {
            print("“\(phrase)”: not found")
        }
    case "ground":
        // End-to-end check of the Python backend adapters (RMBG cutouts, Sapiens2
        // pose, SAM3 grounding) through the same Swift code the app uses. Needs the
        // tools/backend venv; no engine weights.
        guard args.count >= 3 else { print("usage: LiToSmoke ground <img[,img2,...]> [outDir]"); exit(1) }
        let urls = args[2].split(separator: ",").map { URL(filePath: String($0)) }
        let outDir = URL(filePath: args.count > 3 ? args[3]
                         : FileManager.default.temporaryDirectory.appending(path: "lito_ground").path)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        print("backend python: \(PythonBackend.python?.path ?? "NOT INSTALLED — run tools/backend/setup.sh")")
        let glabels = urls.map { ViewLabel.fromFilename($0.deletingPathExtension().lastPathComponent) ?? .unknown }
        if RMBGWorkerBackend.isAvailable {
            let res = try RMBGWorkerBackend.cutouts(images: urls, outDir: outDir.appending(path: "cutouts"))
            print("RMBG: \(res.backend) → \(res.cutouts.compactMap { $0 }.count)/\(urls.count) cutouts → \(outDir.path)/cutouts")
        } else {
            print("RMBG worker: unavailable (venv or briaai/RMBG-2.0 cache missing)")
        }
        if SapiensPoseBackend.isAvailable {
            var boxes = [[Double]?]()
            for u in urls {
                if let cg = loadCG(u.path), let b = ImageAnalyzer.subjectBoxEstimate(cg) {
                    boxes.append([Double(b.x) / Double(cg.width), Double(b.y) / Double(cg.height),
                                  Double(b.width) / Double(cg.width), Double(b.height) / Double(cg.height)])
                } else { boxes.append(nil) }
            }
            let recs = try SapiensPoseBackend.extract(images: urls, labels: glabels,
                                                      subjectBoxes: boxes,
                                                      outDir: outDir.appending(path: "pose"))
            for (i, r) in recs.enumerated() {
                guard let r else { print("Sapiens2 v\(i + 1): no result"); continue }
                let g = r.groups.sorted { $0.key < $1.key }
                    .map { "\($0.key) \($0.value.visible)/\($0.value.total)" }.joined(separator: " ")
                print("Sapiens2 v\(i + 1): \(r.keypointCount) kp · \(g)\(r.raisedHand.map { " · raised \($0)" } ?? "")")
            }
        } else {
            print("Sapiens2: \(SapiensPoseBackend.unavailableReason)")
        }
        // SAM3: native CoreML packages first (repo weights/sam3-coreml), gated
        // python worker second — same precedence as the app pipeline.
        let repoWeights = PythonBackend.backendDir?
            .deletingLastPathComponent().deletingLastPathComponent().appending(path: "weights")
        let sam3Res: Sam3RunResult?
        if let rw = repoWeights, let dir = Sam3CoreML.locate(weightsDir: rw) {
            let sam = try Sam3CoreML(dir: dir)
            sam3Res = try sam.ground(images: urls, labels: glabels,
                                     masksDir: outDir.appending(path: "masks"))
        } else if Sam3Backend.isAvailable {
            sam3Res = try Sam3Backend.detect(images: urls, labels: glabels,
                                             masksDir: outDir.appending(path: "masks"))
        } else {
            sam3Res = nil
            print("SAM3: \(Sam3Backend.unavailableReason)")
        }
        if let res = sam3Res {
            print("SAM3: \(res.backend) → \(res.detectionCount) detections → \(outDir.path)/masks")
            for (i, fs) in res.perView.enumerated() {
                let line = fs.map { f in
                    f.status == "detected"
                        ? String(format: "%@(%.2f)", f.token, f.observation?.confidence ?? 0)
                        : "\(f.token)=\(f.status)"
                }.joined(separator: " ")
                print("  v\(i + 1): \(line)")
            }
        }
    case "texture":
        // Iterate on photo-texture backprojection on an existing splat without
        // re-running the engine. Views are rgba@yawDegrees[@weight], comma-separated.
        guard args.count >= 4 else {
            print("usage: LiToSmoke texture <in.gs.ply> <rgba@yaw[@w],rgba@yaw[@w],...> [pc.ply]")
            exit(1)
        }
        var views = [TextureProject.View]()
        for spec in args[3].split(separator: ",") {
            let parts = spec.split(separator: "@")
            guard parts.count >= 2, let deg = Float(parts[1]) else {
                print("✗ bad view spec '\(spec)' — want path@yawDegrees[@weight]"); exit(1)
            }
            let w = parts.count > 2 ? (Float(parts[2]) ?? 1) : 1
            views.append(TextureProject.View(rgbaURL: URL(filePath: String(parts[0])),
                                             yaw: deg * .pi / 180, weight: w))
        }
        let t0 = Date()
        let n = try TextureProject.recolorSplatPLY(at: URL(filePath: args[2]), views: views) { print("  [texture] \($0)") }
        if args.count > 4 {
            _ = try TextureProject.recolorPointCloudPLY(at: URL(filePath: args[4]), views: views) { print("  [texture] \($0)") }
        }
        print("✓ TEXTURE: \(n) gaussians recolored from \(views.count) views in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    case "split":
        // Contact sheet → per-view crops (no weights needed; RMBG used when present).
        guard args.count >= 4 else { print("usage: LiToSmoke split <sheet.png> <outDir> [weightsDir]"); exit(1) }
        var rmbg: RMBG?
        if args.count > 4 {
            let m = URL(filePath: args[4]).appending(path: "RMBG2.mlpackage")
            if FileManager.default.fileExists(atPath: m.path) { rmbg = try? RMBG(modelURL: m) }
        }
        let t0 = Date()
        if let views = try SheetSplit.splitToFiles(imageURL: URL(filePath: args[2]), rmbg: rmbg,
                                                   outDir: URL(filePath: args[3]),
                                                   log: { print("  [sheet] \($0)") }) {
            print("✓ SPLIT: \(views.count) views in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
            for v in views { print("  → \(v.path)") }
        } else {
            print("✗ not a contact sheet (fewer than 2 comparable figures found)")
            exit(1)
        }
    case "refine":
        // Iterate on an existing splat without re-running the engine.
        guard args.count >= 6 else {
            print("usage: LiToSmoke refine <weightsDir> <in.gs.ply> <rgba.png> <out_base> [meshRes=256] [iso=0.5] [grid=1024] [--selftest]")
            exit(1)
        }
        let meshRes = args.count > 6 ? (Int(args[6]) ?? 256) : 256
        let iso = args.count > 7 ? (Float(args[7]) ?? 0.5) : 0.5
        let grid = args.count > 8 ? (Int(args[8]) ?? 1024) : 1024
        try refineCLI(weightsDir: URL(filePath: args[2]), gsPLY: URL(filePath: args[3]),
                      rgba: URL(filePath: args[4]), outBase: URL(filePath: args[5]),
                      meshRes: meshRes, iso: iso, grid: grid, selftest: selftestFlag)
    case "normals":
        // Visual check for a converted SapiensNormal model: write the predicted
        // normal map of an RGBA cutout as a (n+1)/2 PNG. Run this first after
        // dropping the .mlpackage into weights/.
        guard args.count >= 5 else { print("usage: LiToSmoke normals <weightsDir> <rgba.png> <out.png> [grid=1024]"); exit(1) }
        let grid = args.count > 5 ? (Int(args[5]) ?? 1024) : 1024
        guard let modelURL = SapiensNormal.locate(weightsDir: URL(filePath: args[2])) else {
            print("✗ no SapiensNormal.mlpackage/.mlmodelc in \(args[2]) — convert via docs/sapiens2_normal_coreml_colab.ipynb")
            exit(1)
        }
        let cond = try Preprocess.condRGBAPixels(rgbaURL: URL(filePath: args[3]), resolution: grid)
        let t0 = Date()
        let nm = try SapiensNormal(modelURL: modelURL).predict(condRGBA: cond, size: grid)
        let nValid = nm.valid.reduce(0) { $0 + Int($1) }
        print("  predicted \(nValid) px in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
        var px = [UInt8](repeating: 0, count: grid * grid * 4)
        for p in 0 ..< grid * grid {
            if nm.valid[p] == 1 {
                for c in 0 ..< 3 { px[p * 4 + c] = UInt8(max(0, min(255, (nm.normals[p * 3 + c] + 1) / 2 * 255))) }
                px[p * 4 + 3] = 255
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx2 = CGContext(data: &px, width: grid, height: grid, bitsPerComponent: 8,
                             bytesPerRow: grid * 4, space: cs,
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        try writePNG(ctx2.makeImage()!, to: URL(filePath: args[4]))
        print("✓ wrote normal map vis → \(args[4])")
    case "cond":
        // Dump the exact cond_rgba MLXArray Preprocess produces (DINOv2's input) as a PNG.
        guard args.count >= 4 else { print("usage: LiToSmoke cond <image|rgba.png> <out.png> [--rgba]"); exit(1) }
        let rgbaMode = args.contains("--rgba")
        let arr = rgbaMode ? try Preprocess.condRGBA(rgbaURL: URL(filePath: args[2]), resolution: 518)
                           : try Preprocess.condRGBA(imageURL: URL(filePath: args[2]), resolution: 518)
        let res = arr.dim(0)
        let host: [Float] = arr.asArray(Float.self)
        var px = [UInt8](repeating: 0, count: res * res * 4)
        for i in 0 ..< res * res * 4 { px[i] = UInt8(max(0, min(255, host[i] * 255))) }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx2 = CGContext(data: &px, width: res, height: res, bitsPerComponent: 8,
                             bytesPerRow: res * 4, space: cs,
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        try writePNG(ctx2.makeImage()!, to: URL(filePath: args[3]))
        print("✓ wrote cond_rgba (row 0 of the MLXArray = top row of PNG) → \(args[3])")
    case "mesh":
        // Splat → mesh: marching cubes over the gaussian density field.
        guard args.count >= 4 else { print("usage: LiToSmoke mesh <in.gs.ply> <out_base> [resolution=256] [iso=0.5]"); exit(1) }
        let res = args.count > 4 ? (Int(args[4]) ?? 256) : 256
        let iso = args.count > 5 ? (Float(args[5]) ?? 0.5) : 0.5
        let t0 = Date()
        let mesh = try MeshExtract.extract(gsPLY: URL(filePath: args[2]), resolution: res, iso: iso) {
            print("  [mesh] \($0)")
        }
        let base = URL(filePath: args[3])
        let plyURL = base.pathExtension.isEmpty ? base.appendingPathExtension("ply") : base
        let objURL = plyURL.deletingPathExtension().appendingPathExtension("obj")
        try MeshExtract.writePLY(mesh, to: plyURL)
        try MeshExtract.writeOBJ(mesh, to: objURL)
        print("✓ MESH: \(mesh.vertexCount) verts, \(mesh.triangleCount) tris (res=\(res), iso=\(iso)) in \(Int(Date().timeIntervalSince(t0)))s")
        print("  → \(plyURL.lastPathComponent) + \(objURL.lastPathComponent)")
    case "score":
        // Survey: IoU between the point cloud's orthographic silhouette and the cond alpha,
        // across all 48 axis/rotation/flip combos — used once to pin down the model's
        // conditioning-view convention for the seed-selection scorer.
        guard args.count >= 4 else { print("usage: LiToSmoke score <pc.ply> <image>"); exit(1) }
        let condA = try Preprocess.condRGBA(imageURL: URL(filePath: args[3]), resolution: 518)
        let host: [Float] = condA.asArray(Float.self)
        let S = 128
        var mask = [Bool](repeating: false, count: S * S)
        for y in 0 ..< S { for x in 0 ..< S {
            let sy = y * 518 / S, sx = x * 518 / S
            if host[(sy * 518 + sx) * 4 + 3] > 0.5 { mask[y * S + x] = true }
        }}
        let data = try Data(contentsOf: URL(filePath: args[2]))
        guard let hdr = data.range(of: Data("end_header\n".utf8)) else { exit(1) }
        var pcount = 0
        for line in String(decoding: data[..<hdr.lowerBound], as: UTF8.self).split(separator: "\n")
        where line.hasPrefix("element vertex") { pcount = Int(line.split(separator: " ").last!) ?? 0 }
        var pts = [Float](repeating: 0, count: pcount * 3)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: hdr.upperBound)
            for i in 0 ..< pcount {
                let p = base.advanced(by: i * 15)
                pts[i*3] = p.loadUnaligned(as: Float.self)
                pts[i*3+1] = p.advanced(by: 4).loadUnaligned(as: Float.self)
                pts[i*3+2] = p.advanced(by: 8).loadUnaligned(as: Float.self)
            }
        }
        var results = [(Float, String)]()
        for axis in 0 ..< 3 { for sign in [1, -1] { for rot in 0 ..< 4 { for flip in [false, true] {
            var sil = [Bool](repeating: false, count: S * S)
            for i in 0 ..< pcount {
                let p = (pts[i*3], pts[i*3+1], pts[i*3+2])
                var (u, v): (Float, Float)
                switch axis {                       // project along ±axis; (u,v) = remaining
                case 0: (u, v) = (p.1, p.2)
                case 1: (u, v) = (p.0, p.2)
                default: (u, v) = (p.0, p.1)
                }
                if sign < 0 { u = -u }
                for _ in 0 ..< rot { (u, v) = (v, -u) }
                if flip { u = -u }
                let xi = Int((u + 1.04) / 2.08 * Float(S)), yi = Int((1.04 - v) / 2.08 * Float(S))
                if xi >= 0, xi < S, yi >= 0, yi < S { sil[yi * S + xi] = true }
            }
            var inter = 0, uni = 0
            for j in 0 ..< S * S {
                if sil[j] && mask[j] { inter += 1 }
                if sil[j] || mask[j] { uni += 1 }
            }
            results.append((Float(inter) / Float(max(uni, 1)),
                            "axis=\(["x","y","z"][axis]) sign=\(sign) rot=\(rot * 90) flip=\(flip)"))
        }}}}
        for (iou, desc) in results.sorted(by: { $0.0 > $1.0 }).prefix(5) {
            print(String(format: "IoU %.3f  %@", iou, desc))
        }
    case "gscheck":
        // Validate a 3DGS gaussian PLY by reading it back through SplatIO (what the app viewer uses).
        guard args.count >= 3 else { print("usage: LiToSmoke gscheck <splat.ply>"); exit(1) }
        let sem = DispatchSemaphore(value: 0)
        let gsURL = URL(filePath: args[2])
        // Task.detached: top-level code is MainActor in Swift 6, and sem.wait() blocks the
        // main thread — an inherited-context Task would deadlock against it.
        Task.detached {
            do {
                let points = try await AutodetectSceneReader(gsURL).readAll()
                var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
                var op = (min: Float.greatestFiniteMagnitude, max: -Float.greatestFiniteMagnitude)
                for p in points {
                    lo = simd_min(lo, p.position); hi = simd_max(hi, p.position)
                    op.min = min(op.min, p.opacity.asLinearFloat); op.max = max(op.max, p.opacity.asLinearFloat)
                }
                let sh = points.first?.color.shDegree
                print("✓ gscheck: \(points.count) splats, shDegree=\(String(describing: sh))")
                print("  bbox x[\(lo.x),\(hi.x)] y[\(lo.y),\(hi.y)] z[\(lo.z),\(hi.z)]  opacity[\(op.min),\(op.max)]")
            } catch {
                print("✗ gscheck FAILED: \(error)"); exit(1)
            }
            sem.signal()
        }
        sem.wait()
    case "render":
        guard args.count >= 4 else { print("usage: LiToSmoke render <in.ply> <out.png> [tileSize]"); exit(1) }
        let tile = args.count > 4 ? (Int(args[4]) ?? 512) : 512
        try PLYRender.render(plyURL: URL(filePath: args[2]), to: URL(filePath: args[3]), tile: tile)
    case "coreml":
        guard args.count >= 3 else { print("usage: LiToSmoke coreml <model.mlpackage|.mlmodel>"); exit(1) }
        let compiled = try MLModel.compileModel(at: URL(filePath: args[2]))
        let desc = try MLModel(contentsOf: compiled).modelDescription
        print("INPUTS:");  for (n, d) in desc.inputDescriptionsByName  { print("  \(n): \(describeFeature(d))") }
        print("OUTPUTS:"); for (n, d) in desc.outputDescriptionsByName { print("  \(n): \(describeFeature(d))") }
    case "upscale":
        guard args.count >= 3 else { print("usage: LiToSmoke upscale <model>"); exit(1) }
        let up = try Upscaler(modelURL: URL(filePath: args[2]))
        let W = 96, H = 96      // synthetic: R ramps left→right, G top→bottom, B=0 (catches channel swaps)
        var ipx = [UInt8](repeating: 0, count: W * H * 4)
        for y in 0 ..< H { for x in 0 ..< W { let i = (y * W + x) * 4
            ipx[i] = UInt8(x * 255 / W); ipx[i+1] = UInt8(y * 255 / H); ipx[i+2] = 0; ipx[i+3] = 255 } }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bmp = CGImageAlphaInfo.premultipliedLast.rawValue
        let img = CGContext(data: &ipx, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                            space: cs, bitmapInfo: bmp)!.makeImage()!
        let t0 = Date()
        let out = try up.upscale(img)
        print("upscale: \(img.width)×\(img.height) → \(out.width)×\(out.height) in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
        var opx = [UInt8](repeating: 0, count: out.width * out.height * 4)
        CGContext(data: &opx, width: out.width, height: out.height, bitsPerComponent: 8,
                  bytesPerRow: out.width * 4, space: cs, bitmapInfo: bmp)!
            .draw(out, in: CGRect(x: 0, y: 0, width: out.width, height: out.height))
        func meanR(_ x0: Int, _ x1: Int) -> Double { var s = 0.0, n = 0
            for y in stride(from: 0, to: out.height, by: 8) { for x in stride(from: x0, to: x1, by: 8) {
                s += Double(opx[(y * out.width + x) * 4]); n += 1 } }; return s / Double(max(n, 1)) }
        let lR = meanR(0, out.width / 4), rR = meanR(out.width * 3 / 4, out.width)
        print(String(format: "  R(left)=%.0f R(right)=%.0f → %@", lR, rR,
                     rR > lR + 20 ? "✓ channels & orientation correct" : "✗ suspect channel swap/flip"))
    case "smoke":
        Smoke.run(weights: args.count > 2 ? URL(filePath: args[2]) : nil)
    default:
        Smoke.run(weights: URL(filePath: args[1]))   // back-compat: arg1 = weights
    }
} catch {
    print("ERROR: \(error)"); exit(1)
}
