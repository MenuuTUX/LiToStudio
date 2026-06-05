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
/// Shared photo → splat path for the `engine` and `sculpt` commands: RMBG cutout
/// (with low-light + person-trim conditioning) when the model is present, then the
/// full native engine. Returns the RGBA cutout URL so later stages (Sapiens normal
/// refinement) can reuse the exact conditioning view.
@discardableResult
func runEngineCLI(weightsDir: URL, imagePath: String, outPLY: URL,
                  steps: Int, cfg: Float, seed: UInt64?, bestOf: Int) throws -> (splat: URL, rgba: URL?) {
    var preRGBA: URL?
    var engineInput = URL(filePath: imagePath)
    let rmbgURL = weightsDir.appending(path: "RMBG2.mlpackage")
    if FileManager.default.fileExists(atPath: rmbgURL.path), var cg = loadCG(imagePath) {
        let lum = Preprocess.meanLuminance(cg)
        if lum < 0.27, let fixed = Preprocess.normalizeLowLight(cg) {
            cg = fixed
            let tmp = outPLY.deletingPathExtension().appendingPathExtension("lowlight.png")
            try writePNG(fixed, to: tmp); engineInput = tmp
            print(String(format: "  [LowLight] mean luminance %.2f → normalized", lum))
        }
        let t = Date()
        var rgba = try RMBG(modelURL: rmbgURL).removeBackground(from: cg)
        if let trimmed = Preprocess.personTrim(rgba: rgba, original: cg) {
            rgba = trimmed
            print("  [PersonTrim] cutout trimmed to person mask")
        }
        let tmp = outPLY.deletingPathExtension().appendingPathExtension("rmbg.png")
        try writePNG(rgba, to: tmp); preRGBA = tmp
        print("  [RMBG] \(cg.width)×\(cg.height) masked in \(String(format: "%.1f", Date().timeIntervalSince(t)))s → \(tmp.lastPathComponent)")
    } else {
        print("  [RMBG] model not found — using Vision foreground fallback")
    }
    let t0 = Date()
    let splatOut = outPLY.deletingPathExtension().appendingPathExtension("gs.ply")
    let n = try LiToEngine(weightsDir: weightsDir).generate(
        imageURL: engineInput, steps: steps, outPLY: outPLY,
        preprocessedRGBA: preRGBA, cfgScale: cfg, seed: seed, seedCandidates: bestOf,
        outSplatPLY: splatOut) { f, s in
        print(String(format: "  [%3.0f%%] %@", f * 100, s))
    }
    print("✓ ENGINE DONE: \(n) colored points, cfg=\(cfg), steps=\(steps) in \(Int(Date().timeIntervalSince(t0)))s → \(outPLY.path) (+ \(splatOut.lastPathComponent))")
    return (splatOut, preRGBA)
}

/// Splat → (optionally Sapiens-refined) mesh, written as PLY + OBJ.
func refineCLI(weightsDir: URL, gsPLY: URL, rgba: URL?, outBase: URL,
               meshRes: Int, iso: Float, grid: Int, selftest: Bool) throws {
    let t0 = Date()
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
        guard args.count >= 5 else { print("usage: LiToSmoke engine <weightsDir> <image> <out.ply> [steps] [cfg] [seed] [bestOf]"); exit(1) }
        let steps = args.count > 5 ? (Int(args[5]) ?? 4) : 4
        let cfg = args.count > 6 ? (Float(args[6]) ?? 3.0) : 3.0
        let seed = args.count > 7 ? UInt64(args[7]) : nil
        let bestOf = args.count > 8 ? (Int(args[8]) ?? 1) : 1
        try runEngineCLI(weightsDir: URL(filePath: args[2]), imagePath: args[3],
                         outPLY: URL(filePath: args[4]), steps: steps, cfg: cfg,
                         seed: seed, bestOf: bestOf)
    case "sculpt":
        // Photo → splat → Sapiens-refined mesh, end to end (the quality pipeline).
        guard args.count >= 5 else {
            print("usage: LiToSmoke sculpt <weightsDir> <image> <out_base> [steps=25] [cfg=3.0] [seed] [bestOf=3] [meshRes=256]")
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
        let (splat, rgba) = try runEngineCLI(weightsDir: weightsDir, imagePath: args[3],
                                             outPLY: pcURL, steps: steps, cfg: cfg,
                                             seed: seed, bestOf: bestOf)
        try refineCLI(weightsDir: weightsDir, gsPLY: splat, rgba: rgba,
                      outBase: base.appendingPathExtension("mesh.ply"),
                      meshRes: meshRes, iso: 0.5, grid: 1024, selftest: selftestFlag)
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
