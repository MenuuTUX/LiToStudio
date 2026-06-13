import Foundation
import CoreML
import CoreGraphics

/// Native SAM 3.1 concept segmentation via the community CoreML conversion
/// (AllanVester/SAM3.1-CoreML-FP16, fp16, SAM License — permissive):
///
///   ImageEncoder  image (1,3,1008,1008)            → FPN pyramid 288²/144²/72² ×256 + vis_pos
///   TextEncoder   token_ids (1,32) CLIP-BPE        → text_features (32,1,256) + text_mask (1,32)
///   Detector      pyramid + vis_pos + text         → 200 queries: scores, boxes, mask logits 288²
///
/// Prompts are the fixed taxonomy core set, pre-tokenized into `prompt_tokens.json`
/// (tools/backend/make_sam3_tokens.py). The reported bbox is derived from the
/// thresholded mask (not the raw box head), which sidesteps box-format ambiguity in
/// the undocumented conversion. The Detector cannot compile for the ANE — all three
/// models load as CPU+GPU.
public final class Sam3CoreML {

    /// Input normalization — the conversion ships no preprocessing docs, so this is
    /// selectable and the default was chosen empirically (`LiToSmoke sam3`).
    public enum Norm: String, CaseIterable, Sendable {
        case clip        // (x/255 − CLIP mean) / CLIP std
        case imagenet    // (x/255 − ImageNet mean) / ImageNet std
        case unit        // x/255
        case signed      // x/255 × 2 − 1
        case raw255      // x as-is (conversion normalizes internally)

        var meanStd: ([Float], [Float]) {
            switch self {
            case .clip: return ([0.48145466, 0.4578275, 0.40821073],
                                [0.26862954, 0.26130258, 0.27577711])
            case .imagenet: return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            case .unit: return ([0, 0, 0], [1, 1, 1])
            case .signed: return ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            case .raw255: let s = Float(1.0 / 255.0); return ([0, 0, 0], [s, s, s])
            }
        }
    }

    public struct PromptTokens: Decodable, Sendable {
        public let id: String
        public let token: String
        public let phrase: String
        public let tokenIds: [Int]
    }
    struct PromptFile: Decodable {
        let tokenizer: String
        let prompts: [PromptTokens]
    }

    /// One prompt's best instance in one image.
    public struct Detection: Sendable {
        public let promptID: String
        public let token: String
        public let confidence: Float
        /// Normalized xywh (top-left origin) — tight bbox of the thresholded mask.
        public let box: [Double]
        /// 288² thresholded mask (white = instance), in the model's square frame.
        public let maskPixels: [UInt8]
        public let maskSide: Int
        /// Mask area as a fraction of the whole frame (diagnostic / gating).
        public let coverage: Float
        /// Mask area as a fraction of the person silhouette (when a person mask was
        /// supplied) — the real "is this a part or the whole person?" signal.
        public let personCoverage: Float
    }

    public enum Sam3Error: Error, CustomStringConvertible {
        case missingFile(String), badOutput(String), noPrompts
        public var description: String {
            switch self {
            case .missingFile(let f): return "sam3-coreml: missing \(f)"
            case .badOutput(let m): return "sam3-coreml: unexpected model output — \(m)"
            case .noPrompts: return "sam3-coreml: prompt_tokens.json missing/empty"
            }
        }
    }

    static let inputSide = 1008
    static let maskSide = 288
    static let queryCount = 200
    static let tokenLength = 32

    private let imageEncoder: MLModel
    private let textEncoder: MLModel
    private let detector: MLModel
    public let prompts: [PromptTokens]
    public let norm: Norm
    /// Text features are prompt-fixed — computed once, reused for every view.
    private var textCache = [String: (features: MLMultiArray, mask: MLMultiArray)]()

    /// The models dir (`<weights>/sam3-coreml`) when all three packages + the token
    /// file exist.
    public static func locate(weightsDir: URL) -> URL? {
        let dir = weightsDir.appending(path: "sam3-coreml")
        let needed = ["SAM3.1_ImageEncoder_FP16.mlpackage", "SAM3.1_TextEncoder_FP16.mlpackage",
                      "SAM3.1_Detector_FP16.mlpackage", "prompt_tokens.json"]
        for f in needed where !FileManager.default.fileExists(atPath: dir.appending(path: f).path) {
            return nil
        }
        return dir
    }

    public init(dir: URL,
                norm: Norm = Norm(rawValue: ProcessInfo.processInfo.environment["LITO_SAM3_NORM"] ?? "") ?? .clip) throws {
        func load(_ name: String) throws -> MLModel {
            let url = dir.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Sam3Error.missingFile(name)
            }
            let compiled = try MLModel.compileModel(at: url)
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU      // Detector fails ANE compilation
            return try MLModel(contentsOf: compiled, configuration: config)
        }
        imageEncoder = try load("SAM3.1_ImageEncoder_FP16.mlpackage")
        textEncoder = try load("SAM3.1_TextEncoder_FP16.mlpackage")
        detector = try load("SAM3.1_Detector_FP16.mlpackage")
        let pf = try JSONDecoder().decode(PromptFile.self,
                                          from: Data(contentsOf: dir.appending(path: "prompt_tokens.json")))
        guard !pf.prompts.isEmpty else { throw Sam3Error.noPrompts }
        prompts = pf.prompts
        self.norm = norm
    }

    // MARK: - public API

    /// Default presence floor. The conversion's scores cluster tightly: absent
    /// concepts sit at sigmoid(0) ≈ 0.500; present ones rise above. 0.51 sits in the
    /// gap measured on real multi-view photos (kept ≥ 0.514, rejected ≤ 0.506).
    /// Adjustable via `LITO_SAM3_THRESHOLD`.
    public static var presenceFloor: Float {
        Float(ProcessInfo.processInfo.environment["LITO_SAM3_THRESHOLD"] ?? "") ?? 0.51
    }
    /// A "part" mask that fills more of the person silhouette than this is the
    /// model's whole-person fallback, not a real part — rejected.
    public static let maxPersonCoverage: Float = 0.7

    /// Ground every taxonomy prompt in one image. Returns one entry per prompt;
    /// `Detection` is nil when nothing passes the gates (honest not-detected). The
    /// expensive image encode runs once. `personMask` (288², in the model's squished
    /// frame — see `personMask288`) cleans background speckle out of every mask and
    /// rejects the whole-person fallback; pass nil to fall back to frame-area gating.
    public func detect(image: CGImage, personMask: [Bool]? = nil,
                       threshold: Float = presenceFloor,
                       onlyPrompt: String? = nil) throws -> [(prompt: PromptTokens, detection: Detection?)] {
        let feats = try encodeImage(image)
        var out = [(PromptTokens, Detection?)]()
        for p in prompts {
            if let only = onlyPrompt, p.id != only, p.token != only { continue }
            let text = try encodeText(p)
            let det = try runDetector(feats: feats, text: text, prompt: p,
                                      threshold: threshold, personMask: personMask)
            out.append((p, det))
        }
        return out
    }

    /// Build a person silhouette at the mask grid (288²) from a cutout's alpha, in
    /// the same square frame the model sees (the image is squished to 1008², so the
    /// silhouette must be squished too).
    public static func personMask288(fromCutout cutout: CGImage) -> [Bool]? {
        let s = maskSide
        var px = [UInt8](repeating: 0, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &px, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cutout, in: CGRect(x: 0, y: 0, width: s, height: s))
        var mask = [Bool](repeating: false, count: s * s)
        var fg = 0
        for i in 0 ..< s * s where px[i * 4 + 3] > 127 { mask[i] = true; fg += 1 }
        // A cutout with no transparency (opaque source) tells us nothing — skip.
        return fg > 0 && fg < s * s ? mask : nil
    }

    // MARK: - stages

    struct ImageFeatures {
        let fpn0: MLMultiArray   // (1,256,288,288)
        let fpn1: MLMultiArray   // (1,256,144,144)
        let fpn2: MLMultiArray   // (1,256,72,72)
        let visPos: MLMultiArray // (1,256,72,72)
    }

    private func encodeImage(_ image: CGImage) throws -> ImageFeatures {
        let s = Self.inputSide
        var pixels = [UInt8](repeating: 0, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixels, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw Sam3Error.badOutput("CGContext")
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: s, height: s))

        let (mean, std) = norm.meanStd
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: s), NSNumber(value: s)],
                                   dataType: .float16)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
        let plane = s * s
        for i in 0 ..< plane {
            for c in 0 ..< 3 {
                ptr[c * plane + i] = Float16((Float(pixels[i * 4 + c]) / 255 - mean[c]) / std[c])
            }
        }
        let outp = try imageEncoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)]))

        // Map outputs by shape; the two 72² maps split on the "const" name prefix
        // (the constant one is the positional encoding → Detector's vis_pos).
        var fpn0: MLMultiArray?, fpn1: MLMultiArray?, fpn2: MLMultiArray?, visPos: MLMultiArray?
        for name in outp.featureNames {
            guard let a = outp.featureValue(for: name)?.multiArrayValue, a.shape.count == 4 else { continue }
            switch a.shape[2].intValue {
            case 288: fpn0 = a
            case 144: fpn1 = a
            case 72: if name.hasPrefix("const") { visPos = a } else { fpn2 = a }
            default: break
            }
        }
        guard let f0 = fpn0, let f1 = fpn1, let f2 = fpn2, let vp = visPos else {
            throw Sam3Error.badOutput("image encoder pyramid (got \(Array(outp.featureNames)))")
        }
        return ImageFeatures(fpn0: f0, fpn1: f1, fpn2: f2, visPos: vp)
    }

    private func encodeText(_ p: PromptTokens) throws -> (features: MLMultiArray, mask: MLMultiArray) {
        if let cached = textCache[p.id] { return cached }
        let ids = try MLMultiArray(shape: [1, NSNumber(value: Self.tokenLength)], dataType: .int32)
        let ptr = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0 ..< Self.tokenLength {
            ptr[i] = i < p.tokenIds.count ? Int32(p.tokenIds[i]) : 0
        }
        let outp = try textEncoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["token_ids": MLFeatureValue(multiArray: ids)]))
        var features: MLMultiArray?, mask: MLMultiArray?
        for name in outp.featureNames {
            guard let a = outp.featureValue(for: name)?.multiArrayValue else { continue }
            if a.shape.count == 3 { features = a }
            else if a.shape.count == 2 { mask = a }
        }
        guard let f = features, let m = mask else {
            throw Sam3Error.badOutput("text encoder (got \(Array(outp.featureNames)))")
        }
        // The detector wants the mask as fp16.
        let m16 = try MLMultiArray(shape: m.shape, dataType: .float16)
        let dst = m16.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0 ..< Self.tokenLength { dst[i] = Float16(m[i].floatValue) }
        let result = (f, m16)
        textCache[p.id] = result
        return result
    }

    private func runDetector(feats: ImageFeatures, text: (features: MLMultiArray, mask: MLMultiArray),
                             prompt: PromptTokens, threshold: Float,
                             personMask: [Bool]?) throws -> Detection? {
        let outp = try detector.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "fpn_feat0": MLFeatureValue(multiArray: feats.fpn0),
            "fpn_feat1": MLFeatureValue(multiArray: feats.fpn1),
            "fpn_feat2": MLFeatureValue(multiArray: feats.fpn2),
            "vis_pos": MLFeatureValue(multiArray: feats.visPos),
            "text_features": MLFeatureValue(multiArray: text.features),
            "text_mask": MLFeatureValue(multiArray: text.mask),
        ]))
        var scoresArr: MLMultiArray?, masksArr: MLMultiArray?
        for name in outp.featureNames {
            guard let a = outp.featureValue(for: name)?.multiArrayValue else { continue }
            switch a.shape.count {
            case 2: scoresArr = a                        // (1,200) logits
            case 4: masksArr = a                         // (1,200,288,288) logits
            default: break                               // (1,200,4) raw boxes unused
            }
        }
        guard let sArr = scoresArr, let mArr = masksArr else {
            throw Sam3Error.badOutput("detector (got \(Array(outp.featureNames)))")
        }

        let q = Self.queryCount
        let ms = Self.maskSide
        let sPtr = sArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let mPtr = mArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let personArea = personMask.map { Float($0.lazy.filter { $0 }.count) } ?? 0

        // Rank queries by presence score, then walk down: the top query is often the
        // model's degenerate whole-person fallback, so we take the highest-scoring
        // query whose mask is actually a plausible *part* (after intersecting with
        // the person silhouette to strip background speckle).
        let order = (0 ..< q).sorted { Float(sPtr[$0]) > Float(sPtr[$1]) }
        var fallback: Detection?     // best whole-person-sized candidate, if nothing better

        for idx in order.prefix(12) {
            let conf = 1 / (1 + exp(-Float(sPtr[idx])))
            if conf < threshold { break }                      // ranked: rest are lower

            let offset = idx * ms * ms
            var pixels = [UInt8](repeating: 0, count: ms * ms)
            var minX = ms, minY = ms, maxX = -1, maxY = -1, count = 0, inPerson = 0
            for y in 0 ..< ms {
                for x in 0 ..< ms {
                    let i = y * ms + x
                    guard Float(mPtr[offset + i]) > 0 else { continue }   // sigmoid > 0.5
                    if let pm = personMask, !pm[i] { continue }           // strip background
                    pixels[i] = 255
                    count += 1
                    if personMask?[i] == true { inPerson += 1 }
                    minX = min(minX, x); maxX = max(maxX, x)
                    minY = min(minY, y); maxY = max(maxY, y)
                }
            }
            guard count > 3, maxX >= minX else { continue }    // empty after cleanup

            let coverage = Float(count) / Float(ms * ms)
            let personCoverage = personArea > 0 ? Float(inPerson) / personArea : 0
            let det = Detection(
                promptID: prompt.id, token: prompt.token, confidence: conf,
                box: [Double(minX) / Double(ms), Double(minY) / Double(ms),
                      Double(maxX - minX + 1) / Double(ms), Double(maxY - minY + 1) / Double(ms)],
                maskPixels: pixels, maskSide: ms, coverage: coverage, personCoverage: personCoverage)

            // Whole-person fallback for a part concept → not a real detection. Keep
            // it only as a last resort if no plausible part query exists.
            let tooBig = personMask != nil ? personCoverage > Self.maxPersonCoverage
                                           : coverage > 0.55
            if tooBig {
                if fallback == nil { fallback = det }
                continue
            }
            return det
        }
        // No plausible part found. We deliberately return nil rather than the
        // whole-person fallback — an honest "not detected" beats a wrong region.
        _ = fallback
        return nil
    }

    // MARK: - batch grounding (pipeline entry point, mirrors Sam3Backend.detect)

    /// Ground the core set in every view, writing a binary mask + a highlight
    /// overlay per detection and returning the same result shape as the Python
    /// worker. `cutouts` (RMBG alpha, aligned with `images`) supplies the person
    /// silhouette used to clean masks and reject the whole-person fallback.
    public func ground(images: [URL], labels: [ViewLabel], masksDir: URL,
                       cutouts: [URL?]? = nil,
                       threshold: Float = presenceFloor,
                       cancel: GenCancelToken? = nil) throws -> Sam3RunResult {
        try FileManager.default.createDirectory(at: masksDir, withIntermediateDirectories: true)
        var perView = [[Sam3RunResult.Sam3Finding]]()
        for (v, url) in images.enumerated() {
            if cancel?.isImmediate == true { throw PythonBackend.BackendError.cancelled }
            guard let cg = Preprocess.loadCGImageUpright(url) else {
                perView.append(prompts.map {
                    .init(id: $0.id, token: $0.token, status: "failed", observation: nil)
                })
                continue
            }
            let label = v < labels.count ? labels[v] : .unknown
            // Person silhouette from the cutout (preferred) or the source image itself
            // if it already carries alpha.
            var personMask: [Bool]?
            if let cutURL = cutouts?[v], let cut = Preprocess.loadCGImageUpright(cutURL) {
                personMask = Self.personMask288(fromCutout: cut)
            }
            if personMask == nil { personMask = Self.personMask288(fromCutout: cg) }

            var findings = [Sam3RunResult.Sam3Finding]()
            for (p, det) in try detect(image: cg, personMask: personMask, threshold: threshold) {
                if let det {
                    let maskURL = masksDir.appending(path: "v\(v + 1)_\(p.id).png")
                    let overlayURL = masksDir.appending(path: "v\(v + 1)_\(p.id)_overlay.png")
                    try Self.writeMask(det, to: maskURL, width: cg.width, height: cg.height)
                    try? Self.writeOverlay(det, over: cg, to: overlayURL)
                    findings.append(.init(
                        id: p.id, token: p.token, status: "detected",
                        observation: LandmarkObservation(
                            tokenID: p.id, token: p.token, box: det.box,
                            maskPath: maskURL.path, overlayPath: overlayURL.path,
                            confidence: Double(det.confidence),
                            viewIndex: v, viewLabel: label,
                            personCoverage: Double(det.personCoverage))))
                } else {
                    findings.append(.init(id: p.id, token: p.token,
                                          status: "not_detected", observation: nil))
                }
            }
            perView.append(findings)
        }
        return Sam3RunResult(backend: "SAM 3.1 CoreML fp16 (local, \(norm.rawValue))",
                             perView: perView)
    }

    /// Ground a single free-text concept (the user's text guidance) across views.
    /// Returns one observation per view (nil = not present). `tokenIds` come from the
    /// CLIP tokenizer worker; `id` namespaces the mask/overlay files.
    public func groundConcept(phrase: String, tokenIds: [Int], id: String,
                              images: [URL], labels: [ViewLabel], masksDir: URL,
                              cutouts: [URL?]? = nil, threshold: Float = presenceFloor,
                              cancel: GenCancelToken? = nil) throws -> [LandmarkObservation?] {
        try FileManager.default.createDirectory(at: masksDir, withIntermediateDirectories: true)
        let concept = PromptTokens(id: id, token: phrase, phrase: phrase, tokenIds: tokenIds)
        let text = try encodeText(concept)
        var out = [LandmarkObservation?]()
        for (v, url) in images.enumerated() {
            if cancel?.isImmediate == true { throw PythonBackend.BackendError.cancelled }
            guard let cg = Preprocess.loadCGImageUpright(url) else { out.append(nil); continue }
            var personMask: [Bool]?
            if let cutURL = cutouts?[v], let cut = Preprocess.loadCGImageUpright(cutURL) {
                personMask = Self.personMask288(fromCutout: cut)
            }
            if personMask == nil { personMask = Self.personMask288(fromCutout: cg) }
            let feats = try encodeImage(cg)
            guard let det = try runDetector(feats: feats, text: text, prompt: concept,
                                            threshold: threshold, personMask: personMask) else {
                out.append(nil); continue
            }
            let maskURL = masksDir.appending(path: "v\(v + 1)_\(id).png")
            let overlayURL = masksDir.appending(path: "v\(v + 1)_\(id)_overlay.png")
            try Self.writeMask(det, to: maskURL, width: cg.width, height: cg.height)
            try? Self.writeOverlay(det, over: cg, to: overlayURL)
            out.append(LandmarkObservation(
                tokenID: id, token: phrase, box: det.box,
                maskPath: maskURL.path, overlayPath: overlayURL.path,
                confidence: Double(det.confidence),
                viewIndex: v, viewLabel: v < labels.count ? labels[v] : .unknown,
                personCoverage: Double(det.personCoverage)))
        }
        return out
    }

    // MARK: - mask export

    /// Write a detection's mask as a grayscale PNG scaled to `width`×`height`.
    public static func writeMask(_ det: Detection, to url: URL, width: Int, height: Int) throws {
        var px = det.maskPixels
        let ms = det.maskSide
        let cs = CGColorSpaceCreateDeviceGray()
        guard let small = CGContext(data: &px, width: ms, height: ms, bitsPerComponent: 8,
                                    bytesPerRow: ms, space: cs,
                                    bitmapInfo: CGImageAlphaInfo.none.rawValue)?.makeImage(),
              let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            throw Sam3Error.badOutput("mask context")
        }
        // Bitmap-context memory row 0 = visual top throughout this codebase — the
        // buffer and the draw are both top-down, no flip.
        ctx.interpolationQuality = .medium
        ctx.draw(small, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw Sam3Error.badOutput("mask png")
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { throw Sam3Error.badOutput("mask png write") }
    }

    /// Render the detection over the original photo: the region keeps full color, the
    /// rest is dimmed, and the bounding box is outlined. This is what the UI shows —
    /// a raw white-on-black mask is meaningless without the photo behind it.
    /// Everything is composited directly in a top-down RGBA buffer (row 0 = visual
    /// top, the convention used across this codebase) so there are no flip surprises.
    public static func writeOverlay(_ det: Detection, over original: CGImage, to url: URL) throws {
        let w = original.width, h = original.height
        let ms = det.maskSide
        let cs = CGColorSpaceCreateDeviceRGB()
        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw Sam3Error.badOutput("overlay context")
        }
        ctx.draw(original, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Inside the mask → keep colour with a gentle green lift; outside → dim to 32 %.
        for y in 0 ..< h {
            let my = min(ms - 1, y * ms / h)
            for x in 0 ..< w {
                let mx = min(ms - 1, x * ms / w)
                let o = (y * w + x) * 4
                if det.maskPixels[my * ms + mx] > 0 {
                    px[o + 1] = UInt8(min(255, Int(px[o + 1]) + 30))
                } else {
                    px[o] = UInt8(Int(px[o]) * 32 / 100)
                    px[o + 1] = UInt8(Int(px[o + 1]) * 32 / 100)
                    px[o + 2] = UInt8(Int(px[o + 2]) * 32 / 100)
                }
            }
        }

        // Bounding-box outline, drawn straight into the buffer (top-down coords).
        let bx0 = max(0, Int(det.box[0] * Double(w))), by0 = max(0, Int(det.box[1] * Double(h)))
        let bx1 = min(w - 1, bx0 + Int(det.box[2] * Double(w)))
        let by1 = min(h - 1, by0 + Int(det.box[3] * Double(h)))
        let lw = max(2, w / 320)
        func plot(_ x: Int, _ y: Int) {
            guard x >= 0, x < w, y >= 0, y < h else { return }
            let o = (y * w + x) * 4
            px[o] = 90; px[o + 1] = 242; px[o + 2] = 140
        }
        if bx1 > bx0, by1 > by0 {
            for t in 0 ..< lw {
                for x in bx0 ... bx1 { plot(x, by0 + t); plot(x, by1 - t) }
                for y in by0 ... by1 { plot(bx0 + t, y); plot(bx1 - t, y) }
            }
        }
        guard let out = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw Sam3Error.badOutput("overlay png")
        }
        CGImageDestinationAddImage(dest, out, nil)
        guard CGImageDestinationFinalize(dest) else { throw Sam3Error.badOutput("overlay write") }
    }
}
