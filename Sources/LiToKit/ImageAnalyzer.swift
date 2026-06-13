import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import Vision

/// Analyzes the dropped image(s) and recommends pipeline settings.
public struct RecommendedSettings: Sendable, Codable {
    public var samplingSteps: Double
    public var cfgScale: Double
    public var useRMBG: Bool
    public var useUpscaler: Bool
    public var occupancyThreshold: Double
    public var opacityThreshold: Double
    public var seedCandidates: Double
}

/// Estimated subject (foreground) bounding box in original pixels, top-left origin.
public struct SubjectBox: Sendable, Codable, Equatable {
    public let x: Int, y: Int, width: Int, height: Int
    public var longSide: Int { max(width, height) }
}

/// Camera-relative subject orientation, estimated from Vision body-pose landmark
/// visibility. A heuristic: reported with a confidence and labeled "estimated" in UI.
public enum ViewOrientation: String, Sendable, Codable {
    case front, frontObliqueLeft, frontObliqueRight
    case profileLeft, profileRight, back, unknown

    public var label: String {
        switch self {
        case .front: return "front"
        case .frontObliqueLeft: return "front-left oblique"
        case .frontObliqueRight: return "front-right oblique"
        case .profileLeft: return "left profile"
        case .profileRight: return "right profile"
        case .back: return "back"
        case .unknown: return "unknown"
        }
    }
}

/// How much of the subject the frame covers, classified from which Vision body-pose
/// joints are confidently visible (ankles ⇒ full body, knees ⇒ head-to-knees, …).
public enum ViewFraming: String, Sendable, Codable {
    case faceCrop = "face_crop"
    case upperBody = "upper_body"
    case torso
    case headToKnees = "head_to_knees"
    case fullBody = "full_body"
    case unknown

    public var label: String {
        switch self {
        case .faceCrop: return "face crop"
        case .upperBody: return "upper body"
        case .torso: return "torso (waist-up)"
        case .headToKnees: return "head to knees"
        case .fullBody: return "full body"
        case .unknown: return "unknown"
        }
    }
}

/// Measured per-image statistics → per-view + global recommended settings, with the
/// working shown.
///
/// Per-view measurements (downsampled, EXIF-upright):
///   • edge density ε — fraction of pixels with Sobel |Gx|+|Gy| > 40 (of 255) at 128²,
///     reported for the whole frame and within the estimated subject box.
///   • contrast σ — std-dev of luminance / 80, capped at 1.
///   • sharpness λ — variance of the 3×3 Laplacian at 256², normalized by 300.
///   • luminance ℓ — mean luma in [0,1]; below 0.27 the pipeline auto-normalizes.
///   • texture τ — normalized 32-bin gray-histogram entropy inside the subject box.
///   • subject box — alpha bbox when the image is pre-masked, else a border-color
///     distance heuristic; marked unreliable (and unused) when the background is busy.
///   • orientation — Vision body-pose landmark visibility heuristic (front / oblique /
///     profile / back / unknown). Unknown for non-person subjects.
public enum ImageAnalyzer {

    /// The app's untouched slider defaults (what "default" means in the notes).
    public static let defaults = RecommendedSettings(
        samplingSteps: 20, cfgScale: 3.0, useRMBG: true, useUpscaler: true,
        occupancyThreshold: 0.0, opacityThreshold: 0.10, seedCandidates: 1)

    /// The 2K conditioning-quality target: the *subject's* long side, in pixels.
    /// (Conditioning crops to the subject; canvas/background pixels are irrelevant.)
    public static let subjectTargetPx = 2048
    /// Hard canvas cap — memory budget for 6 × RGBA temporaries on 16 GB machines.
    public static let canvasCapPx = 4096

    public struct ViewAnalysis: Sendable, Codable, Identifiable {
        public let index: Int
        public let fileName: String
        public let width: Int
        public let height: Int
        public let premasked: Bool          // transparent border ⇒ already cut out
        public let luminance: Double        // ℓ
        public let contrast: Double         // σ
        public let sharpness: Double        // λ
        public let edgeDensity: Double      // ε, whole frame
        public let subjectEdgeDensity: Double  // ε within the subject box
        public let textureEntropy: Double   // τ in [0,1]
        public let subjectBox: SubjectBox?  // nil when no reliable estimate
        public let subjectEstimateReliable: Bool
        public let maskAreaRatio: Double?   // est. foreground px / frame px
        public let subjectCropRatio: Double?   // bbox long side / frame long side
        public let subjectLongSide: Int     // px (falls back to canvas long side)
        public let orientation: ViewOrientation
        public let orientationConfidence: Double
        public let framing: ViewFraming     // frame coverage (face crop … full body)
        public let raisedHand: String?      // "left"/"right"/"both" when a wrist is above nose height
        public let needsUpscale: Bool
        public let upscaleNote: String      // why upscale is / isn't recommended
        public var id: Int { index }

        /// The pose-feature record exported into the landmark package.
        public var poseFeatures: PoseFeatures {
            PoseFeatures(framing: framing, orientation: orientation,
                         orientationConfidence: orientationConfidence,
                         raisedHand: raisedHand)
        }
    }

    public struct Note: Identifiable, Equatable, Sendable, Codable {
        public var id: String { name }
        public let name: String
        public let defaultValue: String
        public let recommended: String
        public let changed: Bool
        public let reason: String
    }

    public struct Analysis: Sendable, Codable {
        public let views: [ViewAnalysis]
        public let settings: RecommendedSettings
        public let notes: [Note]
        /// One-line digest of the measured statistics.
        public let summary: String
    }

    /// Analyze one or more views of the same subject.
    public static func analyze(imageURLs: [URL]) -> Analysis {
        let views = imageURLs.enumerated().compactMap { measure(url: $1, index: $0) }
        guard !views.isEmpty else {
            return Analysis(views: [], settings: defaults, notes: [],
                            summary: "could not read image — using defaults")
        }
        let V = views.count

        // ── Global aggregates ──
        // Means drive the detail score (a max would saturate on any one busy view —
        // the old formula's failure mode); minima drive the "weakest view needs help"
        // decisions (blur, contrast, low light).
        func mean(_ xs: [Double]) -> Double { xs.reduce(0, +) / Double(xs.count) }
        let detailNorm = mean(views.map { min(1, $0.subjectEdgeDensity / 0.40) })
        let texture = mean(views.map(\.textureEntropy))
        let resScore = mean(views.map { min(1, Double($0.subjectLongSide) / Double(subjectTargetPx)) })
        let lambdaMin = views.map(\.sharpness).min()!
        let sigmaMin = views.map(\.contrast).min()!
        let lumMin = views.map(\.luminance).min()!
        let allPremasked = views.allSatisfy(\.premasked)

        var notes = [Note]()
        var s = defaults

        // Steps: detail score D blends measured subject detail, texture, and usable
        // resolution; views add constraint (each view's silhouette is scored), blur
        // subtracts (soft inputs converge early — extra steps sharpen noise, not shape).
        //   D = 0.45·detail + 0.30·texture + 0.25·resolution
        //   steps = clamp(16 + 26·D + min(8, 2·(V−1)) − 6·blur, 14, 48), rounded to 2
        let D = 0.45 * detailNorm + 0.30 * texture + 0.25 * resScore
        let blurPenalty = min(1, max(0, 1 - lambdaMin / 0.35))
        let viewBonus = min(8.0, 2.0 * Double(V - 1))
        let rawSteps = 16 + 26 * D + viewBonus - 6 * blurPenalty
        s.samplingSteps = (min(48, max(14, rawSteps)) / 2).rounded() * 2
        notes.append(Note(
            name: "Sampling steps",
            defaultValue: "\(Int(defaults.samplingSteps))",
            recommended: "\(Int(s.samplingSteps))",
            changed: s.samplingSteps != defaults.samplingSteps,
            reason: String(format: "D = 0.45·%.2f + 0.30·%.2f + 0.25·%.2f = %.2f → 16 + 26·D + %.0f (views) − %.1f (blur) = %.1f → clamp[14,48]",
                           detailNorm, texture, resScore, D, viewBonus, 6 * blurPenalty, rawSteps)))

        // CFG: 3.0 is the reference optimum; weak conditioning (low contrast) gets a
        // gentle push: cfg = 3 + 2·max(0, 0.25 − σ), never past 3.5 (over-sharpening).
        s.cfgScale = min(3.5, ((3.0 + 2 * max(0, 0.25 - sigmaMin)) * 10).rounded() / 10)
        notes.append(Note(
            name: "Guidance (CFG)",
            defaultValue: String(format: "%.1f", defaults.cfgScale),
            recommended: String(format: "%.1f", s.cfgScale),
            changed: s.cfgScale != defaults.cfgScale,
            reason: String(format: "σmin = %.2f → 3 + 2·max(0, 0.25−σ) = %.1f", sigmaMin, s.cfgScale)))

        // Seed search: the candidate score is the silhouette IoU averaged over V views,
        // so seed-luck noise on the score shrinks like 1/√V.
        s.seedCandidates = V > 1 ? 2 : 3
        notes.append(Note(
            name: "Seed search",
            defaultValue: "best of \(Int(defaults.seedCandidates))",
            recommended: "best of \(Int(s.seedCandidates))",
            changed: s.seedCandidates != defaults.seedCandidates,
            reason: V > 1
                ? "score noise ∝ 1/√V, V = \(V) views → best-of-2 suffices"
                : "single view: seed luck > most quality knobs → best-of-3"))

        // Background removal: skip only when every view already has a transparent border.
        s.useRMBG = !allPremasked
        notes.append(Note(
            name: "RMBG 2.0",
            defaultValue: defaults.useRMBG ? "on" : "off",
            recommended: s.useRMBG ? "on" : "off",
            changed: s.useRMBG != defaults.useRMBG,
            reason: allPremasked ? "all views already have transparent backgrounds"
                                 : "opaque background(s) → RMBG cutout"))

        // Upscaler: per-view decision (see each view's note); global toggle = any view
        // below the 2K subject target or soft.
        let upViews = views.filter(\.needsUpscale)
        s.useUpscaler = !upViews.isEmpty
        notes.append(Note(
            name: "Real-ESRGAN 4x",
            defaultValue: defaults.useUpscaler ? "on" : "off",
            recommended: s.useUpscaler ? "on" : "off",
            changed: s.useUpscaler != defaults.useUpscaler,
            reason: s.useUpscaler
                ? "\(upViews.count)/\(V) view(s) below the \(subjectTargetPx) px subject target or soft"
                : "every subject already ≥ \(subjectTargetPx) px and sharp — skip"))

        // Occupancy: keep the reference cutoff unless the input is ghost-prone
        // (very busy AND low-contrast — the DiT hallucinates wisps there).
        s.occupancyThreshold = (detailNorm > 0.75 && sigmaMin < 0.30) ? 0.25 : 0.0
        notes.append(Note(
            name: "Occupancy cutoff",
            defaultValue: String(format: "%.2f", defaults.occupancyThreshold),
            recommended: String(format: "%.2f", s.occupancyThreshold),
            changed: s.occupancyThreshold != defaults.occupancyThreshold,
            reason: String(format: "ghost-prone iff detail > 0.75 ∧ σmin < 0.30 (%.2f, %.2f) → %@",
                           detailNorm, sigmaMin, s.occupancyThreshold > 0 ? "raise to 0.25" : "keep reference 0")))

        // Opacity: shapes only the point-cloud preview — prune harder when the subject
        // is clean and simple, gentler when detail must survive.
        s.opacityThreshold = (sigmaMin > 0.6 && detailNorm < 0.35) ? 0.12 : 0.08
        notes.append(Note(
            name: "Opacity cutoff",
            defaultValue: String(format: "%.2f", defaults.opacityThreshold),
            recommended: String(format: "%.2f", s.opacityThreshold),
            changed: s.opacityThreshold != defaults.opacityThreshold,
            reason: sigmaMin > 0.6 && detailNorm < 0.35 ? "clean simple subject → prune floaters harder (0.12)"
                                                        : "busy subject → keep faint detail (0.08)"))

        let meanSubj = Int(mean(views.map { Double($0.subjectLongSide) }))
        let summary = String(format: "%d view%@ · detail %.2f τ %.2f res %.2f → D %.2f · λmin %.2f σmin %.2f · subject ~%d px%@",
                             V, V == 1 ? "" : "s", detailNorm, texture, resScore, D,
                             lambdaMin, sigmaMin, meanSubj,
                             lumMin < 0.27 ? " · low-light fix will trigger" : "")
        return Analysis(views: views, settings: s, notes: notes, summary: summary)
    }

    /// Back-compat single-image entry point.
    public static func recommend(imageURL: URL) -> RecommendedSettings {
        analyze(imageURLs: [imageURL]).settings
    }

    // MARK: - per-view measurement

    static func measure(url: URL, index: Int) -> ViewAnalysis? {
        guard let cg = Preprocess.loadCGImageUpright(url) else { return nil }
        let W = cg.width, H = cg.height
        let s = 128
        guard let rgba = rgbaPixels(cg, s) else { return nil }

        // Luma plane + frame statistics.
        var gray = [Double](repeating: 0, count: s * s)
        var sum = 0.0, sumSq = 0.0
        for i in 0 ..< s * s {
            let l = 0.299 * Double(rgba[i * 4]) + 0.587 * Double(rgba[i * 4 + 1]) + 0.114 * Double(rgba[i * 4 + 2])
            gray[i] = l
            sum += l; sumSq += l * l
        }
        let n = Double(s * s)
        let lum = sum / n / 255
        let sd = (sumSq / n - (sum / n) * (sum / n)).squareRoot()
        let contrast = min(1, sd / 80)

        // Sobel edge map (threshold 40/255), kept per-pixel for the subject-box split.
        var edges = [Bool](repeating: false, count: s * s)
        var edgeCount = 0
        for y in 1 ..< (s - 1) {
            for x in 1 ..< (s - 1) {
                let gx = gray[(y - 1) * s + x + 1] - gray[(y - 1) * s + x - 1]
                       + 2 * (gray[y * s + x + 1] - gray[y * s + x - 1])
                       + gray[(y + 1) * s + x + 1] - gray[(y + 1) * s + x - 1]
                let gy = gray[(y + 1) * s + x - 1] - gray[(y - 1) * s + x - 1]
                       + 2 * (gray[(y + 1) * s + x] - gray[(y - 1) * s + x])
                       + gray[(y + 1) * s + x + 1] - gray[(y - 1) * s + x + 1]
                if abs(gx) + abs(gy) > 40 { edges[y * s + x] = true; edgeCount += 1 }
            }
        }
        let eps = Double(edgeCount) / Double((s - 2) * (s - 2))

        // Foreground estimate → subject box, mask ratio, in-subject detail + texture.
        let premasked = imageHasTransparency(cg)
        let fg = foregroundMask(rgba: rgba, s: s, premasked: premasked)
        var box: SubjectBox?
        var maskRatio: Double?
        var cropRatio: Double?
        var subjEps = eps
        var tau = 0.5
        var reliable = false
        if let m = fg {
            var minX = s, minY = s, maxX = -1, maxY = -1, fgCount = 0
            for y in 0 ..< s { for x in 0 ..< s where m[y * s + x] {
                fgCount += 1
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }}
            let ratio = Double(fgCount) / n
            if maxX >= minX, ratio > 0.02, ratio < 0.95 {
                reliable = true
                maskRatio = ratio
                // Map the 128²-grid box back to original pixels (draw was aspect-distorting).
                let bx = minX * W / s, by = minY * H / s
                let bw = (maxX - minX + 1) * W / s, bh = (maxY - minY + 1) * H / s
                box = SubjectBox(x: bx, y: by, width: bw, height: bh)
                cropRatio = Double(max(maxX - minX + 1, maxY - minY + 1)) / Double(s)
                // Edge density inside the box only — what the conditioning crop sees.
                var inEdges = 0, inPix = 0
                var hist = [Int](repeating: 0, count: 32)
                for y in minY ... maxY { for x in minX ... maxX {
                    inPix += 1
                    if edges[y * s + x] { inEdges += 1 }
                    hist[min(31, Int(gray[y * s + x] / 8))] += 1
                }}
                subjEps = Double(inEdges) / Double(max(1, inPix))
                var ent = 0.0
                for c in hist where c > 0 {
                    let p = Double(c) / Double(inPix)
                    ent -= p * log2(p)
                }
                tau = ent / 5.0   // log2(32)
            }
        }

        let lambda = laplacianSharpness(cg)
        let canvasLong = max(W, H)
        let subjLong = box.map { $0.longSide } ?? canvasLong
        let (orient, orientConf, framing, raisedHand) = estimatePose(cg)

        // Per-view 2K decision (the pipeline applies the same policy; see TECHNICAL_NOTES).
        let needsUpscale: Bool
        let upNote: String
        if subjLong >= subjectTargetPx && lambda >= 0.20 {
            needsUpscale = false
            upNote = "subject \(subjLong) px ≥ \(subjectTargetPx) and sharp — no upscale"
        } else if canvasLong >= canvasCapPx {
            needsUpscale = false
            upNote = "canvas at \(canvasCapPx) px cap but subject only \(subjLong) px — consider a tighter crop"
        } else if subjLong < subjectTargetPx {
            needsUpscale = true
            upNote = "subject \(subjLong) px < \(subjectTargetPx) → upscale"
        } else {
            needsUpscale = true
            upNote = String(format: "soft (λ = %.2f) → upscale to restore detail", lambda)
        }

        return ViewAnalysis(
            index: index, fileName: url.lastPathComponent,
            width: W, height: H, premasked: premasked,
            luminance: lum, contrast: contrast, sharpness: lambda,
            edgeDensity: eps, subjectEdgeDensity: subjEps, textureEntropy: tau,
            subjectBox: box, subjectEstimateReliable: reliable,
            maskAreaRatio: maskRatio, subjectCropRatio: cropRatio,
            subjectLongSide: subjLong,
            orientation: orient, orientationConfidence: orientConf,
            framing: framing, raisedHand: raisedHand,
            needsUpscale: needsUpscale, upscaleNote: upNote)
    }

    /// Public accessor for the pipeline's 2K skip decision (λ ≥ 0.20 ⇒ sharp enough
    /// to leave a ≥2K subject alone).
    public static func measuredSharpness(_ cg: CGImage) -> Double {
        laplacianSharpness(cg)
    }

    /// Estimated subject box for a standalone CGImage — shared with the pipeline's
    /// 2K upscale decision so the analyzer and the pipeline can't disagree.
    public static func subjectBoxEstimate(_ cg: CGImage) -> SubjectBox? {
        let s = 128
        guard let rgba = rgbaPixels(cg, s),
              let m = foregroundMask(rgba: rgba, s: s, premasked: imageHasTransparency(cg)) else { return nil }
        var minX = s, minY = s, maxX = -1, maxY = -1, fgCount = 0
        for y in 0 ..< s { for x in 0 ..< s where m[y * s + x] {
            fgCount += 1
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
        }}
        let ratio = Double(fgCount) / Double(s * s)
        guard maxX >= minX, ratio > 0.02, ratio < 0.95 else { return nil }
        return SubjectBox(x: minX * cg.width / s, y: minY * cg.height / s,
                          width: (maxX - minX + 1) * cg.width / s,
                          height: (maxY - minY + 1) * cg.height / s)
    }

    // MARK: - foreground estimate

    /// Cheap foreground mask at s²: the alpha channel when the image is pre-masked,
    /// else color distance from the average border color. Returns nil when the border
    /// is too busy for the heuristic to mean anything (no fake confidence).
    private static func foregroundMask(rgba: [UInt8], s: Int, premasked: Bool) -> [Bool]? {
        var mask = [Bool](repeating: false, count: s * s)
        if premasked {
            for i in 0 ..< s * s { mask[i] = rgba[i * 4 + 3] > 127 }
            return mask
        }
        // Average border color + its spread; a busy border (σ > 45) means "background
        // is not a flat backdrop" and the estimate would be noise.
        var br = 0.0, bg = 0.0, bb = 0.0, cnt = 0.0
        var sq = 0.0
        func sample(_ i: Int) {
            let r = Double(rgba[i * 4]), g = Double(rgba[i * 4 + 1]), b = Double(rgba[i * 4 + 2])
            br += r; bg += g; bb += b; cnt += 1
            sq += (r * r + g * g + b * b) / 3
        }
        for x in 0 ..< s { sample(x); sample((s - 1) * s + x) }
        for y in 1 ..< s - 1 { sample(y * s); sample(y * s + s - 1) }
        br /= cnt; bg /= cnt; bb /= cnt
        let meanSq = (br * br + bg * bg + bb * bb) / 3
        let borderSD = max(0, sq / cnt - meanSq).squareRoot()
        guard borderSD < 45 else { return nil }
        for i in 0 ..< s * s {
            let dr = Double(rgba[i * 4]) - br
            let dg = Double(rgba[i * 4 + 1]) - bg
            let db = Double(rgba[i * 4 + 2]) - bb
            mask[i] = (dr * dr + dg * dg + db * db).squareRoot() > 55
        }
        return mask
    }

    // MARK: - pose estimate (orientation + framing + raised hand)

    /// Vision body-pose landmark visibility → camera-relative orientation, frame
    /// coverage, and a raised-hand flag. Pure heuristics on real Vision output:
    /// face landmarks visible ⇒ front-ish (nose offset from the shoulder midline
    /// splits front vs oblique), a single ear/eye side ⇒ profile, shoulders without
    /// any face landmarks ⇒ back; ankles visible ⇒ full body, knees ⇒ head-to-knees,
    /// hips ⇒ torso, shoulders only ⇒ upper body, face only ⇒ face crop. A wrist
    /// above nose height counts as a raised hand. Non-person subjects ⇒ unknown.
    static func estimatePose(_ cg: CGImage)
        -> (orientation: ViewOrientation, confidence: Double,
            framing: ViewFraming, raisedHand: String?) {
        let req = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        guard (try? handler.perform([req])) != nil, let obs = req.results?.first else {
            return (.unknown, 0, .unknown, nil)
        }
        typealias P = (conf: Double, x: Double, y: Double)
        func joint(_ j: VNHumanBodyPoseObservation.JointName) -> P? {
            guard let p = try? obs.recognizedPoint(j), p.confidence > 0 else { return nil }
            return (Double(p.confidence), Double(p.location.x), Double(p.location.y))
        }
        let thr = 0.3
        func vis(_ p: P?) -> Bool { (p?.conf ?? 0) > thr }
        let nose = joint(.nose), le = joint(.leftEye), re = joint(.rightEye)
        let lea = joint(.leftEar), rea = joint(.rightEar)
        let ls = joint(.leftShoulder), rs = joint(.rightShoulder)
        let lh = joint(.leftHip), rh = joint(.rightHip)
        let lk = joint(.leftKnee), rk = joint(.rightKnee)
        let la = joint(.leftAnkle), ra = joint(.rightAnkle)
        let lw = joint(.leftWrist), rw = joint(.rightWrist)

        let faceVis = [nose, le, re].filter(vis).count
        let shouldersVis = vis(ls) || vis(rs)
        let midX: Double = (vis(ls) && vis(rs)) ? (ls!.x + rs!.x) / 2 : 0.5
        let span: Double = (vis(ls) && vis(rs)) ? abs(ls!.x - rs!.x) : 0.25
        let conf = ([nose, le, re, lea, rea, ls, rs].compactMap { $0?.conf }.reduce(0, +)) / 7.0

        // Orientation
        var orientation = ViewOrientation.unknown
        var oConf = conf * 0.5
        if faceVis >= 2, let nx = nose?.x, span > 0.01 {
            let off = (nx - midX) / span
            if abs(off) <= 0.18, vis(le), vis(re) { orientation = .front }
            else { orientation = off < 0 ? .frontObliqueLeft : .frontObliqueRight }
            oConf = conf
        } else if faceVis <= 1, vis(lea) != vis(rea) || faceVis == 1 {
            let faceX = [lea, rea, nose, le, re].compactMap { vis($0) ? $0?.x : nil }.first ?? midX
            orientation = faceX < midX ? .profileLeft : .profileRight
            oConf = conf
        } else if shouldersVis, faceVis == 0, !vis(lea), !vis(rea) {
            orientation = .back
            oConf = conf
        }

        // Framing: lowest confidently visible joint band decides frame coverage.
        let framing: ViewFraming
        if vis(la) || vis(ra) { framing = .fullBody }
        else if vis(lk) || vis(rk) { framing = .headToKnees }
        else if vis(lh) || vis(rh) { framing = .torso }
        else if shouldersVis { framing = .upperBody }
        else if faceVis >= 2 { framing = .faceCrop }
        else { framing = .unknown }

        // Raised hand: wrist confidently above nose height (Vision y-up).
        var raised: String?
        if let n = nose, vis(nose) {
            let leftUp = vis(lw) && lw!.y > n.y + 0.02
            let rightUp = vis(rw) && rw!.y > n.y + 0.02
            if leftUp && rightUp { raised = "both" }
            else if leftUp { raised = "left" }
            else if rightUp { raised = "right" }
        }
        return (orientation, oConf, framing, raised)
    }

    // MARK: - raster helpers

    private static func rgbaPixels(_ img: CGImage, _ s: Int) -> [UInt8]? {
        var px = [UInt8](repeating: 0, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &px, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: s, height: s))
        return px
    }

    private static func gray(_ img: CGImage, _ s: Int) -> [UInt8]? {
        var px = [UInt8](repeating: 0, count: s * s)
        guard let ctx = CGContext(data: &px, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: s, height: s))
        return px
    }

    /// Variance of the 3×3 Laplacian at 256², normalized by 300 (≈ crisp photo level).
    private static func laplacianSharpness(_ img: CGImage) -> Double {
        let s = 256
        guard let px = gray(img, s) else { return 0.5 }
        var sum = 0.0, sumSq = 0.0
        let n = Double((s - 2) * (s - 2))
        for y in 1 ..< (s - 1) {
            for x in 1 ..< (s - 1) {
                let lap = 4 * Int(px[y * s + x])
                    - Int(px[y * s + x - 1]) - Int(px[y * s + x + 1])
                    - Int(px[(y - 1) * s + x]) - Int(px[(y + 1) * s + x])
                let d = Double(lap)
                sum += d; sumSq += d * d
            }
        }
        let mean = sum / n
        return min(1, (sumSq / n - mean * mean) / 300)
    }

    static func imageHasTransparency(_ img: CGImage) -> Bool {
        let alpha = img.alphaInfo
        guard alpha == .first || alpha == .last ||
              alpha == .premultipliedFirst || alpha == .premultipliedLast else {
            return false
        }
        // Sample a border strip — if any border pixels are transparent, the image is pre-masked.
        let sampleSize = min(128, img.width, img.height)
        let cs = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: sampleSize * sampleSize * 4)
        guard let ctx = CGContext(data: &pixels, width: sampleSize, height: sampleSize,
                                  bitsPerComponent: 8, bytesPerRow: sampleSize * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: sampleSize, height: sampleSize))

        var transparentCount = 0
        let threshold = sampleSize * 4 / 10  // 10% of border pixels transparent → pre-masked
        for x in 0 ..< sampleSize {
            if pixels[x * 4 + 3] < 200 { transparentCount += 1 }
            if pixels[((sampleSize - 1) * sampleSize + x) * 4 + 3] < 200 { transparentCount += 1 }
        }
        for y in 0 ..< sampleSize {
            if pixels[(y * sampleSize) * 4 + 3] < 200 { transparentCount += 1 }
            if pixels[(y * sampleSize + sampleSize - 1) * 4 + 3] < 200 { transparentCount += 1 }
        }
        return transparentCount > threshold
    }
}
