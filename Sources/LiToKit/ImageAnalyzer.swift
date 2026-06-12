import Foundation
import CoreGraphics
import CoreImage
import ImageIO

/// Analyzes the dropped image(s) and recommends pipeline settings.
public struct RecommendedSettings: Sendable {
    public var samplingSteps: Double
    public var cfgScale: Double
    public var useRMBG: Bool
    public var useUpscaler: Bool
    public var occupancyThreshold: Double
    public var opacityThreshold: Double
    public var seedCandidates: Double
}

/// Measured image statistics → recommended settings, with the working shown.
///
/// The measurements (all on downsampled grayscale, EXIF-upright):
///   • edge density ε — fraction of pixels whose Sobel magnitude |Gx|+|Gy| > 40 (of 255)
///     at 128². Busy silhouettes (hair, straps, thin structures) need finer ODE
///     resolution to come out as geometry instead of ghost volume.
///   • contrast σ — std-dev of luminance / 80, capped at 1. Low contrast = weak
///     conditioning signal for DINOv2.
///   • sharpness λ — variance of the 3×3 Laplacian at 256², normalized by 300.
///     Soft/blurry inputs benefit from Real-ESRGAN before conditioning.
///   • luminance ℓ — mean luma in [0,1]; below 0.27 the pipeline auto-normalizes.
///
/// Aggregation over multiple views is worst-case: ε = max (the busiest view dictates
/// step count), σ and λ = min (the weakest view needs the help), resolution = min.
public enum ImageAnalyzer {

    /// The app's untouched slider defaults (what "default" means in the notes).
    public static let defaults = RecommendedSettings(
        samplingSteps: 20, cfgScale: 3.0, useRMBG: true, useUpscaler: true,
        occupancyThreshold: 0.0, opacityThreshold: 0.10, seedCandidates: 1)

    public struct Metrics {
        public let width: Int, height: Int
        public let luminance: Double
        public let edgeDensity: Double
        public let contrast: Double
        public let sharpness: Double
        public let premasked: Bool          // transparent border ⇒ already cut out
    }

    public struct Note: Identifiable, Equatable {
        public var id: String { name }
        public let name: String
        public let defaultValue: String
        public let recommended: String
        public let changed: Bool
        public let reason: String
    }

    public struct Analysis {
        public let metrics: [Metrics]
        public let settings: RecommendedSettings
        public let notes: [Note]
        /// One-line digest of the measured statistics.
        public let summary: String
    }

    /// Analyze one or more views of the same subject.
    public static func analyze(imageURLs: [URL]) -> Analysis {
        let metrics = imageURLs.compactMap(measure)
        guard !metrics.isEmpty else {
            return Analysis(metrics: [], settings: defaults, notes: [],
                            summary: "could not read image — using defaults")
        }
        let V = metrics.count
        let eps = metrics.map(\.edgeDensity).max()!
        let sigma = metrics.map(\.contrast).min()!
        let lambda = metrics.map(\.sharpness).min()!
        let lum = metrics.map(\.luminance).min()!
        let minSide = metrics.map { min($0.width, $0.height) }.min()!
        let allPremasked = metrics.allSatisfy(\.premasked)

        var notes = [Note]()
        var s = defaults

        // Steps: the paper's sampler converges by ~25 steps on simple shapes; busy
        // silhouettes keep improving toward ~45. Linear in ε: steps = 20 + 80·ε,
        // clamped to [25,45], rounded to 5.
        let rawSteps = 20 + 80 * eps
        s.samplingSteps = (min(45, max(25, rawSteps)) / 5).rounded() * 5
        notes.append(Note(
            name: "Sampling steps",
            defaultValue: "\(Int(defaults.samplingSteps))",
            recommended: "\(Int(s.samplingSteps))",
            changed: s.samplingSteps != defaults.samplingSteps,
            reason: String(format: "ε = %.2f → 20 + 80·ε = %.0f → clamp[25,45], round 5 → %.0f",
                           eps, rawSteps, s.samplingSteps)))

        // CFG: 3.0 is the reference optimum; weak conditioning (low contrast) gets a
        // gentle push: cfg = 3 + 2·max(0, 0.25 − σ), never past 3.5 (over-sharpening).
        s.cfgScale = ((3.0 + 2 * max(0, 0.25 - sigma)) * 10).rounded() / 10
        notes.append(Note(
            name: "Guidance (CFG)",
            defaultValue: String(format: "%.1f", defaults.cfgScale),
            recommended: String(format: "%.1f", s.cfgScale),
            changed: s.cfgScale != defaults.cfgScale,
            reason: String(format: "σ = %.2f → 3 + 2·max(0, 0.25−σ) = %.1f", sigma, s.cfgScale)))

        // Seed search: the candidate score is the silhouette IoU averaged over V views,
        // so seed-luck noise on the score shrinks like 1/√V — one view needs best-of-3
        // to beat variance, several views constrain the shape enough for best-of-2.
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

        // Upscaler: conditioning is 518² but photo texture samples up to 4096 — upscale
        // when any view is small (< 1024 short side) or soft (λ < 0.20).
        s.useUpscaler = minSide < 1024 || lambda < 0.20
        notes.append(Note(
            name: "Real-ESRGAN 4x",
            defaultValue: defaults.useUpscaler ? "on" : "off",
            recommended: s.useUpscaler ? "on" : "off",
            changed: s.useUpscaler != defaults.useUpscaler,
            reason: String(format: "min side %d px, sharpness λ = %.2f → %@", minSide, lambda,
                           s.useUpscaler ? (minSide < 1024 ? "small input, upscale"
                                                           : "soft input, restore detail")
                                         : "large and sharp — skip")))

        // Occupancy: keep the reference cutoff unless the input is ghost-prone
        // (busy AND low-contrast — the DiT hallucinates wisps there).
        s.occupancyThreshold = (eps > 0.30 && sigma < 0.30) ? 0.25 : 0.0
        notes.append(Note(
            name: "Occupancy cutoff",
            defaultValue: String(format: "%.2f", defaults.occupancyThreshold),
            recommended: String(format: "%.2f", s.occupancyThreshold),
            changed: s.occupancyThreshold != defaults.occupancyThreshold,
            reason: String(format: "ghost-prone iff ε > 0.30 ∧ σ < 0.30 (ε = %.2f, σ = %.2f) → %@",
                           eps, sigma, s.occupancyThreshold > 0 ? "raise to 0.25" : "keep reference 0")))

        // Opacity: shapes only the point-cloud preview — prune harder when the subject
        // is clean and simple, gentler when detail must survive.
        s.opacityThreshold = (sigma > 0.6 && eps < 0.15) ? 0.12 : 0.08
        notes.append(Note(
            name: "Opacity cutoff",
            defaultValue: String(format: "%.2f", defaults.opacityThreshold),
            recommended: String(format: "%.2f", s.opacityThreshold),
            changed: s.opacityThreshold != defaults.opacityThreshold,
            reason: sigma > 0.6 && eps < 0.15 ? "clean simple subject → prune floaters harder (0.12)"
                                              : "busy subject → keep faint detail (0.08)"))

        let summary = String(format: "%d view%@ · ε = %.2f  σ = %.2f  λ = %.2f  ℓ = %.2f · min %d px%@",
                             V, V == 1 ? "" : "s", eps, sigma, lambda, lum, minSide,
                             lum < 0.27 ? " · low-light fix will trigger" : "")
        return Analysis(metrics: metrics, settings: s, notes: notes, summary: summary)
    }

    /// Back-compat single-image entry point.
    public static func recommend(imageURL: URL) -> RecommendedSettings {
        analyze(imageURLs: [imageURL]).settings
    }

    // MARK: - measurement

    static func measure(_ url: URL) -> Metrics? {
        guard let cg = Preprocess.loadCGImageUpright(url) else { return nil }
        let (eps, sigma, lum) = sobelStats(cg)
        return Metrics(width: cg.width, height: cg.height,
                       luminance: lum,
                       edgeDensity: eps,
                       contrast: sigma,
                       sharpness: laplacianSharpness(cg),
                       premasked: imageHasTransparency(cg))
    }

    private static func gray(_ img: CGImage, _ s: Int) -> [UInt8]? {
        var px = [UInt8](repeating: 0, count: s * s)
        guard let ctx = CGContext(data: &px, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: s, height: s))
        return px
    }

    /// (edge density, contrast, mean luminance) at 128².
    private static func sobelStats(_ img: CGImage) -> (Double, Double, Double) {
        let s = 128
        guard let px = gray(img, s) else { return (0.15, 0.4, 0.5) }
        var edgeCount = 0
        for y in 1 ..< (s - 1) {
            for x in 1 ..< (s - 1) {
                let gx = Int(px[(y - 1) * s + (x + 1)]) - Int(px[(y - 1) * s + (x - 1)])
                       + 2 * (Int(px[y * s + (x + 1)]) - Int(px[y * s + (x - 1)]))
                       + Int(px[(y + 1) * s + (x + 1)]) - Int(px[(y + 1) * s + (x - 1)])
                let gy = Int(px[(y + 1) * s + (x - 1)]) - Int(px[(y - 1) * s + (x - 1)])
                       + 2 * (Int(px[(y + 1) * s + x]) - Int(px[(y - 1) * s + x]))
                       + Int(px[(y + 1) * s + (x + 1)]) - Int(px[(y - 1) * s + (x + 1)])
                if abs(gx) + abs(gy) > 40 { edgeCount += 1 }
            }
        }
        let eps = Double(edgeCount) / Double((s - 2) * (s - 2))
        var sum = 0.0, sumSq = 0.0
        for v in px { let d = Double(v); sum += d; sumSq += d * d }
        let n = Double(s * s)
        let mean = sum / n
        let sd = (sumSq / n - mean * mean).squareRoot()
        return (eps, min(1, sd / 80), mean / 255)
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

    private static func imageHasTransparency(_ img: CGImage) -> Bool {
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
