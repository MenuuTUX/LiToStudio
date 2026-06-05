import Foundation
import CoreGraphics
import CoreImage
import ImageIO

/// Analyzes a dropped image and recommends pipeline settings.
public struct RecommendedSettings {
    public var samplingSteps: Double
    public var cfgScale: Double
    public var useRMBG: Bool
    public var useUpscaler: Bool
    public var occupancyThreshold: Double
    public var opacityThreshold: Double
}

public enum ImageAnalyzer {
    /// Analyze the image at the given URL and return recommended pipeline settings.
    public static func recommend(imageURL: URL) -> RecommendedSettings {
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            return defaults
        }

        let w = cg.width, h = cg.height
        let shortSide = min(w, h)

        let hasAlpha = imageHasTransparency(cg)
        let useRMBG = !hasAlpha
        let useUpscaler = shortSide < 1024

        let (edgeDensity, contrast) = analyzeComplexity(cg)

        // Quality-first: the reference default is 20 Heun steps; convergence keeps improving
        // (with diminishing returns) toward ~50. Simple isolated objects converge sooner,
        // busy/intricate subjects (hair, foliage, thin structures) need the extra steps to
        // resolve clean geometry instead of ghost volumes. Never trade quality for time.
        let steps: Double
        if edgeDensity < 0.08 && contrast > 0.5 {
            steps = 25
        } else if edgeDensity > 0.25 || contrast < 0.25 {
            steps = 40
        } else {
            steps = 30
        }

        // CFG 3.0 is the reference optimum. Push slightly harder only when the image is
        // low-contrast (weak conditioning signal); >4 over-sharpens and adds artifacts.
        let cfg: Double = contrast < 0.25 ? 3.5 : 3.0

        // Occupancy stays at the reference cutoff (0 = sigmoid 0.5) so real geometry is never
        // lost — disconnected ghost islands are removed by component pruning instead. Only
        // genuinely busy low-contrast inputs (ghost-prone) get a mild confidence bump.
        let occupancy: Double = (edgeDensity > 0.3 && contrast < 0.3) ? 0.25 : 0.0

        // Opacity cutoff only shapes the point-cloud preview (the splat keeps everything and
        // blends faint gaussians correctly) — keep it gentle so detail survives.
        let opacity: Double = contrast > 0.6 && edgeDensity < 0.15 ? 0.12 : 0.08

        return RecommendedSettings(
            samplingSteps: steps,
            cfgScale: cfg,
            useRMBG: useRMBG,
            useUpscaler: useUpscaler,
            occupancyThreshold: occupancy,
            opacityThreshold: opacity
        )
    }

    private static var defaults: RecommendedSettings {
        RecommendedSettings(samplingSteps: 30, cfgScale: 3.0, useRMBG: true, useUpscaler: true,
                            occupancyThreshold: 0.0, opacityThreshold: 0.10)
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
        // Check top and bottom rows, left and right columns
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

    /// Returns (edgeDensity, contrast) in [0,1].
    /// edgeDensity: fraction of pixels with strong gradients (busy = high).
    /// contrast: normalized std-dev of luminance (isolated subject on bg = high).
    private static func analyzeComplexity(_ img: CGImage) -> (Double, Double) {
        let s = 128
        let cs = CGColorSpaceCreateDeviceGray()
        var pixels = [UInt8](repeating: 0, count: s * s)
        guard let ctx = CGContext(data: &pixels, width: s, height: s,
                                  bitsPerComponent: 8, bytesPerRow: s, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return (0.15, 0.4) }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: s, height: s))

        // Edge density via simple Sobel magnitude
        var edgeCount = 0
        let edgeThreshold: Int = 40
        for y in 1 ..< (s - 1) {
            for x in 1 ..< (s - 1) {
                let gx = Int(pixels[(y - 1) * s + (x + 1)]) - Int(pixels[(y - 1) * s + (x - 1)])
                       + 2 * (Int(pixels[y * s + (x + 1)]) - Int(pixels[y * s + (x - 1)]))
                       + Int(pixels[(y + 1) * s + (x + 1)]) - Int(pixels[(y + 1) * s + (x - 1)])
                let gy = Int(pixels[(y + 1) * s + (x - 1)]) - Int(pixels[(y - 1) * s + (x - 1)])
                       + 2 * (Int(pixels[(y + 1) * s + x]) - Int(pixels[(y - 1) * s + x]))
                       + Int(pixels[(y + 1) * s + (x + 1)]) - Int(pixels[(y - 1) * s + (x + 1)])
                let mag = abs(gx) + abs(gy)
                if mag > edgeThreshold { edgeCount += 1 }
            }
        }
        let edgeDensity = Double(edgeCount) / Double((s - 2) * (s - 2))

        // Contrast via normalized standard deviation
        var sum: Double = 0, sumSq: Double = 0
        let n = Double(s * s)
        for i in 0 ..< (s * s) {
            let v = Double(pixels[i])
            sum += v; sumSq += v * v
        }
        let mean = sum / n
        let variance = sumSq / n - mean * mean
        let stddev = sqrt(max(0, variance))
        let contrast = min(1.0, stddev / 80.0)  // normalize: stddev ~80 = high contrast

        return (edgeDensity, contrast)
    }
}
