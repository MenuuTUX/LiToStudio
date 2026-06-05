import Foundation
import CoreImage
import Vision
import MLX

#if canImport(AppKit)
import AppKit
#endif

/// Stage 1 — native preprocessing: foreground-isolate the subject, crop to it,
/// pad to square, resize to 518², and emit straight-RGB + alpha as an MLXArray (518,518,4)
/// in [0,1] — the `cond_rgba` the DINOv2 encoder expects.
public enum Preprocess {
    enum Err: Error { case load, render }

    /// Load a pre-made RGBA image (e.g. from native RMBG 2.0) directly into
    /// the (resolution, resolution, 4) MLXArray the pipeline expects.
    /// Crops to the foreground bbox, pads to square with fillRatio, then resizes to 518².
    public static func condRGBA(rgbaURL: URL, resolution: Int = 518, fillRatio: Float = 0.8) throws -> MLXArray {
        MLXArray(try condRGBAPixels(rgbaURL: rgbaURL, resolution: resolution, fillRatio: fillRatio))
            .reshaped([resolution, resolution, 4])
    }

    /// Same crop/resize as `condRGBA(rgbaURL:)` but returns the raw host buffer
    /// (resolution·resolution·4 floats, straight RGB + alpha, rows top-down).
    /// Used at high resolution (1024+) by the Sapiens normal refiner, which must be
    /// pixel-aligned with the 518² conditioning view the shape was generated from.
    public static func condRGBAPixels(rgbaURL: URL, resolution: Int, fillRatio: Float = 0.8) throws -> [Float] {
        guard let cg = loadCGImage(rgbaURL) else { throw Err.load }
        let ctx = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        let ci = CIImage(cgImage: cg)
        let W = ci.extent.width, H = ci.extent.height

        // Extract alpha as luminance (R=G=B=A) so alphaBBox's red-channel check works.
        let alpha = ci.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
        ])
        let bbox = alphaBBox(alpha, ctx: ctx, w: Int(W), h: Int(H)) ?? CGRect(x: 0, y: 0, width: W, height: H)
        let side = max(bbox.width, bbox.height) / CGFloat(fillRatio)
        let cx = bbox.midX, cy = bbox.midY
        let square = CGRect(x: cx - side / 2, y: cy - side / 2, width: side, height: side)

        let scale = CGFloat(resolution) / side
        let xf = CGAffineTransform(translationX: -square.minX, y: -square.minY).concatenating(.init(scaleX: scale, y: scale))
        let outRect = CGRect(x: 0, y: 0, width: resolution, height: resolution)
        // CIContext.render(toBitmap:) already emits rows top-to-bottom — row 0 is the top of
        // the image, exactly the layout the reference (PIL) cond_rgba uses. No flip.
        return renderRGBA(ci.transformed(by: xf), ctx: ctx, rect: outRect)
    }

    /// Fallback: use Apple Vision for foreground masking (less accurate than RMBG 2.0).
    public static func condRGBA(imageURL: URL, resolution: Int = 518, fillRatio: Float = 0.8) throws -> MLXArray {
        guard let cg = loadCGImage(imageURL) else { throw Err.load }
        let ctx = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        var rgb = CIImage(cgImage: cg)
        let W = rgb.extent.width, H = rgb.extent.height

        // foreground mask (alpha); fall back to fully-opaque if Vision finds nothing
        var alpha = foregroundMask(cg, ctx: ctx) ?? CIImage(color: .white).cropped(to: rgb.extent)
        rgb = rgb.transformed(by: CGAffineTransform(translationX: -rgb.extent.minX, y: -rgb.extent.minY))
        alpha = alpha.transformed(by: CGAffineTransform(translationX: -alpha.extent.minX, y: -alpha.extent.minY))

        // bbox of the subject (from a downscaled alpha), then square-pad with fillRatio
        let bbox = alphaBBox(alpha, ctx: ctx, w: Int(W), h: Int(H)) ?? CGRect(x: 0, y: 0, width: W, height: H)
        let side = max(bbox.width, bbox.height) / CGFloat(fillRatio)
        let cx = bbox.midX, cy = bbox.midY
        let square = CGRect(x: cx - side / 2, y: cy - side / 2, width: side, height: side)

        // cond_rgba = STRAIGHT rgb + alpha (the DINOv2 encoder premultiplies internally, like the
        // Python rembg path). Render both to float buffers at 518².
        let scale = CGFloat(resolution) / side
        let xf = CGAffineTransform(translationX: -square.minX, y: -square.minY).concatenating(.init(scaleX: scale, y: scale))
        let outRect = CGRect(x: 0, y: 0, width: resolution, height: resolution)
        let rgbF = renderRGBA(rgb.transformed(by: xf), ctx: ctx, rect: outRect)
        let aF = renderRGBA(alpha.transformed(by: xf), ctx: ctx, rect: outRect)

        // assemble (518,518,4): straight rgb + alpha (mask luminance).
        // render(toBitmap:) rows are already top-to-bottom — keep them as-is.
        var buf = [Float](repeating: 0, count: resolution * resolution * 4)
        for i in 0 ..< resolution * resolution {
            let p = i * 4
            buf[p] = rgbF[p]; buf[p + 1] = rgbF[p + 1]; buf[p + 2] = rgbF[p + 2]
            buf[p + 3] = aF[p]                                       // alpha = mask luminance
        }
        return MLXArray(buf).reshaped([resolution, resolution, 4])
    }

    // MARK: input conditioning quality

    /// Mean luminance in [0,1] from a 64² grayscale render — cheap low-light detector.
    public static func meanLuminance(_ cg: CGImage) -> Float {
        let s = 64
        var px = [UInt8](repeating: 0, count: s * s)
        guard let ctx = CGContext(data: &px, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: s, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return 0.5 }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: s, height: s))
        var sum = 0; for v in px { sum += Int(v) }
        return Float(sum) / Float(s * s * 255)
    }

    /// Conservative low-light normalization for night photos: exposure lift toward a mid
    /// target, shadow recovery, and mild denoise. DINOv2 can't condition on what it can't
    /// see — without this, night shots produce mushy geometry.
    public static func normalizeLowLight(_ cg: CGImage) -> CGImage? {
        let mean = meanLuminance(cg)
        guard mean > 0.001 else { return nil }
        let ev = max(0, min(2.2, log2(0.42 / mean)))
        var ci = CIImage(cgImage: cg)
        ci = ci.applyingFilter("CIExposureAdjust", parameters: ["inputEV": ev])
        ci = ci.applyingFilter("CIHighlightShadowAdjust", parameters: [
            "inputShadowAmount": 0.6, "inputHighlightAmount": 1.0])
        ci = ci.applyingFilter("CINoiseReduction", parameters: [
            "inputNoiseLevel": 0.02, "inputSharpness": 0.4])
        let ctx = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        return ctx.createCGImage(ci, from: ci.extent)
    }

    /// Trim a background-removed RGBA cutout to Vision's person mask (generously dilated so
    /// hair, held objects and loose clothing survive). Kills mask contamination — mirror
    /// frames, pillows, furniture — that RMBG keeps because it touches the subject.
    /// Returns nil when no confident person is found (keep the RMBG cutout unchanged).
    public static func personTrim(rgba: CGImage, original: CGImage) -> CGImage? {
        let req = VNGeneratePersonSegmentationRequest()
        req.qualityLevel = .accurate
        let handler = VNImageRequestHandler(cgImage: original, options: [:])
        guard (try? handler.perform([req])) != nil,
              let mask = req.results?.first?.pixelBuffer else { return nil }

        let W = rgba.width, H = rgba.height
        // GPU: scale mask to the cutout size and blur-dilate it (clamp → blur ≈ dilation).
        var m = CIImage(cvPixelBuffer: mask)
        m = m.transformed(by: CGAffineTransform(scaleX: CGFloat(W) / m.extent.width,
                                                y: CGFloat(H) / m.extent.height))
        let dilate = max(8.0, 0.03 * Double(min(W, H)))
        m = m.clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": dilate])
            .cropped(to: CGRect(x: 0, y: 0, width: W, height: H))
        let ctx = CIContext(options: [.workingColorSpace: NSNull()])
        guard let maskCG = ctx.createCGImage(m, from: m.extent) else { return nil }

        var maskPx = [UInt8](repeating: 0, count: W * H)
        guard let mctx = CGContext(data: &maskPx, width: W, height: H, bitsPerComponent: 8,
                                   bytesPerRow: W, space: CGColorSpaceCreateDeviceGray(),
                                   bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        mctx.draw(maskCG, in: CGRect(x: 0, y: 0, width: W, height: H))

        // sanity: a real person should cover ≥2% of the frame
        var covered = 0
        for v in maskPx where v > 32 { covered += 1 }
        guard covered > W * H / 50 else { return nil }

        var px = [UInt8](repeating: 0, count: W * H * 4)
        guard let rctx = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8,
                                   bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        rctx.draw(rgba, in: CGRect(x: 0, y: 0, width: W, height: H))
        for i in 0 ..< W * H where maskPx[i] <= 8 {
            px[i * 4] = 0; px[i * 4 + 1] = 0; px[i * 4 + 2] = 0; px[i * 4 + 3] = 0
        }
        guard let outCtx = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8,
                                     bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return outCtx.makeImage()
    }

    // MARK: helpers
    private static func loadCGImage(_ url: URL) -> CGImage? { loadCGImageUpright(url) }

    /// Load an image with its EXIF orientation baked in (iPhone photos are often stored
    /// rotated with an orientation tag — DINOv2 must see the upright pixels).
    public static func loadCGImageUpright(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { return nil }
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let exif = (props?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        guard exif != 1, let o = CGImagePropertyOrientation(rawValue: exif) else { return cg }
        let oriented = CIImage(cgImage: cg).oriented(o)
        let ctx = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        return ctx.createCGImage(oriented, from: oriented.extent) ?? cg
    }

    private static func foregroundMask(_ cg: CGImage, ctx: CIContext) -> CIImage? {
        let req = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        guard (try? handler.perform([req])) != nil,
              let result = req.results?.first,
              let pb = try? result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        else { return nil }
        return CIImage(cvPixelBuffer: pb)
    }

    /// Bounding box of alpha>0.5 via a downscaled render (fast, robust).
    private static func alphaBBox(_ alpha: CIImage, ctx: CIContext, w: Int, h: Int) -> CGRect? {
        let s = 128
        let scaled = alpha.transformed(by: .init(scaleX: CGFloat(s) / CGFloat(w), y: CGFloat(s) / CGFloat(h)))
        let px = renderRGBA(scaled, ctx: ctx, rect: CGRect(x: 0, y: 0, width: s, height: s))
        var x0 = s, y0 = s, x1 = 0, y1 = 0, any = false
        for y in 0 ..< s { for x in 0 ..< s where px[(y * s + x) * 4] > 0.8 {   // reference th_alpha = 0.8
            any = true; x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        } }
        guard any else { return nil }
        // Pixel rows from render(toBitmap:) are top-down; CIImage coords are bottom-up —
        // mirror the row range into CI space or the crop window lands mirrored vertically.
        let fx = CGFloat(w) / CGFloat(s), fy = CGFloat(h) / CGFloat(s)
        return CGRect(x: CGFloat(x0) * fx, y: CGFloat(s - 1 - y1) * fy,
                      width: CGFloat(x1 - x0 + 1) * fx, height: CGFloat(y1 - y0 + 1) * fy)
    }

    private static func renderRGBA(_ image: CIImage, ctx: CIContext, rect: CGRect) -> [Float] {
        let w = Int(rect.width), h = Int(rect.height)
        var buf = [Float](repeating: 0, count: w * h * 4)
        buf.withUnsafeMutableBytes {
            ctx.render(image, toBitmap: $0.baseAddress!, rowBytes: w * 16, bounds: rect,
                       format: .RGBAf, colorSpace: CGColorSpaceCreateDeviceRGB())
        }
        return buf
    }
}
