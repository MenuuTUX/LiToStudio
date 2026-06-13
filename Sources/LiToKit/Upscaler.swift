import Foundation
import CoreML
import CoreImage
import CoreGraphics
import CoreVideo

/// Native CoreML 4x super-resolution using Real-ESRGAN. Tiles the input to handle
/// arbitrary sizes, blends overlapping regions with linear ramp weights.
public final class Upscaler {
    private let model: MLModel
    private let inputName: String
    private let tileSize: Int
    private let overlap: Int
    private let scale: Int

    public enum UpscalerError: Error { case compile, predict, image }

    /// `modelURL` is a `.mlmodel` / `.mlpackage`. This driver feeds an **image** (CVPixelBuffer,
    /// 32BGRA) input and reads an image output — matching the common Real-ESRGAN CoreML
    /// conversions (512×512 → 2048×2048). `tileSize` defaults to the model's declared input size.
    public init(modelURL: URL, tileSize: Int = 512, overlap: Int = 32) throws {
        let compiled = try MLModel.compileModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all   // ANE-enabled (~5× faster); fp16 here is fine — the result is downsampled to 518²
        self.model = try MLModel(contentsOf: compiled, configuration: config)
        self.inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "input"
        // Honor the model's actual fixed input size if it declares one.
        if let c = model.modelDescription.inputDescriptionsByName[inputName]?.imageConstraint, c.pixelsWide > 0 {
            self.tileSize = c.pixelsWide
        } else {
            self.tileSize = tileSize
        }
        self.overlap = overlap
        self.scale = 4
    }

    /// Upscale a CGImage by 4x, returning the result as a new CGImage.
    public func upscale(_ source: CGImage) throws -> CGImage {
        let srcW = source.width, srcH = source.height
        let dstW = srcW * scale, dstH = srcH * scale
        let srcPixels = extractRGB(source)

        let stride = tileSize - overlap
        let tilesX = max(1, Int(ceil(Double(srcW - overlap) / Double(stride))))
        let tilesY = max(1, Int(ceil(Double(srcH - overlap) / Double(stride))))

        var output = [Float](repeating: 0, count: dstW * dstH * 3)
        var weights = [Float](repeating: 0, count: dstW * dstH)

        for ty in 0 ..< tilesY {
            for tx in 0 ..< tilesX {
                let x = min(tx * stride, max(0, srcW - tileSize))
                let y = min(ty * stride, max(0, srcH - tileSize))
                let tw = min(tileSize, srcW - x)
                let th = min(tileSize, srcH - y)

                let tile = extractTile(srcPixels, srcW: srcW, x: x, y: y, tw: tw, th: th)
                let upscaled = try runTile(tile, tw: tw, th: th)
                blendTile(upscaled, into: &output, weights: &weights,
                          dstW: dstW, x: x * scale, y: y * scale,
                          tw: tw * scale, th: th * scale)
            }
        }

        for i in 0 ..< (dstW * dstH) where weights[i] > 0 {
            output[i * 3] /= weights[i]
            output[i * 3 + 1] /= weights[i]
            output[i * 3 + 2] /= weights[i]
        }

        return try makeCGImage(output, width: dstW, height: dstH)
    }

    /// Upscale to a target maximum dimension (pixels). If the source is already larger, returns as-is.
    public func upscaleToMax(_ source: CGImage, maxDim: Int) throws -> CGImage {
        let srcMax = max(source.width, source.height)
        if srcMax * scale <= maxDim {
            return try upscale(source)
        }
        if srcMax >= maxDim { return source }
        let needed = Double(maxDim) / Double(srcMax)
        if needed <= 1 { return source }
        let upscaled = try upscale(source)
        let targetW = Int(Double(source.width) * needed)
        let targetH = Int(Double(source.height) * needed)
        return downsample(upscaled, toWidth: targetW, height: targetH)
    }

    /// Like `upscaleToMax`, but keeps the source's alpha channel. The model only sees
    /// RGB (a transparent region would land in it as black and the cutout would be
    /// lost), so the alpha plane is resampled separately with high-quality
    /// interpolation and re-attached to the upscaled color. Sources without an alpha
    /// channel take the plain path unchanged.
    public func upscaleToMaxPreservingAlpha(_ source: CGImage, maxDim: Int) throws -> CGImage {
        let rgb = try upscaleToMax(source, maxDim: maxDim)
        let hasAlpha: Bool
        switch source.alphaInfo {
        case .first, .last, .premultipliedFirst, .premultipliedLast: hasAlpha = true
        default: hasAlpha = false
        }
        guard hasAlpha, rgb !== source else { return rgb }
        return Self.attachAlpha(rgb, from: source) ?? rgb
    }

    /// Resample `source`'s alpha plane to `rgb`'s size and combine (premultiplied).
    private static func attachAlpha(_ rgb: CGImage, from source: CGImage) -> CGImage? {
        let w = rgb.width, h = rgb.height
        let cs = CGColorSpaceCreateDeviceRGB()

        // High-quality resample of the source (incl. its alpha) to the output size;
        // only the alpha channel is read from this draw.
        var srcPx = [UInt8](repeating: 0, count: w * h * 4)
        guard let actx = CGContext(data: &srcPx, width: w, height: h, bitsPerComponent: 8,
                                   bytesPerRow: w * 4, space: cs,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        actx.interpolationQuality = .high
        actx.draw(source, in: CGRect(x: 0, y: 0, width: w, height: h))

        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let cctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                   bytesPerRow: w * 4, space: cs,
                                   bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        cctx.draw(rgb, in: CGRect(x: 0, y: 0, width: w, height: h))
        for i in 0 ..< (w * h) {
            let a = Int(srcPx[i * 4 + 3])
            px[i * 4] = UInt8(Int(px[i * 4]) * a / 255)
            px[i * 4 + 1] = UInt8(Int(px[i * 4 + 1]) * a / 255)
            px[i * 4 + 2] = UInt8(Int(px[i * 4 + 2]) * a / 255)
            px[i * 4 + 3] = UInt8(a)
        }
        guard let octx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                   bytesPerRow: w * 4, space: cs,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return octx.makeImage()
    }

    // MARK: - Private

    private func runTile(_ tile: [Float], tw: Int, th: Int) throws -> [Float] {
        // Pack the tile (top-left) into a tileSize² 32BGRA pixel buffer; the rest stays black.
        guard let pb = Self.makePixelBuffer(size: tileSize) else { throw UpscalerError.image }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) {
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            memset(base, 0, bpr * tileSize)                              // black padding
            for y in 0 ..< th {
                for x in 0 ..< tw {
                    let o = y * bpr + x * 4, si = (y * tw + x) * 3
                    base[o + 0] = UInt8(max(0, min(255, tile[si + 2] * 255)))   // B
                    base[o + 1] = UInt8(max(0, min(255, tile[si + 1] * 255)))   // G
                    base[o + 2] = UInt8(max(0, min(255, tile[si + 0] * 255)))   // R
                    base[o + 3] = 255                                           // A
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)])
        let result = try model.prediction(from: provider)
        guard let outKey = result.featureNames.first,
              let outPB = result.featureValue(for: outKey)?.imageBufferValue else {
            throw UpscalerError.predict
        }

        // Read back the valid (tw·scale × th·scale) top-left region as [0,1] RGB.
        let outW = tw * scale, outH = th * scale
        var out = [Float](repeating: 0, count: outW * outH * 3)
        CVPixelBufferLockBaseAddress(outPB, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(outPB)?.assumingMemoryBound(to: UInt8.self) {
            let bpr = CVPixelBufferGetBytesPerRow(outPB)
            for y in 0 ..< outH {
                for x in 0 ..< outW {
                    let o = y * bpr + x * 4, di = (y * outW + x) * 3
                    out[di + 0] = Float(base[o + 2]) / 255   // R (from BGRA)
                    out[di + 1] = Float(base[o + 1]) / 255   // G
                    out[di + 2] = Float(base[o + 0]) / 255   // B
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(outPB, .readOnly)
        return out
    }

    private static func makePixelBuffer(size: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attrs, &pb)
        return pb
    }

    private func extractRGB(_ img: CGImage) -> [Float] {
        let w = img.width, h = img.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: w * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [Float](repeating: 0, count: w * h * 3)
        for i in 0 ..< (w * h) {
            rgb[i * 3] = Float(pixels[i * 4]) / 255.0
            rgb[i * 3 + 1] = Float(pixels[i * 4 + 1]) / 255.0
            rgb[i * 3 + 2] = Float(pixels[i * 4 + 2]) / 255.0
        }
        return rgb
    }

    private func extractTile(_ px: [Float], srcW: Int, x: Int, y: Int, tw: Int, th: Int) -> [Float] {
        var tile = [Float](repeating: 0, count: tw * th * 3)
        for ty in 0 ..< th {
            for tx in 0 ..< tw {
                let si = ((y + ty) * srcW + (x + tx)) * 3
                let di = (ty * tw + tx) * 3
                tile[di] = px[si]; tile[di + 1] = px[si + 1]; tile[di + 2] = px[si + 2]
            }
        }
        return tile
    }

    private func blendTile(_ tile: [Float], into output: inout [Float], weights: inout [Float],
                           dstW: Int, x: Int, y: Int, tw: Int, th: Int) {
        let ov = overlap * scale
        for ty in 0 ..< th {
            let wy: Float = if ty < ov { Float(ty + 1) / Float(ov + 1) }
                            else if ty >= th - ov { Float(th - ty) / Float(ov + 1) }
                            else { 1.0 }
            for tx in 0 ..< tw {
                let wx: Float = if tx < ov { Float(tx + 1) / Float(ov + 1) }
                                else if tx >= tw - ov { Float(tw - tx) / Float(ov + 1) }
                                else { 1.0 }
                let w = wx * wy
                let di = ((y + ty) * dstW + (x + tx))
                let si = (ty * tw + tx) * 3
                output[di * 3] += tile[si] * w
                output[di * 3 + 1] += tile[si + 1] * w
                output[di * 3 + 2] += tile[si + 2] * w
                weights[di] += w
            }
        }
    }

    private func makeCGImage(_ rgb: [Float], width: Int, height: Int) throws -> CGImage {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0 ..< (width * height) {
            pixels[i * 4] = UInt8(max(0, min(255, rgb[i * 3] * 255)))
            pixels[i * 4 + 1] = UInt8(max(0, min(255, rgb[i * 3 + 1] * 255)))
            pixels[i * 4 + 2] = UInt8(max(0, min(255, rgb[i * 3 + 2] * 255)))
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixels, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else { throw UpscalerError.image }
        return img
    }

    private func downsample(_ img: CGImage, toWidth w: Int, height h: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: 0, space: cs,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }
}
