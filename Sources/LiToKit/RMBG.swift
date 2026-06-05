import Foundation
import CoreML
import CoreImage
import CoreGraphics
import CoreVideo

/// Native CoreML background removal using RMBG 2.0 (BiRefNet).
/// Expects `RMBG2.mlpackage` in the weights directory.
/// Input: any CGImage. Output: alpha mask at the original resolution.
public final class RMBG {
    private let model: MLModel
    private static let inputSize = 1024

    public enum RMBGError: Error { case compile, predict, noOutput }

    public init(modelURL: URL) throws {
        let compiled = try MLModel.compileModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU   // force fp32 GPU compute (ANE would run fp16)
        self.model = try MLModel(contentsOf: compiled, configuration: config)
    }

    /// Produce an alpha mask for the given image. Returns a single-channel grayscale CGImage
    /// at the original image's resolution.
    public func alphaMask(for image: CGImage) throws -> CGImage {
        let w = image.width, h = image.height
        let result = try predict(image)
        switch result {
        case .mask(let mask):
            return resizeMask(mask, toWidth: w, height: h)
        case .composited(let rgba):
            return extractAlphaChannel(rgba, toWidth: w, height: h)
        }
    }

    private enum PredictionResult {
        case mask(CGImage)
        case composited(CGImage)
    }

    private func predict(_ image: CGImage) throws -> PredictionResult {
        let inputArray = try prepareInput(image)

        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "input"
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: inputArray)
        ])
        let output = try model.prediction(from: provider)
        return try extractResult(output)
    }

    /// Convenience: apply the mask to the source image, returning an RGBA CGImage.
    public func removeBackground(from image: CGImage) throws -> CGImage {
        let result = try predict(image)
        switch result {
        case .mask(let mask):
            let resized = resizeMask(mask, toWidth: image.width, height: image.height)
            return applyMask(image, mask: resized)
        case .composited(let rgba):
            return rgba
        }
    }

    // MARK: - Private

    private func prepareInput(_ image: CGImage) throws -> MLMultiArray {
        let s = RMBG.inputSize
        let resized = resizeToSquare(image, size: s)

        var pixels = [UInt8](repeating: 0, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixels, width: s, height: s,
                                  bitsPerComponent: 8, bytesPerRow: s * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw RMBGError.compile
        }
        ctx.draw(resized, in: CGRect(x: 0, y: 0, width: s, height: s))

        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: s), NSNumber(value: s)],
                                     dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let planeSize = s * s

        for y in 0 ..< s {
            for x in 0 ..< s {
                let i = y * s + x
                let pi = i * 4
                for c in 0 ..< 3 {
                    let normalized = (Float(pixels[pi + c]) / 255.0 - mean[c]) / std[c]
                    ptr[c * planeSize + i] = normalized
                }
            }
        }
        return array
    }

    private func extractResult(_ output: MLFeatureProvider) throws -> PredictionResult {
        // Check for image output — distinguish RGBA composited from grayscale mask.
        for name in output.featureNames {
            if let pb = output.featureValue(for: name)?.imageBufferValue {
                let pixFmt = CVPixelBufferGetPixelFormatType(pb)
                let hasAlpha = (pixFmt == kCVPixelFormatType_32BGRA ||
                                pixFmt == kCVPixelFormatType_32ARGB ||
                                pixFmt == kCVPixelFormatType_32RGBA)
                guard let img = Self.cgImage(from: pb) else { continue }
                if hasAlpha && img.alphaInfo != .none && img.alphaInfo != .noneSkipLast && img.alphaInfo != .noneSkipFirst {
                    if Self.hasNontrivialAlpha(img) {
                        return .composited(img)
                    }
                }
                return .mask(Self.toGrayscale(img))
            }
        }

        // Multiarray output. BiRefNet / RMBG-2.0 emit raw logits (apply sigmoid);
        // some IS-Net / RMBG-1.4 exports emit an already-[0,1] probability mask.
        // output_3 is the final refined decoder stage in BiRefNet (preds[-1]).
        let candidates = ["output_3", "output_2", "output_1", "output_0", "output"]
        var maskArray: MLMultiArray?
        for name in candidates {
            if let val = output.featureValue(for: name)?.multiArrayValue { maskArray = val; break }
        }
        if maskArray == nil {
            for name in output.featureNames {
                if let val = output.featureValue(for: name)?.multiArrayValue { maskArray = val; break }
            }
        }
        guard let arr = maskArray else { throw RMBGError.noOutput }

        let totalElements = arr.shape.reduce(1) { $0 * $1.intValue }
        let side = Int(sqrt(Double(totalElements)))
        let s = side > 0 ? side : RMBG.inputSize
        let count = s * s

        let floats = Self.readMultiArray(arr, count: count)
        var lo: Float = .greatestFiniteMagnitude, hi: Float = -.greatestFiniteMagnitude
        for i in 0 ..< count { lo = min(lo, floats[i]); hi = max(hi, floats[i]) }
        let isProbability = lo >= -0.001 && hi <= 1.001

        var gray = [UInt8](repeating: 0, count: count)
        for i in 0 ..< count {
            let v = isProbability ? floats[i] : 1.0 / (1.0 + exp(-floats[i]))
            // Sharpen the soft mask: a second sigmoid pushes uncertain values
            // (0.3–0.7) firmly toward 0 or 1, cutting background bleed at edges.
            let sharp = 1.0 / (1.0 + exp(-10.0 * (v - 0.5)))
            gray[i] = UInt8(max(0, min(255, sharp * 255)))
        }

        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &gray, width: s, height: s,
                                  bitsPerComponent: 8, bytesPerRow: s, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let img = ctx.makeImage() else { throw RMBGError.noOutput }
        return .mask(refineMask(img))
    }

    private static func readMultiArray(_ arr: MLMultiArray, count: Int) -> [Float] {
        var floats = [Float](repeating: 0, count: count)
        switch arr.dataType {
        case .float32:
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0 ..< count { floats[i] = ptr[i] }
        case .float16:
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0 ..< count { floats[i] = Float(ptr[i]) }
        case .double:
            let ptr = arr.dataPointer.assumingMemoryBound(to: Double.self)
            for i in 0 ..< count { floats[i] = Float(ptr[i]) }
        default:
            for i in 0 ..< count { floats[i] = arr[i].floatValue }
        }
        return floats
    }

    /// Morphological cleanup: open (erode→dilate) removes thin background artifacts,
    /// then a mild blur antialiases the edges.
    private func refineMask(_ mask: CGImage) -> CGImage {
        let ci = CIImage(cgImage: mask)
        let eroded = ci.applyingFilter("CIMorphologyMinimum", parameters: [kCIInputRadiusKey: 2.0])
        let dilated = eroded.applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 2.0])
        let smooth = dilated.applyingGaussianBlur(sigma: 0.8).clamped(to: ci.extent)
        let ctx = CIContext()
        return ctx.createCGImage(smooth, from: ci.extent) ?? mask
    }

    private static func hasNontrivialAlpha(_ img: CGImage) -> Bool {
        let sampleSize = min(64, img.width, img.height)
        let cs = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: sampleSize * sampleSize * 4)
        guard let ctx = CGContext(data: &pixels, width: sampleSize, height: sampleSize,
                                  bitsPerComponent: 8, bytesPerRow: sampleSize * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: sampleSize, height: sampleSize))
        for i in 0 ..< (sampleSize * sampleSize) where pixels[i * 4 + 3] < 250 {
            return true
        }
        return false
    }

    private static func toGrayscale(_ img: CGImage) -> CGImage {
        let w = img.width, h = img.height
        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return img }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage() ?? img
    }

    private func extractAlphaChannel(_ img: CGImage, toWidth w: Int, height h: Int) -> CGImage {
        let srcW = img.width, srcH = img.height
        let cs = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: srcW * srcH * 4)
        guard let ctx = CGContext(data: &pixels, width: srcW, height: srcH,
                                  bitsPerComponent: 8, bytesPerRow: srcW * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return img }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: srcW, height: srcH))

        var gray = [UInt8](repeating: 0, count: srcW * srcH)
        for i in 0 ..< (srcW * srcH) { gray[i] = pixels[i * 4 + 3] }

        let grayCs = CGColorSpaceCreateDeviceGray()
        guard let gCtx = CGContext(data: &gray, width: srcW, height: srcH,
                                   bitsPerComponent: 8, bytesPerRow: srcW, space: grayCs,
                                   bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let mask = gCtx.makeImage() else { return img }
        return resizeMask(mask, toWidth: w, height: h)
    }

    /// CVPixelBuffer → CGImage (for models that output the mask as an image).
    private static func cgImage(from pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        return CIContext().createCGImage(ci, from: ci.extent)
    }

    private func resizeToSquare(_ image: CGImage, size: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: size, height: size,
                            bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()!
    }

    private func resizeMask(_ mask: CGImage, toWidth w: Int, height h: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceGray()
        let ctx = CGContext(data: nil, width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                            bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }

    private func applyMask(_ image: CGImage, mask: CGImage) -> CGImage {
        let w = image.width, h = image.height
        let cs = CGColorSpaceCreateDeviceRGB()

        var rgbPixels = [UInt8](repeating: 0, count: w * h * 4)
        let rgbCtx = CGContext(data: &rgbPixels, width: w, height: h,
                               bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        rgbCtx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        var maskPixels = [UInt8](repeating: 0, count: w * h)
        let grayCs = CGColorSpaceCreateDeviceGray()
        let maskCtx = CGContext(data: &maskPixels, width: w, height: h,
                                bitsPerComponent: 8, bytesPerRow: w, space: grayCs,
                                bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        maskCtx.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))

        var outPixels = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0 ..< (w * h) {
            let a = maskPixels[i]
            let af = Float(a) / 255.0
            outPixels[i * 4]     = UInt8(Float(rgbPixels[i * 4]) * af)
            outPixels[i * 4 + 1] = UInt8(Float(rgbPixels[i * 4 + 1]) * af)
            outPixels[i * 4 + 2] = UInt8(Float(rgbPixels[i * 4 + 2]) * af)
            outPixels[i * 4 + 3] = a
        }

        guard let outCtx = CGContext(data: &outPixels, width: w, height: h,
                                     bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let result = outCtx.makeImage() else { return image }
        return result
    }
}
