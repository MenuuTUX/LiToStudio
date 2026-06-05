import Foundation
import CoreML
import CoreGraphics
import CoreVideo

/// Native CoreML surface-normal estimation using Meta's Sapiens normal ViT
/// (converted via docs/sapiens2_normal_coreml_colab.ipynb — never local Python).
/// Expects `SapiensNormal.mlpackage` (or `.mlmodelc`) in the weights directory.
///
/// Input is the high-res square conditioning crop (`Preprocess.condRGBAPixels`),
/// so the predicted normals are pixel-aligned with the view the shape was
/// generated from. The model's native input is 768×1024 (W×H); since the cond
/// square uses fillRatio 0.8, a person almost always fits a 3:4 vertical window
/// of the square — we feed that window 1:1 (no detail lost) and fall back to
/// letterboxing the full square only for wide poses.
///
/// Output normals stay in the model's own camera-space channel convention —
/// `NormalRefine` calibrates the axis permutation/signs against the mesh itself,
/// so nothing here depends on guessing Sapiens' coordinate frame.
public final class SapiensNormal {
    private let model: MLModel
    private let inputName: String
    private let inputIsImage: Bool
    private let normalizedInput: Bool   // mean/std baked into the model graph?
    private let inW = 768, inH = 1024

    /// Sapiens preprocessing constants (applied on 0–255 RGB) — used only when the
    /// converted model does NOT bake them in (no "lito.normalized-input" metadata).
    /// ImageNet×255, matching sapiens2 configs; v1's rounded values differ by <0.2%.
    private static let mean: [Float] = [123.675, 116.28, 103.53]
    private static let std: [Float] = [58.395, 57.12, 57.375]

    public enum SapiensError: Error, CustomStringConvertible {
        case noInput, badInput, predict, noOutput, gridMismatch
        public var description: String {
            switch self {
            case .noInput: return "SapiensNormal: model has no inputs"
            case .badInput: return "SapiensNormal: unsupported input type/shape"
            case .predict: return "SapiensNormal: prediction failed"
            case .noOutput: return "SapiensNormal: no multiarray output found"
            case .gridMismatch: return "SapiensNormal: normal map grid != refiner grid"
            }
        }
    }

    /// A per-pixel normal map on the square cond grid. `normals` are unit vectors in
    /// the model's camera-space channel order; `valid` marks subject pixels (alpha>0.5
    /// inside the model window) — everything else must be ignored.
    public struct NormalMap {
        public let size: Int
        public var normals: [Float]   // size·size·3
        public var valid: [UInt8]     // size·size
        public init(size: Int, normals: [Float], valid: [UInt8]) {
            self.size = size; self.normals = normals; self.valid = valid
        }
    }

    /// First existing Sapiens model in the weights dir, or nil (feature off).
    public static func locate(weightsDir: URL) -> URL? {
        for name in ["SapiensNormal.mlpackage", "SapiensNormal.mlmodelc", "SapiensNormal.mlmodel"] {
            let u = weightsDir.appending(path: name)
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        return nil
    }

    public init(modelURL: URL) throws {
        let loadURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL
            : try MLModel.compileModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: loadURL, configuration: config)

        let desc = model.modelDescription
        guard let (name, input) = desc.inputDescriptionsByName.first else { throw SapiensError.noInput }
        inputName = name
        inputIsImage = input.type == .image
        let meta = desc.metadata[.creatorDefinedKey] as? [String: String]
        normalizedInput = inputIsImage || meta?["lito.normalized-input"] == "1"
    }

    /// Predict normals for a square cond crop (`condRGBA` = size²·4 floats from
    /// `Preprocess.condRGBAPixels`). Returns a map on the same grid.
    public func predict(condRGBA: [Float], size G: Int) throws -> NormalMap {
        precondition(condRGBA.count == G * G * 4, "condRGBA must be size²·4")

        // Subject bbox from alpha — decides the 3:4 window placement.
        var minX = G, maxX = -1
        for y in 0 ..< G {
            for x in 0 ..< G where condRGBA[(y * G + x) * 4 + 3] > 0.5 {
                minX = min(minX, x); maxX = max(maxX, x)
            }
        }
        guard maxX >= minX else {
            return NormalMap(size: G, normals: [Float](repeating: 0, count: G * G * 3),
                             valid: [UInt8](repeating: 0, count: G * G))
        }

        // Window in cond-grid coords mapped to the 768×1024 model input:
        //   subject fits 3:4 → vertical strip of the square, 1:1 aspect (scale = inH/G)
        //   wide pose        → whole square letterboxed into the top-aligned 768×768
        let bboxW = maxX - minX + 1
        let stripW = G * inW / inH                      // 3:4 strip width in cond pixels
        let winX0: Float, winY0: Float, scale: Float, padY: Float
        if bboxW <= stripW {
            let cx = Float(minX + maxX + 1) / 2
            winX0 = max(0, min(Float(G - stripW), cx - Float(stripW) / 2))
            winY0 = 0
            scale = Float(inH) / Float(G)               // cond px → input px
            padY = 0
        } else {
            winX0 = 0; winY0 = 0
            scale = Float(inW) / Float(G)
            padY = Float(inH - inW) / 2                 // vertical letterbox bars
        }

        // Resample the window into the model input, compositing on neutral gray
        // (Sapiens was trained on real photos — a hard black cutout biases it).
        var input = [Float](repeating: 0.5, count: inW * inH * 3)   // RGB planar-agnostic staging
        for iy in 0 ..< inH {
            let cy = (Float(iy) - padY) / scale + winY0
            guard cy >= 0, cy < Float(G) else { continue }
            for ix in 0 ..< inW {
                let cx = Float(ix) / scale + winX0
                guard cx >= 0, cx < Float(G) else { continue }
                let (r, g, b, a) = Self.bilinearRGBA(condRGBA, G, cx, cy)
                let o = (iy * inW + ix) * 3
                input[o] = a * r + (1 - a) * 0.5
                input[o + 1] = a * g + (1 - a) * 0.5
                input[o + 2] = a * b + (1 - a) * 0.5
            }
        }

        let out = try run(rgb01: input)

        // Map the model output back onto the cond grid.
        var normals = [Float](repeating: 0, count: G * G * 3)
        var valid = [UInt8](repeating: 0, count: G * G)
        let plane = inW * inH
        for cy in 0 ..< G {
            for cx in 0 ..< G {
                let p = cy * G + cx
                guard condRGBA[p * 4 + 3] > 0.5 else { continue }
                let ix = (Float(cx) - winX0) * scale
                let iy = (Float(cy) - winY0) * scale + padY
                guard ix >= 0, ix < Float(inW), iy >= 0, iy < Float(inH) else { continue }
                var n = Self.bilinear3(out, inW, inH, plane, ix, iy)
                let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).squareRoot()
                guard len > 1e-4 else { continue }
                n = (n.0 / len, n.1 / len, n.2 / len)
                normals[p * 3] = n.0; normals[p * 3 + 1] = n.1; normals[p * 3 + 2] = n.2
                valid[p] = 1
            }
        }
        return NormalMap(size: G, normals: normals, valid: valid)
    }

    // MARK: - Model I/O

    /// rgb01: interleaved RGB in [0,1] at inW×inH. Returns planar (3, inH, inW) floats.
    private func run(rgb01: [Float]) throws -> [Float] {
        let provider: MLDictionaryFeatureProvider
        if inputIsImage {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: try Self.makePixelBuffer(rgb01: rgb01, w: inW, h: inH))
            ])
        } else {
            let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: inH), NSNumber(value: inW)],
                                       dataType: .float32)
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            let plane = inW * inH
            for i in 0 ..< plane {
                for c in 0 ..< 3 {
                    let v255 = rgb01[i * 3 + c] * 255
                    ptr[c * plane + i] = normalizedInput ? v255 : (v255 - Self.mean[c]) / Self.std[c]
                }
            }
            provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
        }

        let output = try model.prediction(from: provider)
        var arr: MLMultiArray?
        for name in output.featureNames {
            if let v = output.featureValue(for: name)?.multiArrayValue { arr = v; break }
        }
        guard let o = arr else { throw SapiensError.noOutput }

        // Accept (1,3,H,W) at any resolution — resample to inW×inH if it differs.
        let shape = o.shape.map { $0.intValue }
        guard shape.count >= 3, shape[shape.count - 3] == 3 else { throw SapiensError.noOutput }
        let oh = shape[shape.count - 2], ow = shape[shape.count - 1]
        let raw = Self.readFloats(o, count: 3 * oh * ow)
        if oh == inH, ow == inW { return raw }
        var resized = [Float](repeating: 0, count: 3 * inH * inW)
        let sx = Float(ow) / Float(inW), sy = Float(oh) / Float(inH)
        for c in 0 ..< 3 {
            let src = c * oh * ow, dst = c * inH * inW
            for y in 0 ..< inH {
                for x in 0 ..< inW {
                    let v = Self.bilinear1(raw, base: src, ow, oh, Float(x) * sx, Float(y) * sy)
                    resized[dst + y * inW + x] = v
                }
            }
        }
        return resized
    }

    private static func makePixelBuffer(rgb01: [Float], w: Int, h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
              let buf = pb else { throw SapiensError.badInput }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        let stride = CVPixelBufferGetBytesPerRow(buf)
        let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< h {
            let row = base.advanced(by: y * stride)
            for x in 0 ..< w {
                let i = (y * w + x) * 3
                row[x * 4] = UInt8(max(0, min(255, rgb01[i + 2] * 255)))       // B
                row[x * 4 + 1] = UInt8(max(0, min(255, rgb01[i + 1] * 255)))   // G
                row[x * 4 + 2] = UInt8(max(0, min(255, rgb01[i] * 255)))       // R
                row[x * 4 + 3] = 255
            }
        }
        return buf
    }

    private static func readFloats(_ arr: MLMultiArray, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0 ..< count { out[i] = p[i] }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0 ..< count { out[i] = Float(p[i]) }
        case .double:
            let p = arr.dataPointer.assumingMemoryBound(to: Double.self)
            for i in 0 ..< count { out[i] = Float(p[i]) }
        default:
            for i in 0 ..< count { out[i] = arr[i].floatValue }
        }
        return out
    }

    // MARK: - Sampling helpers

    private static func bilinearRGBA(_ px: [Float], _ G: Int, _ x: Float, _ y: Float) -> (Float, Float, Float, Float) {
        let x0 = max(0, min(G - 1, Int(x))), y0 = max(0, min(G - 1, Int(y)))
        let x1 = min(G - 1, x0 + 1), y1 = min(G - 1, y0 + 1)
        let fx = max(0, min(1, x - Float(x0))), fy = max(0, min(1, y - Float(y0)))
        func at(_ xx: Int, _ yy: Int, _ c: Int) -> Float { px[(yy * G + xx) * 4 + c] }
        func lerp(_ c: Int) -> Float {
            let top = at(x0, y0, c) * (1 - fx) + at(x1, y0, c) * fx
            let bot = at(x0, y1, c) * (1 - fx) + at(x1, y1, c) * fx
            return top * (1 - fy) + bot * fy
        }
        return (lerp(0), lerp(1), lerp(2), lerp(3))
    }

    private static func bilinear3(_ buf: [Float], _ w: Int, _ h: Int, _ plane: Int,
                                  _ x: Float, _ y: Float) -> (Float, Float, Float) {
        let x0 = max(0, min(w - 1, Int(x))), y0 = max(0, min(h - 1, Int(y)))
        let x1 = min(w - 1, x0 + 1), y1 = min(h - 1, y0 + 1)
        let fx = max(0, min(1, x - Float(x0))), fy = max(0, min(1, y - Float(y0)))
        func lerp(_ c: Int) -> Float {
            let b = c * plane
            let top = buf[b + y0 * w + x0] * (1 - fx) + buf[b + y0 * w + x1] * fx
            let bot = buf[b + y1 * w + x0] * (1 - fx) + buf[b + y1 * w + x1] * fx
            return top * (1 - fy) + bot * fy
        }
        return (lerp(0), lerp(1), lerp(2))
    }

    private static func bilinear1(_ buf: [Float], base: Int, _ w: Int, _ h: Int,
                                  _ x: Float, _ y: Float) -> Float {
        let x0 = max(0, min(w - 1, Int(x))), y0 = max(0, min(h - 1, Int(y)))
        let x1 = min(w - 1, x0 + 1), y1 = min(h - 1, y0 + 1)
        let fx = max(0, min(1, x - Float(x0))), fy = max(0, min(1, y - Float(y0)))
        let top = buf[base + y0 * w + x0] * (1 - fx) + buf[base + y0 * w + x1] * fx
        let bot = buf[base + y1 * w + x0] * (1 - fx) + buf[base + y1 * w + x1] * fx
        return top * (1 - fy) + bot * fy
    }
}
