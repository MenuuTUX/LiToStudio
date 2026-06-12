import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision

/// Contact-sheet ingestion: split one AI-generated multi-view sheet (several renders
/// of the same subject on a uniform background, often with text labels) into
/// per-figure crops ready for the per-view conditioning chain.
///
/// Pipeline: Vision OCR (.accurate) finds text → boxes that sit on background are
/// erased with the sheet's border color → foreground mask (RMBG 2.0 when installed,
/// else color-distance from the border color) → connected components → expanded
/// bounding boxes merged to a fixpoint → reading-order crops from the full-resolution
/// text-erased original. Background removal proper happens later, *per crop*, where
/// RMBG gets its full 1024² input for each figure instead of 1/6th of a sheet.
public enum SheetSplit {

    public struct Figure {
        public let image: CGImage    // full-res crop, text erased, background intact
        public let rect: CGRect      // crop rect in original pixels (top-left origin)
    }

    /// Returns nil when the image doesn't look like a multi-figure sheet (fewer than
    /// two figures of comparable height) — callers fall back to the single-image path.
    public static func split(image: CGImage, rmbg: RMBG? = nil,
                             maxFigures: Int = 8,
                             log: ((String) -> Void)? = nil) -> [Figure]? {
        let W = image.width, H = image.height
        guard W > 16, H > 16 else { return nil }
        let bg = borderColor(image)

        // Working raster keeps aspect; component analysis never needs full res.
        let ws = min(1024, W)
        let hs = max(1, H * ws / W)
        let rgba = drawRGBA(image, width: ws, height: hs)

        // ── 1. Text: detect, gate on background coverage, erase at full res ──
        let textRects = detectText(image)
        var clean = image
        if !textRects.isEmpty {
            let prelim = colorDistanceMask(rgba: rgba, count: ws * hs, bg: bg)
            let sx = CGFloat(ws) / CGFloat(W), sy = CGFloat(hs) / CGFloat(H)
            // Only erase boxes mostly on background — text printed on the subject
            // (shirt graphics, tattoos) must survive.
            let erasable = textRects.filter { r in
                let wr = CGRect(x: r.minX * sx, y: r.minY * sy, width: r.width * sx, height: r.height * sy)
                return coverage(of: prelim, ws: ws, hs: hs, in: wr) < 0.7
            }
            if !erasable.isEmpty {
                clean = erase(image, rects: erasable, color: bg) ?? image
                log?("text: erased \(erasable.count)/\(textRects.count) boxes")
            }
        }

        // ── 2. Foreground mask on the text-erased sheet ──
        var mask: [UInt8]
        if let rmbg, let m = try? rmbg.alphaMask(for: clean) {
            mask = drawGray(m, width: ws, height: hs)
            log?("mask: RMBG 2.0 (\(W)×\(H))")
        } else {
            let cleanRGBA = clean === image ? rgba : drawRGBA(clean, width: ws, height: hs)
            mask = colorDistanceMask(rgba: cleanRGBA, count: ws * hs, bg: bg)
            log?("mask: border-color fallback")
        }

        // ── 3. Connected components → bounding boxes ──
        var boxes = components(mask: mask, ws: ws, hs: hs,
                               minArea: max(64, ws * hs / 700))
        guard !boxes.isEmpty else { return nil }

        // Merge boxes that overlap once expanded — reconnects hair wisps and
        // accessories that the mask separates from their figure by a thin gap.
        let gap = CGFloat(max(4, ws / 80))
        boxes = mergeBoxes(boxes, gap: gap)

        // ── 4. Sheet decision: ≥2 figures of comparable height ──
        let maxH = boxes.map(\.height).max() ?? 0
        boxes = boxes.filter { $0.height >= 0.5 * maxH }
        guard boxes.count >= 2 else { return nil }
        if boxes.count > maxFigures {
            boxes = boxes.sorted { $0.width * $0.height > $1.width * $1.height }
            boxes = Array(boxes.prefix(maxFigures))
        }

        // Reading order: band rows by center-y, left→right within a row.
        boxes.sort { $0.midY < $1.midY }
        var rows = [[CGRect]]()
        for b in boxes {
            if var row = rows.last, let first = row.first, abs(b.midY - first.midY) < 0.5 * maxH {
                row.append(b); rows[rows.count - 1] = row
            } else {
                rows.append([b])
            }
        }
        let ordered = rows.flatMap { $0.sorted { $0.midX < $1.midX } }

        // ── 5. Crop full-res with padding ──
        let scale = CGFloat(W) / CGFloat(ws)
        var figures = [Figure]()
        for b in ordered {
            var r = CGRect(x: b.minX * scale, y: b.minY * scale,
                           width: b.width * scale, height: b.height * scale)
            let pad = 0.04 * max(r.width, r.height)
            r = r.insetBy(dx: -pad, dy: -pad)
                .intersection(CGRect(x: 0, y: 0, width: W, height: H))
                .integral
            guard r.width >= 32, r.height >= 32, let crop = clean.cropping(to: r) else { continue }
            figures.append(Figure(image: crop, rect: r))
        }
        guard figures.count >= 2 else { return nil }
        log?("split: \(figures.count) figures " + figures.map {
            "(\(Int($0.rect.width))×\(Int($0.rect.height)))"
        }.joined(separator: " "))
        return figures
    }

    /// Convenience: split and write `view_01.png …` into `outDir`. Nil = not a sheet.
    public static func splitToFiles(imageURL: URL, rmbg: RMBG? = nil, outDir: URL,
                                    log: ((String) -> Void)? = nil) throws -> [URL]? {
        guard let cg = Preprocess.loadCGImageUpright(imageURL) else { return nil }
        guard let figures = split(image: cg, rmbg: rmbg, log: log) else { return nil }
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        var urls = [URL]()
        for (i, f) in figures.enumerated() {
            let url = outDir.appending(path: String(format: "view_%02d.png", i + 1))
            try writePNG(f.image, to: url)
            urls.append(url)
        }
        return urls
    }

    // MARK: - text

    private static func detectText(_ cg: CGImage) -> [CGRect] {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate          // quality over speed, per the use case
        req.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        guard (try? handler.perform([req])) != nil, let results = req.results else { return [] }
        let W = CGFloat(cg.width), H = CGFloat(cg.height)
        var rects = [CGRect]()
        for obs in results where obs.confidence > 0.3 {
            let bb = obs.boundingBox              // normalized, bottom-left origin
            var r = CGRect(x: bb.minX * W, y: (1 - bb.maxY) * H,
                           width: bb.width * W, height: bb.height * H)
            r = r.insetBy(dx: -0.25 * r.height, dy: -0.25 * r.height)
            rects.append(r)
        }
        return rects
    }

    /// Fill the given top-left-origin pixel rects with a solid color.
    private static func erase(_ cg: CGImage, rects: [CGRect], color: (r: UInt8, g: UInt8, b: UInt8)) -> CGImage? {
        let W = cg.width, H = cg.height
        guard let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                  bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        ctx.setFillColor(CGColor(red: CGFloat(color.r) / 255, green: CGFloat(color.g) / 255,
                                 blue: CGFloat(color.b) / 255, alpha: 1))
        for r in rects {
            // context coords are bottom-left origin
            ctx.fill(CGRect(x: r.minX, y: CGFloat(H) - r.maxY, width: r.width, height: r.height))
        }
        return ctx.makeImage()
    }

    // MARK: - masking

    /// Median RGB of the border ring — the sheet's background color.
    private static func borderColor(_ cg: CGImage) -> (r: UInt8, g: UInt8, b: UInt8) {
        let s = 128
        let px = drawRGBA(cg, width: s, height: s)
        var rs = [UInt8](), gs = [UInt8](), bs = [UInt8]()
        for i in 0 ..< s {
            for j in [i, (s - 1) * s + i, i * s, i * s + (s - 1)] {     // top, bottom, left, right
                rs.append(px[j * 4]); gs.append(px[j * 4 + 1]); bs.append(px[j * 4 + 2])
            }
        }
        rs.sort(); gs.sort(); bs.sort()
        return (rs[rs.count / 2], gs[gs.count / 2], bs[bs.count / 2])
    }

    private static func colorDistanceMask(rgba: [UInt8], count: Int,
                                          bg: (r: UInt8, g: UInt8, b: UInt8)) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: count)
        for i in 0 ..< count {
            let dr = abs(Int(rgba[i * 4]) - Int(bg.r))
            let dg = abs(Int(rgba[i * 4 + 1]) - Int(bg.g))
            let db = abs(Int(rgba[i * 4 + 2]) - Int(bg.b))
            mask[i] = max(dr, dg, db) > 28 ? 255 : 0
        }
        return mask
    }

    /// Fraction of mask>127 pixels inside a working-raster rect (top-left origin).
    private static func coverage(of mask: [UInt8], ws: Int, hs: Int, in r: CGRect) -> Float {
        let x0 = max(0, Int(r.minX)), x1 = min(ws - 1, Int(r.maxX))
        let y0 = max(0, Int(r.minY)), y1 = min(hs - 1, Int(r.maxY))
        guard x1 >= x0, y1 >= y0 else { return 0 }
        var on = 0, total = 0
        for y in y0 ... y1 { for x in x0 ... x1 {
            total += 1
            if mask[y * ws + x] > 127 { on += 1 }
        }}
        return total > 0 ? Float(on) / Float(total) : 0
    }

    // MARK: - components

    /// 4-connected components over mask>127; returns bounding boxes (working-raster
    /// coords, top-left origin) of components with ≥ minArea pixels.
    private static func components(mask: [UInt8], ws: Int, hs: Int, minArea: Int) -> [CGRect] {
        var seen = [Bool](repeating: false, count: ws * hs)
        var boxes = [CGRect]()
        var stack = [Int]()
        for start in 0 ..< ws * hs where mask[start] > 127 && !seen[start] {
            var x0 = ws, x1 = 0, y0 = hs, y1 = 0, area = 0
            stack.removeAll(keepingCapacity: true)
            stack.append(start); seen[start] = true
            while let idx = stack.popLast() {
                area += 1
                let y = idx / ws, x = idx % ws
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                for n in [idx - 1, idx + 1, idx - ws, idx + ws] {
                    guard n >= 0, n < ws * hs, !seen[n], mask[n] > 127 else { continue }
                    if n == idx - 1 && x == 0 { continue }
                    if n == idx + 1 && x == ws - 1 { continue }
                    seen[n] = true; stack.append(n)
                }
            }
            if area >= minArea {
                boxes.append(CGRect(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1))
            }
        }
        return boxes
    }

    /// Union boxes whose `gap`-expanded versions intersect, repeated to a fixpoint.
    private static func mergeBoxes(_ input: [CGRect], gap: CGFloat) -> [CGRect] {
        var boxes = input
        var merged = true
        while merged {
            merged = false
            outer: for i in 0 ..< boxes.count {
                for j in (i + 1) ..< boxes.count {
                    if boxes[i].insetBy(dx: -gap, dy: -gap).intersects(boxes[j]) {
                        boxes[i] = boxes[i].union(boxes[j])
                        boxes.remove(at: j)
                        merged = true
                        break outer
                    }
                }
            }
        }
        return boxes
    }

    // MARK: - raster helpers

    private static func drawRGBA(_ cg: CGImage, width: Int, height: Int) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: width * height * 4)
        if let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8,
                               bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return px
    }

    private static func drawGray(_ cg: CGImage, width: Int, height: Int) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: width * height)
        if let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8,
                               bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGImageAlphaInfo.none.rawValue) {
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return px
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        enum E: Error { case png }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw E.png }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw E.png }
    }
}
