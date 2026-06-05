import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Dev-only headless renderer: reads the binary little-endian colored point-cloud `.ply`
/// that `Splat.writePointCloud` emits and rasterizes a few orthographic turntable views
/// into a single PNG montage, so a run can be eyeballed without launching the SwiftUI app.
/// Convention matches the SceneKit viewer: +Y up, camera on +Z looking at the origin.
/// LiTo world space is +Z up (the reference VTK viewer sets viewUp=[0,0,1]), so parse()
/// remaps (x, y, z) → (x, z, −y) to stand models upright, same as the app viewer.
enum PLYRender {
    struct Pt { var x, y, z: Float; var r, g, b: UInt8 }

    static func render(plyURL: URL, to outURL: URL, tile: Int = 512,
                       yaws: [Float] = [0, 90, 180, 270]) throws {
        let data = try Data(contentsOf: plyURL)
        let pts = try parse(data)
        guard !pts.isEmpty else { throw Err.empty }

        // Normalize to fit: center on centroid, scale by the largest half-extent.
        var lo = SIMDish(repeating: Float.greatestFiniteMagnitude)
        var hi = SIMDish(repeating: -Float.greatestFiniteMagnitude)
        for p in pts {
            lo.x = min(lo.x, p.x); lo.y = min(lo.y, p.y); lo.z = min(lo.z, p.z)
            hi.x = max(hi.x, p.x); hi.y = max(hi.y, p.y); hi.z = max(hi.z, p.z)
        }
        let cx = (lo.x + hi.x) / 2, cy = (lo.y + hi.y) / 2, cz = (lo.z + hi.z) / 2
        let half = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z)) / 2
        let inv = half > 0 ? 1 / half : 1
        print("  [render] \(pts.count) pts  bbox x[\(f(lo.x)),\(f(hi.x))] y[\(f(lo.y)),\(f(hi.y))] z[\(f(lo.z)),\(f(hi.z))]")

        let cols = yaws.count
        let W = tile * cols, H = tile
        var px = [UInt8](repeating: 0, count: W * H * 4)
        // light background so dark floaters/ghost geometry stay visible
        for i in 0 ..< W * H { px[i*4] = 232; px[i*4+1] = 232; px[i*4+2] = 236; px[i*4+3] = 255 }

        for (ti, yawDeg) in yaws.enumerated() {
            var depth = [Float](repeating: -Float.greatestFiniteMagnitude, count: tile * tile)
            let th = yawDeg * .pi / 180
            let ct = cos(th), st = sin(th)
            let pad: Float = 0.92                       // leave a margin
            let s = Float(tile) / 2 * pad
            let ox = ti * tile
            for p in pts {
                let x = (p.x - cx) * inv, y = (p.y - cy) * inv, z = (p.z - cz) * inv
                let rx = x * ct + z * st                // yaw about Y
                let rz = -x * st + z * ct
                let u = Int(Float(tile) / 2 + rx * s)
                let v = Int(Float(tile) / 2 - y * s)    // image y is down
                guard u >= 1, u < tile - 1, v >= 1, v < tile - 1 else { continue }
                // 3x3 splat with a per-tile z-buffer (camera on +Z → nearest = larger rz)
                for dv in -1 ... 1 { for du in -1 ... 1 {
                    let uu = u + du, vv = v + dv
                    let di = vv * tile + uu
                    if rz > depth[di] {
                        depth[di] = rz
                        let oi = ((vv) * W + (ox + uu)) * 4
                        px[oi] = p.r; px[oi+1] = p.g; px[oi+2] = p.b; px[oi+3] = 255
                    }
                }}
            }
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8,
                            bytesPerRow: W * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let img = ctx.makeImage()!
        guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Err.write
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { throw Err.write }
        print("  [render] wrote \(W)×\(H) montage (yaws \(yaws.map { Int($0) })) → \(outURL.lastPathComponent)")
    }

    private struct SIMDish { var x, y, z: Float; init(repeating v: Float) { x = v; y = v; z = v } }
    private static func f(_ v: Float) -> String { String(format: "%.2f", v) }

    private static func parse(_ data: Data) throws -> [Pt] {
        // Find end_header
        guard let hdrRange = data.range(of: Data("end_header\n".utf8)) else { throw Err.header }
        let header = String(decoding: data[data.startIndex ..< hdrRange.lowerBound], as: UTF8.self)
        var count = 0
        for line in header.split(separator: "\n") {
            if line.hasPrefix("element vertex") { count = Int(line.split(separator: " ").last ?? "0") ?? 0 }
        }
        guard header.contains("binary_little_endian") else { throw Err.header }
        let stride = 15                                  // 3*float + 3*uchar
        var pts = [Pt](); pts.reserveCapacity(count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: hdrRange.upperBound - data.startIndex)
            for i in 0 ..< count {
                let p = base.advanced(by: i * stride)
                let x = p.loadUnaligned(as: Float.self)
                let y = p.advanced(by: 4).loadUnaligned(as: Float.self)
                let z = p.advanced(by: 8).loadUnaligned(as: Float.self)
                let r = p.advanced(by: 12).load(as: UInt8.self)
                let g = p.advanced(by: 13).load(as: UInt8.self)
                let b = p.advanced(by: 14).load(as: UInt8.self)
                pts.append(Pt(x: x, y: z, z: -y, r: r, g: g, b: b))   // z-up → y-up
            }
        }
        return pts
    }

    enum Err: Error { case header, empty, write }
}
