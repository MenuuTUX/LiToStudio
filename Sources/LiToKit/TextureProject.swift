import Foundation

/// HD photo texture by backprojection: replace generated splat/mesh colors with colors
/// sampled from the input photos themselves. The generation's colors come from a 64³
/// latent — soft, guessed; the photos (upscaled by Real-ESRGAN before cutout) carry the
/// actual pixels. With several views around the subject this recolors most of the
/// surface with measured detail.
///
/// View model (same convention as `LiToEngine.silhouetteIoU` / `NormalRefine`): each
/// view is an orthographic camera at azimuth `yaw` around +z (0 = the conditioning
/// camera at +x), image right u = y·cosθ − x·sinθ, image up v = z, and depth
/// d = −(x·cosθ + y·sinθ) (smaller = closer). Photos are sampled through their
/// bbox-normalized cond-crop mapping, refined per view by matching the projected
/// geometry's bounding box to the photo's alpha bounding box — that absorbs crop/fill
/// differences between views.
///
/// Occlusion: a min-depth buffer is splatted from the surface points per view; a point
/// takes color from a view only if it sits within `depthTolerance` of that view's front
/// surface. Blending weights: view confidence × sampled alpha × |cos(normal, view)|^k.
public enum TextureProject {

    public struct View {
        public let rgbaURL: URL     // straight-alpha cutout (RMBG output of the upscaled photo)
        public let yaw: Float       // camera azimuth (radians; 0 = cond convention, camera at +x)
        public let weight: Float    // view confidence multiplier (e.g. its silhouette IoU)
        public init(rgbaURL: URL, yaw: Float, weight: Float = 1) {
            self.rgbaURL = rgbaURL; self.yaw = yaw; self.weight = weight
        }
    }

    public struct Params {
        /// Photo sampling resolution (the cond-crop is re-rendered at this grid).
        public var grid: Int = 1024
        /// Depth-buffer resolution for the occlusion test.
        public var visGrid: Int = 384
        /// A point must sit within this world-distance of the view's front surface.
        public var depthTolerance: Float = 0.045
        /// Power on |cos(normal, view direction)| — higher favors head-on views.
        public var cosPower: Float = 2
        public init() {}
    }

    enum Err: Error, CustomStringConvertible {
        case parse(String)
        public var description: String { "texture: \(self)" }
    }

    static let worldHalf: Float = 1.04
    static let shC0: Float = 0.28209479177387814

    // MARK: - mesh

    /// Recolor mesh vertices from the photos. Vertices no view can see keep their
    /// generated color. Returns the number of recolored vertices.
    @discardableResult
    public static func recolor(mesh: inout MeshExtract.Mesh, views: [View],
                               params: Params = Params(),
                               log: ((String) -> Void)? = nil) -> Int {
        let V = mesh.vertexCount
        guard V > 0, !views.isEmpty else { return 0 }
        guard let vds = try? buildViews(views, surface: mesh.positions, count: V, params: params, log: log)
        else { return 0 }
        let normals = vertexNormals(mesh)
        let (colors, hit, hits) = blend(points: mesh.positions, normals: normals, count: V,
                                        views: vds, params: params)
        for i in 0 ..< V where hit[i] {
            for c in 0 ..< 3 {
                mesh.colors[i * 3 + c] = UInt8(max(0, min(255, colors[i * 3 + c] * 255)))
            }
        }
        log?("mesh: \(hits)/\(V) vertices took photo color")
        return hits
    }

    // MARK: - splat PLY (3DGS INRIA format, our `writeGaussians` output)

    /// Recolor the gaussians in a 3DGS PLY in place: f_dc gets the backprojected photo
    /// color, f_rest of recolored gaussians is zeroed (the generated view-dependence no
    /// longer matches the new base color). Gaussian normals come from each splat's
    /// shortest principal axis. Returns the number of recolored gaussians.
    @discardableResult
    public static func recolorSplatPLY(at url: URL, views: [View],
                                       params: Params = Params(),
                                       log: ((String) -> Void)? = nil) throws -> Int {
        var data = try Data(contentsOf: url)
        let (count, props, headerEnd) = try plyHeader(data)
        func idx(_ name: String) throws -> Int {
            guard let i = props.firstIndex(of: name) else { throw Err.parse("missing \(name)") }
            return i
        }
        let ox = try idx("x"), oy = try idx("y"), oz = try idx("z")
        let od = try idx("f_dc_0"), oo = try idx("opacity")
        let os = try idx("scale_0"), or0 = try idx("rot_0")
        let rest = props.filter { $0.hasPrefix("f_rest_") }.count
        let oRest = rest > 0 ? try idx("f_rest_0") : -1
        let stride = props.count * 4

        var pos = [Float](repeating: 0, count: count * 3)
        var normals = [Float](repeating: 0, count: count * 3)
        var opacity = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: headerEnd)
            for i in 0 ..< count {
                let r = base.advanced(by: i * stride)
                func f(_ p: Int) -> Float { r.advanced(by: p * 4).loadUnaligned(as: Float.self) }
                pos[i * 3] = f(ox); pos[i * 3 + 1] = f(oy); pos[i * 3 + 2] = f(oz)
                opacity[i] = 1 / (1 + exp(-f(oo)))
                // shortest principal axis ≈ surface normal of a flattened splat
                let s0 = f(os), s1 = f(os + 1), s2 = f(os + 2)        // log scales
                let axis = s0 <= s1 && s0 <= s2 ? 0 : (s1 <= s2 ? 1 : 2)
                // rot_0..3 is wxyz
                let qw = f(or0), qx = f(or0 + 1), qy = f(or0 + 2), qz = f(or0 + 3)
                let n = rotatedAxis(qw: qw, qx: qx, qy: qy, qz: qz, axis: axis)
                normals[i * 3] = n.0; normals[i * 3 + 1] = n.1; normals[i * 3 + 2] = n.2
            }
        }

        let vds = try buildViews(views, surface: pos, count: count, params: params,
                                 minWeight: opacity, log: log)
        let (colors, hit, hits) = blend(points: pos, normals: normals, count: count,
                                        views: vds, params: params)

        data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: headerEnd)
            for i in 0 ..< count where hit[i] {
                let r = base.advanced(by: i * stride)
                for c in 0 ..< 3 {
                    var dc = (colors[i * 3 + c] - 0.5) / shC0
                    withUnsafeBytes(of: &dc) { src in
                        r.advanced(by: (od + c) * 4).copyMemory(from: src.baseAddress!, byteCount: 4)
                    }
                }
                if oRest >= 0 {
                    var zero: Float = 0
                    for j in 0 ..< rest {
                        withUnsafeBytes(of: &zero) { src in
                            r.advanced(by: (oRest + j) * 4).copyMemory(from: src.baseAddress!, byteCount: 4)
                        }
                    }
                }
            }
        }
        try data.write(to: url)
        log?("splat: \(hits)/\(count) gaussians took photo color")
        return hits
    }

    /// Recolor a `x,y,z float + r,g,b uchar` point-cloud PLY (the preview artifact).
    @discardableResult
    public static func recolorPointCloudPLY(at url: URL, views: [View],
                                            params: Params = Params(),
                                            log: ((String) -> Void)? = nil) throws -> Int {
        var data = try Data(contentsOf: url)
        let (count, props, headerEnd) = try plyHeader(data)
        guard props.contains("x"), count > 0 else { return 0 }
        let stride = 15                                   // 3 floats + 3 uchar
        var pos = [Float](repeating: 0, count: count * 3)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: headerEnd)
            for i in 0 ..< count {
                let r = base.advanced(by: i * stride)
                for c in 0 ..< 3 { pos[i * 3 + c] = r.advanced(by: c * 4).loadUnaligned(as: Float.self) }
            }
        }
        let vds = try buildViews(views, surface: pos, count: count, params: params, log: log)
        let (colors, hit, hits) = blend(points: pos, normals: nil, count: count,
                                        views: vds, params: params)
        data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: headerEnd)
            for i in 0 ..< count where hit[i] {
                let r = base.advanced(by: i * stride)
                for c in 0 ..< 3 {
                    r.advanced(by: 12 + c).storeBytes(of: UInt8(max(0, min(255, colors[i * 3 + c] * 255))), as: UInt8.self)
                }
            }
        }
        try data.write(to: url)
        log?("point cloud: \(hits)/\(count) points took photo color")
        return hits
    }

    // MARK: - internals

    struct ViewData {
        var pix: [Float]            // G²·4 straight RGBA, rows top-down
        var G: Int
        var cosT: Float, sinT: Float
        var weight: Float
        var depth: [Float]          // V² min-depth buffer (+inf where empty)
        var V: Int
        var tol: Float              // effective depth tolerance (grows with splat radius)
        // (u,v) world → photo pixel: U = u·s + tu, Vpix = −v·s + tv (rows top-down)
        var s: Float, tu: Float, tv: Float
    }

    /// Load each view's photo grid, build its depth buffer from the surface points, and
    /// fit the world→pixel transform by matching the projected-geometry bbox to the
    /// photo's alpha bbox. `minWeight` (optional, e.g. splat opacity) gates which points
    /// shape the depth buffer.
    private static func buildViews(_ views: [View], surface: [Float], count: Int,
                                   params: Params, minWeight: [Float]? = nil,
                                   log: ((String) -> Void)? = nil) throws -> [ViewData] {
        var out = [ViewData]()
        // The depth buffer is only watertight when every surface point's footprint
        // overlaps its neighbors' — splat each point as a disk whose radius scales with
        // the mean point spacing, or back-surface points leak through the holes and
        // take front colors. The tolerance grows with the footprint (a splatted cell
        // may sit up to ~radius·cell shallower than the true surface there).
        let Vg = params.visGrid
        var gated = count
        if let mw = minWeight {
            gated = 0
            for i in 0 ..< count where mw[i] >= 0.15 { gated += 1 }
        }
        let spacing = Float(Vg) / max(1, Float(gated).squareRoot())     // px between points
        let radius = max(1, min(6, Int((0.7 * spacing).rounded())))
        let cell = 2 * worldHalf / Float(Vg)
        let tol = max(params.depthTolerance, Float(radius) * cell * 1.5)
        for view in views {
            let G = params.grid
            let pix = try Preprocess.condRGBAPixels(rgbaURL: view.rgbaURL, resolution: G)
            var vd = ViewData(pix: pix, G: G, cosT: cos(view.yaw), sinT: sin(view.yaw),
                              weight: max(0, view.weight),
                              depth: [Float](repeating: .greatestFiniteMagnitude, count: Vg * Vg),
                              V: Vg, tol: tol, s: 0, tu: 0, tv: 0)

            // Depth splat + projected-geometry bbox (in u,v world units).
            var u0 = Float.greatestFiniteMagnitude, u1 = -Float.greatestFiniteMagnitude
            var v0 = Float.greatestFiniteMagnitude, v1 = -Float.greatestFiniteMagnitude
            for i in 0 ..< count {
                if let mw = minWeight, mw[i] < 0.15 { continue }
                let x = surface[i * 3], y = surface[i * 3 + 1], z = surface[i * 3 + 2]
                let u = y * vd.cosT - x * vd.sinT
                let v = z
                let d = -(x * vd.cosT + y * vd.sinT)
                u0 = min(u0, u); u1 = max(u1, u); v0 = min(v0, v); v1 = max(v1, v)
                let cx = Int((u + worldHalf) / (2 * worldHalf) * Float(Vg))
                let cy = Int((worldHalf - v) / (2 * worldHalf) * Float(Vg))
                for dy in -radius ... radius { for dx in -radius ... radius {
                    let px = cx + dx, py = cy + dy
                    guard px >= 0, px < Vg, py >= 0, py < Vg else { continue }
                    if d < vd.depth[py * Vg + px] { vd.depth[py * Vg + px] = d }
                }}
            }

            // Photo alpha bbox (pixels, top-down rows).
            var a0 = G, a1 = -1, b0 = G, b1 = -1
            for py in 0 ..< G { for px in 0 ..< G where pix[(py * G + px) * 4 + 3] > 0.5 {
                a0 = min(a0, px); a1 = max(a1, px); b0 = min(b0, py); b1 = max(b1, py)
            }}
            guard a1 > a0, b1 > b0, u1 > u0, v1 > v0 else {
                log?(String(format: "view yaw %.0f° skipped: empty alpha or geometry", view.yaw * 180 / .pi))
                continue
            }
            // One uniform scale (the cond crop preserves aspect), centers matched per axis.
            // v grows up, pixel rows grow down → negative scale on v is folded into tv.
            let su = Float(a1 - a0) / (u1 - u0)
            let sv = Float(b1 - b0) / (v1 - v0)
            vd.s = (su + sv) / 2
            vd.tu = Float(a0 + a1) / 2 - (u0 + u1) / 2 * vd.s
            vd.tv = Float(b0 + b1) / 2 + (v0 + v1) / 2 * vd.s
            out.append(vd)
        }
        return out
    }

    /// Blend photo colors onto points. Returns straight-RGB colors in [0,1], a hit mask,
    /// and the hit count. Points no view can see are left unmarked.
    private static func blend(points: [Float], normals: [Float]?, count: Int,
                              views: [ViewData], params: Params) -> ([Float], [Bool], Int) {
        var colors = [Float](repeating: 0, count: count * 3)
        var hit = [Bool](repeating: false, count: count)
        var hits = 0
        for i in 0 ..< count {
            let x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2]
            var wsum: Float = 0
            var acc: (Float, Float, Float) = (0, 0, 0)
            for vd in views {
                let u = y * vd.cosT - x * vd.sinT
                let v = z
                let d = -(x * vd.cosT + y * vd.sinT)
                // occlusion: must be at (or just behind) this view's front surface
                let cx = Int((u + worldHalf) / (2 * worldHalf) * Float(vd.V))
                let cy = Int((worldHalf - v) / (2 * worldHalf) * Float(vd.V))
                guard cx >= 0, cx < vd.V, cy >= 0, cy < vd.V else { continue }
                guard d <= vd.depth[cy * vd.V + cx] + vd.tol else { continue }
                // photo sample (alpha-weighted bilinear)
                let fx = u * vd.s + vd.tu, fy = -v * vd.s + vd.tv
                guard let (r, g, b, a) = sampleRGBA(vd.pix, vd.G, fx, fy), a > 0.5 else { continue }
                var w = vd.weight * a
                if let n = normals {
                    // forward ∝ (cosθ, sinθ, 0); splat/mesh normal signs are ambiguous → |cos|
                    let c = abs(n[i * 3] * vd.cosT + n[i * 3 + 1] * vd.sinT)
                    w *= pow(c, params.cosPower) + 0.02     // floor keeps grazing surfaces texturable
                }
                wsum += w
                acc.0 += w * r; acc.1 += w * g; acc.2 += w * b
            }
            if wsum > 1e-5 {
                colors[i * 3] = acc.0 / wsum
                colors[i * 3 + 1] = acc.1 / wsum
                colors[i * 3 + 2] = acc.2 / wsum
                hit[i] = true; hits += 1
            }
        }
        return (colors, hit, hits)
    }

    /// Alpha-weighted bilinear sample of a straight-RGBA grid (rows top-down).
    private static func sampleRGBA(_ pix: [Float], _ G: Int, _ fx: Float, _ fy: Float)
        -> (Float, Float, Float, Float)? {
        let x0 = Int(floor(fx)), y0 = Int(floor(fy))
        guard x0 >= 0, y0 >= 0, x0 < G - 1, y0 < G - 1 else { return nil }
        let tx = fx - Float(x0), ty = fy - Float(y0)
        var r: Float = 0, g: Float = 0, b: Float = 0, a: Float = 0, asum: Float = 0
        for (dx, dy, w) in [(0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                            (0, 1, (1 - tx) * ty), (1, 1, tx * ty)] {
            let p = ((y0 + dy) * G + x0 + dx) * 4
            let pa = pix[p + 3] * w
            r += pix[p] * pa; g += pix[p + 1] * pa; b += pix[p + 2] * pa
            asum += pa; a += pix[p + 3] * w
        }
        guard asum > 1e-6 else { return nil }
        return (r / asum, g / asum, b / asum, a)
    }

    /// Area-weighted vertex normals (unnormalized sign — callers use |cos|).
    private static func vertexNormals(_ mesh: MeshExtract.Mesh) -> [Float] {
        var n = [Float](repeating: 0, count: mesh.positions.count)
        let p = mesh.positions
        for t in 0 ..< mesh.triangleCount {
            let a = Int(mesh.triangles[t * 3]), b = Int(mesh.triangles[t * 3 + 1]), c = Int(mesh.triangles[t * 3 + 2])
            let e1 = (p[b * 3] - p[a * 3], p[b * 3 + 1] - p[a * 3 + 1], p[b * 3 + 2] - p[a * 3 + 2])
            let e2 = (p[c * 3] - p[a * 3], p[c * 3 + 1] - p[a * 3 + 1], p[c * 3 + 2] - p[a * 3 + 2])
            let cr = (e1.1 * e2.2 - e1.2 * e2.1, e1.2 * e2.0 - e1.0 * e2.2, e1.0 * e2.1 - e1.1 * e2.0)
            for v in [a, b, c] {
                n[v * 3] += cr.0; n[v * 3 + 1] += cr.1; n[v * 3 + 2] += cr.2
            }
        }
        for v in 0 ..< mesh.vertexCount {
            let len = (n[v * 3] * n[v * 3] + n[v * 3 + 1] * n[v * 3 + 1] + n[v * 3 + 2] * n[v * 3 + 2]).squareRoot()
            if len > 1e-12 { for c in 0 ..< 3 { n[v * 3 + c] /= len } }
        }
        return n
    }

    /// Rotate unit axis e_axis by quaternion (w,x,y,z) — column `axis` of the rotation matrix.
    private static func rotatedAxis(qw: Float, qx: Float, qy: Float, qz: Float, axis: Int)
        -> (Float, Float, Float) {
        switch axis {
        case 0: return (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy))
        case 1: return (2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx))
        default: return (2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy))
        }
    }

    /// Parse a binary little-endian PLY header: vertex count, float-property names in
    /// order, and the byte offset of the body.
    private static func plyHeader(_ data: Data) throws -> (count: Int, props: [String], bodyOffset: Int) {
        guard let hdrEnd = data.range(of: Data("end_header\n".utf8)) else { throw Err.parse("no header") }
        let header = String(decoding: data[data.startIndex ..< hdrEnd.lowerBound], as: UTF8.self)
        guard header.contains("binary_little_endian") else { throw Err.parse("not binary_little_endian") }
        var count = 0, props = [String]()
        for line in header.split(separator: "\n") {
            let p = line.split(separator: " ")
            if p.count >= 3, p[0] == "element", p[1] == "vertex" { count = Int(p[2]) ?? 0 }
            else if p.count >= 3, p[0] == "property", p[1] == "float" { props.append(String(p[2])) }
        }
        guard count > 0 else { throw Err.parse("no vertices") }
        return (count, props, hdrEnd.upperBound - data.startIndex)
    }
}
