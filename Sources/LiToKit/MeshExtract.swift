import Foundation

/// Splat → mesh: rasterize the gaussians into a density grid, run marching cubes,
/// Taubin-smooth, and color vertices from the nearest splat. This turns the fuzzy
/// radiance field into an actual surface (OBJ/PLY mesh) — geometry-first output.
///
/// Quality ceiling note: LiTo's geometry lives on a 64³ occupancy scaffold, so the
/// extracted surface is a faithful *de-fuzzed* version of the splat, not new detail.
public enum MeshExtract {
    public struct Mesh {
        public var positions: [Float]      // (V·3) LiTo world coords (z-up)
        public var colors: [UInt8]         // (V·3) sRGB
        public var triangles: [Int32]      // (T·3) indices
        public var vertexCount: Int { positions.count / 3 }
        public var triangleCount: Int { triangles.count / 3 }
    }

    public enum Err: Error, CustomStringConvertible {
        case parse(String), empty
        public var description: String {
            switch self {
            case .parse(let m): return "gs.ply parse error: \(m)"
            case .empty: return "no surface found at this iso level"
            }
        }
    }

    // MARK: - Splat loading (our 3DGS export format)

    struct Splats {
        var pos: [Float]      // (N·3)
        var rgb: [Float]      // (N·3) linear sRGB [0,1] from SH0
        var opacity: [Float]  // (N)
        var sigma: [Float]    // (N) mean world-space std dev
        var count: Int
    }

    static let shC0: Float = 0.28209479177387814

    static func loadSplats(gsPLY url: URL) throws -> Splats {
        let data = try Data(contentsOf: url)
        guard let hdrEnd = data.range(of: Data("end_header\n".utf8)) else { throw Err.parse("no header") }
        let header = String(decoding: data[data.startIndex ..< hdrEnd.lowerBound], as: UTF8.self)
        guard header.contains("binary_little_endian") else { throw Err.parse("not binary_little_endian") }

        var count = 0, props = [String]()
        for line in header.split(separator: "\n") {
            let p = line.split(separator: " ")
            if p.count >= 3, p[0] == "element" { if p[1] == "vertex" { count = Int(p[2]) ?? 0 } }
            else if p.count >= 3, p[0] == "property", p[1] == "float" { props.append(String(p[2])) }
        }
        guard count > 0 else { throw Err.parse("no vertices") }
        func idx(_ name: String) throws -> Int {
            guard let i = props.firstIndex(of: name) else { throw Err.parse("missing \(name)") }
            return i
        }
        let ox = try idx("x"), oy = try idx("y"), oz = try idx("z")
        let od = try idx("f_dc_0"), oo = try idx("opacity"), os = try idx("scale_0")
        let stride = props.count * 4

        var s = Splats(pos: .init(repeating: 0, count: count * 3),
                       rgb: .init(repeating: 0, count: count * 3),
                       opacity: .init(repeating: 0, count: count),
                       sigma: .init(repeating: 0, count: count), count: count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: hdrEnd.upperBound - data.startIndex)
            for i in 0 ..< count {
                let r = base.advanced(by: i * stride)
                func f(_ p: Int) -> Float { r.advanced(by: p * 4).loadUnaligned(as: Float.self) }
                s.pos[i * 3] = f(ox); s.pos[i * 3 + 1] = f(oy); s.pos[i * 3 + 2] = f(oz)
                for c in 0 ..< 3 { s.rgb[i * 3 + c] = max(0, min(1, 0.5 + shC0 * f(od + c))) }
                s.opacity[i] = 1 / (1 + exp(-f(oo)))                       // logit → opacity
                let sc = (exp(f(os)) + exp(f(os + 1)) + exp(f(os + 2))) / 3   // log → mean σ
                s.sigma[i] = sc
            }
        }
        return s
    }

    // MARK: - Pipeline

    /// Extract a mesh from a 3DGS gaussian PLY. `resolution` = density-grid cells per axis
    /// over [-1.05, 1.05]; `iso` = density threshold; higher = tighter/thinner surface.
    public static func extract(gsPLY: URL, resolution: Int = 256, iso: Float = 0.5,
                               smoothIterations: Int = 15,
                               log: ((String) -> Void)? = nil) throws -> Mesh {
        let splats = try loadSplats(gsPLY: gsPLY)
        log?("loaded \(splats.count) splats")
        return try extract(splats: splats, resolution: resolution, iso: iso,
                           smoothIterations: smoothIterations, log: log)
    }

    static func extract(splats: Splats, resolution R: Int, iso: Float,
                        smoothIterations: Int, log: ((String) -> Void)?) throws -> Mesh {
        let lo: Float = -1.05, hi: Float = 1.05
        let cw = (hi - lo) / Float(R)

        // 1. density grid — accumulate opacity-weighted isotropic gaussians
        var grid = [Float](repeating: 0, count: R * R * R)
        grid.withUnsafeMutableBufferPointer { g in
            for i in 0 ..< splats.count {
                let op = splats.opacity[i]
                if op < 0.05 { continue }                       // dead splats add only noise
                let px = splats.pos[i * 3], py = splats.pos[i * 3 + 1], pz = splats.pos[i * 3 + 2]
                // clamp σ: at least ~a cell so thin splats still register, at most a few cells
                let sigma = min(max(splats.sigma[i], cw * 0.75), cw * 3)
                let r = 2.5 * sigma
                let inv2s2 = 1 / (2 * sigma * sigma)
                let i0 = max(0, Int((px - r - lo) / cw)), i1 = min(R - 1, Int((px + r - lo) / cw))
                let j0 = max(0, Int((py - r - lo) / cw)), j1 = min(R - 1, Int((py + r - lo) / cw))
                let k0 = max(0, Int((pz - r - lo) / cw)), k1 = min(R - 1, Int((pz + r - lo) / cw))
                guard i0 <= i1, j0 <= j1, k0 <= k1 else { continue }
                for k in k0 ... k1 {
                    let dz = lo + (Float(k) + 0.5) * cw - pz
                    for j in j0 ... j1 {
                        let dy = lo + (Float(j) + 0.5) * cw - py
                        let row = (k * R + j) * R
                        for ii in i0 ... i1 {
                            let dx = lo + (Float(ii) + 0.5) * cw - px
                            let d2 = dx * dx + dy * dy + dz * dz
                            g[row + ii] += op * exp(-d2 * inv2s2)
                        }
                    }
                }
            }
        }
        log?("density grid \(R)³ done")

        // 2. marching cubes (Bourke tables, corners: bottom ring 0-3 at z, top ring 4-7 at z+1)
        var verts = [Float]()
        var tris = [Int32]()
        var edgeVertex = [Int64: Int32]()                       // global edge key → vertex index
        let cornerOff: [(Int, Int, Int)] = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
        // edge → (corner offset of canonical low endpoint, axis 0=x 1=y 2=z)
        let edgeDef: [((Int, Int, Int), Int)] = [
            ((0,0,0),0), ((1,0,0),1), ((0,1,0),0), ((0,0,0),1),
            ((0,0,1),0), ((1,0,1),1), ((0,1,1),0), ((0,0,1),1),
            ((0,0,0),2), ((1,0,0),2), ((1,1,0),2), ((0,1,0),2),
        ]
        func gval(_ i: Int, _ j: Int, _ k: Int) -> Float { grid[(k * R + j) * R + i] }
        func cellCenter(_ i: Int, _ j: Int, _ k: Int) -> (Float, Float, Float) {
            (lo + (Float(i) + 0.5) * cw, lo + (Float(j) + 0.5) * cw, lo + (Float(k) + 0.5) * cw)
        }

        grid.withUnsafeBufferPointer { g in
            func val(_ i: Int, _ j: Int, _ k: Int) -> Float { g[(k * R + j) * R + i] }
            for k in 0 ..< R - 1 { for j in 0 ..< R - 1 { for i in 0 ..< R - 1 {
                var cubeIndex = 0
                for (c, off) in cornerOff.enumerated() where val(i + off.0, j + off.1, k + off.2) < iso {
                    cubeIndex |= 1 << c
                }
                let edges = MCTables.edgeTable[cubeIndex]
                if edges == 0 { continue }

                var local = [Int32](repeating: -1, count: 12)
                for e in 0 ..< 12 where edges & (1 << e) != 0 {
                    let (off, axis) = edgeDef[e]
                    let ei = i + off.0, ej = j + off.1, ek = k + off.2
                    let key = Int64(((ek * R + ej) * R + ei)) << 2 | Int64(axis)
                    if let existing = edgeVertex[key] { local[e] = existing; continue }
                    // interpolate along +axis from the canonical endpoint
                    let v1 = val(ei, ej, ek)
                    let (ni, nj, nk) = (ei + (axis == 0 ? 1 : 0), ej + (axis == 1 ? 1 : 0), ek + (axis == 2 ? 1 : 0))
                    let v2 = val(ni, nj, nk)
                    let t = abs(v2 - v1) < 1e-12 ? 0.5 : (iso - v1) / (v2 - v1)
                    let (cx, cy, cz) = cellCenter(ei, ej, ek)
                    var p = (cx, cy, cz)
                    if axis == 0 { p.0 += t * cw } else if axis == 1 { p.1 += t * cw } else { p.2 += t * cw }
                    let idx = Int32(verts.count / 3)
                    verts.append(p.0); verts.append(p.1); verts.append(p.2)
                    edgeVertex[key] = idx
                    local[e] = idx
                }

                let row = MCTables.triTable[cubeIndex]
                var t = 0
                while row[t] != -1 {
                    tris.append(local[Int(row[t])])
                    tris.append(local[Int(row[t + 1])])
                    tris.append(local[Int(row[t + 2])])
                    t += 3
                }
            }}}
        }
        guard !tris.isEmpty else { throw Err.empty }
        log?("marching cubes: \(verts.count / 3) verts, \(tris.count / 3) tris")

        // 3. Taubin smoothing (λ/µ keeps volume, unlike pure Laplacian)
        if smoothIterations > 0 {
            var neighbors = [[Int32]](repeating: [], count: verts.count / 3)
            var seen = Set<Int64>()
            for t in stride(from: 0, to: tris.count, by: 3) {
                let a = tris[t], b = tris[t + 1], c = tris[t + 2]
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    let key = Int64(min(u, v)) << 32 | Int64(max(u, v))
                    if seen.insert(key).inserted {
                        neighbors[Int(u)].append(v); neighbors[Int(v)].append(u)
                    }
                }
            }
            var cur = verts
            var next = verts
            for it in 0 ..< smoothIterations * 2 {
                let lambda: Float = it % 2 == 0 ? 0.5 : -0.53
                for v in 0 ..< cur.count / 3 {
                    let ns = neighbors[v]
                    guard !ns.isEmpty else { continue }
                    var ax: Float = 0, ay: Float = 0, az: Float = 0
                    for n in ns {
                        ax += cur[Int(n) * 3]; ay += cur[Int(n) * 3 + 1]; az += cur[Int(n) * 3 + 2]
                    }
                    let inv = 1 / Float(ns.count)
                    next[v * 3] = cur[v * 3] + lambda * (ax * inv - cur[v * 3])
                    next[v * 3 + 1] = cur[v * 3 + 1] + lambda * (ay * inv - cur[v * 3 + 1])
                    next[v * 3 + 2] = cur[v * 3 + 2] + lambda * (az * inv - cur[v * 3 + 2])
                }
                swap(&cur, &next)
            }
            verts = cur
            log?("taubin smoothing ×\(smoothIterations) done")
        }

        // 4. vertex colors from the nearest opaque splat (hash grid lookup)
        var colors = [UInt8](repeating: 128, count: verts.count)
        do {
            let cell: Float = 0.025
            var hash = [Int64: [Int32]]()
            func key(_ x: Float, _ y: Float, _ z: Float) -> Int64 {
                let i = Int64((x + 2) / cell), j = Int64((y + 2) / cell), k = Int64((z + 2) / cell)
                return (k * 4096 + j) * 4096 + i
            }
            for i in 0 ..< splats.count where splats.opacity[i] > 0.15 {
                hash[key(splats.pos[i * 3], splats.pos[i * 3 + 1], splats.pos[i * 3 + 2]), default: []].append(Int32(i))
            }
            for v in 0 ..< verts.count / 3 {
                let x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2]
                let ci = Int64((x + 2) / cell), cj = Int64((y + 2) / cell), ck = Int64((z + 2) / cell)
                var best: Int32 = -1; var bestD: Float = .greatestFiniteMagnitude
                for dk in -1 ... 1 { for dj in -1 ... 1 { for di in -1 ... 1 {
                    let k = ((ck + Int64(dk)) * 4096 + (cj + Int64(dj))) * 4096 + (ci + Int64(di))
                    guard let bucket = hash[k] else { continue }
                    for s in bucket {
                        let dx = splats.pos[Int(s) * 3] - x, dy = splats.pos[Int(s) * 3 + 1] - y, dz = splats.pos[Int(s) * 3 + 2] - z
                        let d2 = dx * dx + dy * dy + dz * dz
                        if d2 < bestD { bestD = d2; best = s }
                    }
                }}}
                if best >= 0 {
                    for c in 0 ..< 3 { colors[v * 3 + c] = UInt8(splats.rgb[Int(best) * 3 + c] * 255) }
                }
            }
            log?("vertex colors done")
        }

        return Mesh(positions: verts, colors: colors, triangles: tris)
    }

    // MARK: - Writers

    /// LiTo world is +Z up, but OBJ/PLY mesh consumers (Quick Look, Blender importers,
    /// most web viewers) assume +Y up and would show the model lying on its side.
    /// Writers therefore rotate −90° about X at export: (x, y, z) → (x, z, −y).
    @inline(__always)
    static func yUpVertex(_ mesh: Mesh, _ v: Int) -> (Float, Float, Float) {
        (mesh.positions[v * 3], mesh.positions[v * 3 + 2], -mesh.positions[v * 3 + 1])
    }

    /// Binary little-endian PLY mesh. Vertex layout is x,y,z float + r,g,b uchar (15 bytes,
    /// same as the point-cloud export) so existing tooling can still point-render it.
    /// Exported Y-up (see `yUpVertex`); pass `yUp: false` for raw LiTo z-up coords.
    public static func writePLY(_ mesh: Mesh, to url: URL, yUp: Bool = true) throws {
        let V = mesh.vertexCount, T = mesh.triangleCount
        let header = """
        ply
        format binary_little_endian 1.0
        comment LiToStudio mesh (\(yUp ? "+Y up" : "LiTo native +Z up"))
        element vertex \(V)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        element face \(T)
        property list uchar int vertex_indices
        end_header

        """
        var out = Data(header.utf8)
        out.reserveCapacity(out.count + V * 15 + T * 13)
        for v in 0 ..< V {
            let p = yUp ? yUpVertex(mesh, v)
                        : (mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])
            for var f in [p.0, p.1, p.2] {
                withUnsafeBytes(of: &f) { out.append(contentsOf: $0) }
            }
            out.append(mesh.colors[v * 3]); out.append(mesh.colors[v * 3 + 1]); out.append(mesh.colors[v * 3 + 2])
        }
        for t in 0 ..< T {
            out.append(3)
            for c in 0 ..< 3 {
                var i = mesh.triangles[t * 3 + c]
                withUnsafeBytes(of: &i) { out.append(contentsOf: $0) }
            }
        }
        try out.write(to: url)
    }

    /// OBJ with per-vertex colors ("v x y z r g b" extension — Blender et al. read it).
    /// Exported Y-up (see `yUpVertex`); pass `yUp: false` for raw LiTo z-up coords.
    public static func writeOBJ(_ mesh: Mesh, to url: URL, yUp: Bool = true) throws {
        var s = "# LiToStudio splat→mesh export (\(yUp ? "+Y up" : "LiTo world coords, +Z up"))\n"
        s.reserveCapacity(mesh.vertexCount * 48 + mesh.triangleCount * 24)
        for v in 0 ..< mesh.vertexCount {
            let p = yUp ? yUpVertex(mesh, v)
                        : (mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])
            let c = (Float(mesh.colors[v * 3]) / 255, Float(mesh.colors[v * 3 + 1]) / 255, Float(mesh.colors[v * 3 + 2]) / 255)
            s += "v \(p.0) \(p.1) \(p.2) \(c.0) \(c.1) \(c.2)\n"
        }
        for t in 0 ..< mesh.triangleCount {
            s += "f \(mesh.triangles[t * 3] + 1) \(mesh.triangles[t * 3 + 1] + 1) \(mesh.triangles[t * 3 + 2] + 1)\n"
        }
        try s.write(to: url, atomically: true, encoding: .utf8)
    }
}
