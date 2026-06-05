import Foundation
import Dispatch

/// Photo-metric mesh refinement: re-sculpt the front (camera-facing) surface of a
/// MeshExtract mesh so its screen-space depth gradients match the Sapiens-predicted
/// normal map, and snap the outline to the RMBG silhouette. This puts *measured*
/// detail (cloth folds, facial planes) onto the mesh instead of generative guesses.
///
/// View model (pinned empirically via `LiToSmoke score`): the conditioning camera is
/// orthographic along the world x axis with image-right = +y and image-up = +z, i.e.
/// pixel U = (y+1.04)/2.08·G, V = (1.04−z)/2.08·G. Two things are *not* assumed:
///   • which x direction faces the camera — decided per-mesh by comparing the splat
///     colors of the min-x vs max-x depth layer against the photo, and
///   • Sapiens' camera-space channel convention — calibrated per-image by correlating
///     the predicted normals against the mesh's own rasterized normals (all 6
///     permutations × signs), so a checkpoint with a different axis order still works.
///
/// The front depth field f(u,v) is then solved as a screened Poisson problem
///   min ∫|∇f − g|² + λ(f − f0)²,  g from the predicted normals,
/// coarse-to-fine with red-black SOR, and the *depth delta* (f − f0) is applied to
/// front-facing vertices along the view axis (deltas, not absolute snapping, so mesh
/// topology and raster quantization never fight each other).
public enum NormalRefine {
    public struct Params {
        /// Working grid resolution (square, matches the cond crop fed to Sapiens).
        public var gridSize: Int = 1024
        /// Screened-Poisson anchor weight at full res (world-scale; coarser levels scale by 4×/level).
        public var lambda: Float = 5e-4
        /// Blend factor for the solved depth delta (1 = full).
        public var strength: Float = 1.0
        /// Hard cap on per-vertex depth change, world units (grid spans 2.08).
        public var maxDisplacement: Float = 0.05
        /// Snap vertices projecting outside the alpha silhouette back inside. Only near
        /// misses move (splat fuzz is ~1 voxel = 0.033 world); a vertex farther out than
        /// maxSilhouettePull is global misalignment — dragging it would mangle the mesh.
        public var silhouette: Bool = true
        public var maxSilhouettePull: Float = 0.04
        /// A vertex must sit within this depth of the rasterized front layer to be sculpted.
        public var depthTolerance: Float = 0.02
        public init() {}
    }

    public struct Stats {
        public var aborted: String?            // non-nil = refinement skipped, reason
        public var frontIsMinX = true
        public var colorErrFront: Float = 0, colorErrBack: Float = 0
        public var axisMap = ""                // e.g. "y=+c0 z=+c1 x=-c2"
        public var corrSum: Float = 0          // Σ|corr| over the 3 axes (3 = perfect)
        public var gradientPixels = 0
        public var movedVertices = 0, silhouettePulled = 0
        public var meanDisp: Float = 0, maxDisp: Float = 0
        public var summary: String {
            if let a = aborted { return "refine ABORTED: \(a)" }
            return String(format: "front=%@x colorRMSE %.3f/%.3f | %@ Σ|corr|=%.2f | %d grad px | moved %d verts (mean %.4f max %.4f) | sil-pulled %d",
                          frontIsMinX ? "min-" : "max-", colorErrFront, colorErrBack,
                          axisMap, corrSum, gradientPixels, movedVertices, meanDisp, maxDisp, silhouettePulled)
        }
    }

    static let worldHalf: Float = 1.04         // grid spans [-1.04, 1.04], same as the IoU scorer

    /// Refine `mesh` in place. `condRGBA` is the G²·4 cond crop (straight RGB+alpha,
    /// rows top-down) from `Preprocess.condRGBAPixels`; `pred` is the Sapiens normal
    /// map on the same grid, or nil to self-test against the mesh's own normals
    /// (validates the whole solve path — displacement should come back ≈ 0).
    @discardableResult
    public static func refine(mesh: inout MeshExtract.Mesh, condRGBA: [Float], gridSize G: Int,
                              pred: SapiensNormal.NormalMap?, params: Params = Params(),
                              log: ((String) -> Void)? = nil) -> Stats {
        var stats = Stats()
        precondition(condRGBA.count == G * G * 4)
        if let p = pred, p.size != G { stats.aborted = "normal map grid \(pred!.size) != \(G)"; return stats }
        let V = mesh.vertexCount
        guard V > 0, mesh.triangleCount > 0 else { stats.aborted = "empty mesh"; return stats }
        let h = 2 * worldHalf / Float(G)

        // ---- 1. Rasterize both depth layers (min-x and max-x) with face normals + colors
        var raster = Raster(G: G)
        raster.draw(mesh: mesh)
        log?("raster: \(raster.coveredCount) px covered")
        guard raster.coveredCount > G else { stats.aborted = "mesh projects to almost nothing"; return stats }

        // ---- 2. Which x direction faces the camera? The front layer's splat colors match the photo.
        var errMin: Double = 0, errMax: Double = 0, nc = 0
        for p in 0 ..< G * G where raster.covered[p] && condRGBA[p * 4 + 3] > 0.9 {
            for c in 0 ..< 3 {
                let pc = condRGBA[p * 4 + c]
                let dn = Double(raster.colMin[p * 3 + c] - pc), dx = Double(raster.colMax[p * 3 + c] - pc)
                errMin += dn * dn; errMax += dx * dx
            }
            nc += 1
        }
        guard nc > 64 else { stats.aborted = "no overlap between mesh and silhouette"; return stats }
        stats.colorErrFront = Float((errMin / Double(nc * 3)).squareRoot())
        stats.colorErrBack = Float((errMax / Double(nc * 3)).squareRoot())
        let frontIsMin = errMin <= errMax
        stats.frontIsMinX = frontIsMin
        if !frontIsMin { swap(&stats.colorErrFront, &stats.colorErrBack) }
        let s: Float = frontIsMin ? 1 : -1      // depth d = s·x, front layer = min d

        // Front-layer fields. Mesh normals oriented toward the camera (−s x̂); also derive
        // the global winding orientation β so vertex normals can be made outward later.
        var f0 = [Float](repeating: 0, count: G * G)
        var meshN = [Float](repeating: 0, count: G * G * 3)
        var windingVotes = 0
        for p in 0 ..< G * G where raster.covered[p] {
            f0[p] = frontIsMin ? raster.dMin[p] : -raster.dMax[p]
            var nx = frontIsMin ? raster.nMin[p * 3] : raster.nMax[p * 3]
            var ny = frontIsMin ? raster.nMin[p * 3 + 1] : raster.nMax[p * 3 + 1]
            var nz = frontIsMin ? raster.nMin[p * 3 + 2] : raster.nMax[p * 3 + 2]
            windingVotes += (-s * nx) > 0 ? 1 : -1
            if -s * nx < 0 { nx = -nx; ny = -ny; nz = -nz }     // face the camera
            meshN[p * 3] = nx; meshN[p * 3 + 1] = ny; meshN[p * 3 + 2] = nz
        }
        let beta: Float = windingVotes >= 0 ? 1 : -1

        // ---- 3. Predicted normals → world space (calibrate permutation + signs against meshN)
        let predN: [Float], predValid: [UInt8]
        if let p = pred { predN = p.normals; predValid = p.valid }
        else {   // self-test: feed the mesh's own normals back as the "prediction"
            var pn = [Float](repeating: 0, count: G * G * 3)
            var pv = [UInt8](repeating: 0, count: G * G)
            for p in 0 ..< G * G where raster.covered[p] {
                pn[p * 3] = meshN[p * 3 + 1]; pn[p * 3 + 1] = meshN[p * 3 + 2]; pn[p * 3 + 2] = meshN[p * 3]
                pv[p] = 1
            }
            predN = pn; predValid = pv
        }

        guard let cal = calibrate(predN: predN, predValid: predValid, meshN: meshN,
                                  covered: raster.covered, G: G) else {
            stats.aborted = "too few pixels to calibrate axis mapping"; return stats
        }
        stats.axisMap = cal.describe
        stats.corrSum = cal.corrSum
        log?("axis calibration: \(cal.describe)  Σ|corr|=\(String(format: "%.2f", cal.corrSum))")
        if cal.corrSum < 0.5 {
            stats.aborted = String(format: "axis correlation too weak (Σ|corr|=%.2f) — normal map doesn't match this view", cal.corrSum)
            return stats
        }
        if cal.corrSum < 1.2 { log?("⚠️ weak normal/mesh agreement — refinement will be conservative") }

        // ---- 4. Target depth gradients from the calibrated world normals
        var gU = [Float](repeating: 0, count: G * G)     // target ∂f/∂u (depth per pixel)
        var gV = [Float](repeating: 0, count: G * G)
        var gValid = [UInt8](repeating: 0, count: G * G)
        for p in 0 ..< G * G where raster.covered[p] && predValid[p] == 1 {
            let n = cal.toWorld(predN[p * 3], predN[p * 3 + 1], predN[p * 3 + 2])   // (x,y,z), unit
            guard -s * n.0 > 0.15 else { continue }       // grazing/backfacing → unreliable slope
            var dy = -s * n.1 / n.0, dz = -s * n.2 / n.0
            dy = max(-5, min(5, dy)); dz = max(-5, min(5, dz))
            gU[p] = h * dy
            gV[p] = -h * dz                               // v grows downward, z grows upward
            gValid[p] = 1
            stats.gradientPixels += 1
        }
        guard stats.gradientPixels > 256 else {
            stats.aborted = "only \(stats.gradientPixels) usable gradient pixels"; return stats
        }

        // ---- 5. Screened Poisson solve, coarse-to-fine
        let f = solve(f0: f0, covered: raster.covered, gU: gU, gV: gV, gValid: gValid,
                      G: G, lambda: params.lambda, log: log)

        // ---- 6. Vertex outward normals (β-corrected winding) for eligibility + rim fade
        var vertexN = [Float](repeating: 0, count: V * 3)
        mesh.positions.withUnsafeBufferPointer { pos in
            for t in 0 ..< mesh.triangleCount {
                let a = Int(mesh.triangles[t * 3]), b = Int(mesh.triangles[t * 3 + 1]), c = Int(mesh.triangles[t * 3 + 2])
                let ax = pos[a*3], ay = pos[a*3+1], az = pos[a*3+2]
                let e1 = (pos[b*3]-ax, pos[b*3+1]-ay, pos[b*3+2]-az)
                let e2 = (pos[c*3]-ax, pos[c*3+1]-ay, pos[c*3+2]-az)
                let n = (e1.1*e2.2 - e1.2*e2.1, e1.2*e2.0 - e1.0*e2.2, e1.0*e2.1 - e1.1*e2.0)
                for v in [a, b, c] {
                    vertexN[v*3] += beta * n.0; vertexN[v*3+1] += beta * n.1; vertexN[v*3+2] += beta * n.2
                }
            }
        }

        // ---- 7. Silhouette snap (lateral, before depth sampling so moved verts land on the grid)
        if params.silhouette {
            let (nearest, _) = insideEDT(condRGBA: condRGBA, G: G)
            for v in 0 ..< V {
                let y = mesh.positions[v*3+1], z = mesh.positions[v*3+2]
                let (U, Vv) = project(y: y, z: z, G: G)
                let pu = Int(U), pv = Int(Vv)
                let p = min(G - 1, max(0, pv)) * G + min(G - 1, max(0, pu))
                let outside = pu < 0 || pu >= G || pv < 0 || pv >= G || condRGBA[p * 4 + 3] <= 0.5
                guard outside, nearest[p] >= 0 else { continue }
                let ti = Int(nearest[p])
                let ty = (Float(ti % G) + 0.5) * h - worldHalf
                let tz = worldHalf - (Float(ti / G) + 0.5) * h
                let dy = ty - y, dz = tz - z
                let L = (dy * dy + dz * dz).squareRoot()
                guard L > 1e-6, L <= params.maxSilhouettePull else { continue }
                mesh.positions[v*3+1] += dy; mesh.positions[v*3+2] += dz
                stats.silhouettePulled += 1
            }
        }

        // ---- 8. Apply the depth delta to front-facing vertices
        var dispSum: Double = 0
        for v in 0 ..< V {
            let x = mesh.positions[v*3], y = mesh.positions[v*3+1], z = mesh.positions[v*3+2]
            var nx = vertexN[v*3], ny = vertexN[v*3+1], nz = vertexN[v*3+2]
            let nl = (nx*nx + ny*ny + nz*nz).squareRoot()
            guard nl > 1e-12 else { continue }
            nx /= nl; ny /= nl; nz /= nl
            let facing = -s * nx
            guard facing > 0.05 else { continue }                     // back side untouched
            let (U, Vv) = project(y: y, z: z, G: G)
            guard let (f0v, fv, wcov) = sampleDepth(f0: f0, f: f, covered: raster.covered, G: G,
                                                    u: U, v: Vv), wcov > 0.3 else { continue }
            guard abs(s * x - f0v) < params.depthTolerance else { continue }   // not on the front layer
            let rim = smoothstep(0.05, 0.30, facing)
            var d = params.strength * rim * wcov * (fv - f0v)
            d = max(-params.maxDisplacement, min(params.maxDisplacement, d))
            guard abs(d) > 1e-7 else { continue }
            mesh.positions[v*3] = x + s * d
            stats.movedVertices += 1
            dispSum += Double(abs(d))
            stats.maxDisp = max(stats.maxDisp, abs(d))
        }
        stats.meanDisp = stats.movedVertices > 0 ? Float(dispSum / Double(stats.movedVertices)) : 0
        log?(stats.summary)
        return stats
    }

    // MARK: - Projection helpers

    @inline(__always)
    static func project(y: Float, z: Float, G: Int) -> (Float, Float) {
        ((y + worldHalf) / (2 * worldHalf) * Float(G), (worldHalf - z) / (2 * worldHalf) * Float(G))
    }

    @inline(__always)
    static func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }

    /// Coverage-weighted bilinear sample of f0 and f at continuous grid coords.
    static func sampleDepth(f0: [Float], f: [Float], covered: [Bool], G: Int,
                            u: Float, v: Float) -> (Float, Float, Float)? {
        let uc = u - 0.5, vc = v - 0.5                     // values live at pixel centers
        let x0 = Int(floor(uc)), y0 = Int(floor(vc))
        let fx = uc - Float(x0), fy = vc - Float(y0)
        var w0: Float = 0, a0: Float = 0, a1: Float = 0
        for (dx, dy, wt) in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)] {
            let xx = x0 + dx, yy = y0 + dy
            guard xx >= 0, xx < G, yy >= 0, yy < G else { continue }
            let p = yy * G + xx
            guard covered[p] else { continue }
            w0 += wt; a0 += wt * f0[p]; a1 += wt * f[p]
        }
        guard w0 > 1e-6 else { return nil }
        return (a0 / w0, a1 / w0, w0)
    }

    // MARK: - Rasterizer (both depth layers, ortho along x)

    struct Raster {
        let G: Int
        var dMin: [Float], dMax: [Float]
        var nMin: [Float], nMax: [Float]       // wound face normals (unit)
        var colMin: [Float], colMax: [Float]   // interpolated vertex colors [0,1]
        var covered: [Bool]
        var coveredCount = 0

        init(G: Int) {
            self.G = G
            dMin = [Float](repeating: .greatestFiniteMagnitude, count: G * G)
            dMax = [Float](repeating: -.greatestFiniteMagnitude, count: G * G)
            nMin = [Float](repeating: 0, count: G * G * 3)
            nMax = [Float](repeating: 0, count: G * G * 3)
            colMin = [Float](repeating: 0, count: G * G * 3)
            colMax = [Float](repeating: 0, count: G * G * 3)
            covered = [Bool](repeating: false, count: G * G)
        }

        mutating func draw(mesh: MeshExtract.Mesh) {
            let T = mesh.triangleCount
            for t in 0 ..< T {
                let ia = Int(mesh.triangles[t*3]), ib = Int(mesh.triangles[t*3+1]), ic = Int(mesh.triangles[t*3+2])
                let pa = (mesh.positions[ia*3], mesh.positions[ia*3+1], mesh.positions[ia*3+2])
                let pb = (mesh.positions[ib*3], mesh.positions[ib*3+1], mesh.positions[ib*3+2])
                let pc = (mesh.positions[ic*3], mesh.positions[ic*3+1], mesh.positions[ic*3+2])
                let (ua, va) = NormalRefine.project(y: pa.1, z: pa.2, G: G)
                let (ub, vb) = NormalRefine.project(y: pb.1, z: pb.2, G: G)
                let (uc, vc) = NormalRefine.project(y: pc.1, z: pc.2, G: G)
                let area = (ub - ua) * (vc - va) - (uc - ua) * (vb - va)
                guard abs(area) > 1e-9 else { continue }   // rim-perpendicular sliver

                // wound face normal in world space
                let e1 = (pb.0-pa.0, pb.1-pa.1, pb.2-pa.2), e2 = (pc.0-pa.0, pc.1-pa.1, pc.2-pa.2)
                var fn = (e1.1*e2.2 - e1.2*e2.1, e1.2*e2.0 - e1.0*e2.2, e1.0*e2.1 - e1.1*e2.0)
                let fl = (fn.0*fn.0 + fn.1*fn.1 + fn.2*fn.2).squareRoot()
                guard fl > 1e-12 else { continue }
                fn = (fn.0/fl, fn.1/fl, fn.2/fl)

                let ca = (Float(mesh.colors[ia*3])/255, Float(mesh.colors[ia*3+1])/255, Float(mesh.colors[ia*3+2])/255)
                let cb = (Float(mesh.colors[ib*3])/255, Float(mesh.colors[ib*3+1])/255, Float(mesh.colors[ib*3+2])/255)
                let cc = (Float(mesh.colors[ic*3])/255, Float(mesh.colors[ic*3+1])/255, Float(mesh.colors[ic*3+2])/255)

                let x0 = max(0, Int(min(ua, ub, uc) - 0.5)), x1 = min(G - 1, Int(max(ua, ub, uc) + 0.5))
                let y0 = max(0, Int(min(va, vb, vc) - 0.5)), y1 = min(G - 1, Int(max(va, vb, vc) + 0.5))
                guard x0 <= x1, y0 <= y1 else { continue }
                let inv = 1 / area
                for py in y0 ... y1 {
                    let sy = Float(py) + 0.5
                    for px in x0 ... x1 {
                        let sx = Float(px) + 0.5
                        let w0 = ((ub - sx) * (vc - sy) - (uc - sx) * (vb - sy)) * inv
                        let w1 = ((uc - sx) * (va - sy) - (ua - sx) * (vc - sy)) * inv
                        let w2 = 1 - w0 - w1
                        guard w0 >= -1e-5, w1 >= -1e-5, w2 >= -1e-5 else { continue }
                        let d = w0 * pa.0 + w1 * pb.0 + w2 * pc.0       // world x at the pixel
                        let p = py * G + px
                        if !covered[p] { covered[p] = true; coveredCount += 1 }
                        if d < dMin[p] {
                            dMin[p] = d
                            nMin[p*3] = fn.0; nMin[p*3+1] = fn.1; nMin[p*3+2] = fn.2
                            colMin[p*3]   = w0*ca.0 + w1*cb.0 + w2*cc.0
                            colMin[p*3+1] = w0*ca.1 + w1*cb.1 + w2*cc.1
                            colMin[p*3+2] = w0*ca.2 + w1*cb.2 + w2*cc.2
                        }
                        if d > dMax[p] {
                            dMax[p] = d
                            nMax[p*3] = fn.0; nMax[p*3+1] = fn.1; nMax[p*3+2] = fn.2
                            colMax[p*3]   = w0*ca.0 + w1*cb.0 + w2*cc.0
                            colMax[p*3+1] = w0*ca.1 + w1*cb.1 + w2*cc.1
                            colMax[p*3+2] = w0*ca.2 + w1*cb.2 + w2*cc.2
                        }
                    }
                }
            }
        }
    }

    // MARK: - Axis calibration

    struct Calibration {
        var perm: (Int, Int, Int)      // pred channel index feeding world (y, z, x)
        var sign: (Float, Float, Float)
        var corrSum: Float
        var describe: String {
            func f(_ s: Float, _ c: Int) -> String { "\(s > 0 ? "+" : "-")c\(c)" }
            return "y=\(f(sign.0, perm.0)) z=\(f(sign.1, perm.1)) x=\(f(sign.2, perm.2))"
        }
        @inline(__always)
        func toWorld(_ c0: Float, _ c1: Float, _ c2: Float) -> (Float, Float, Float) {
            let c = [c0, c1, c2]
            var n = (sign.2 * c[perm.2], sign.0 * c[perm.0], sign.1 * c[perm.1])  // (x, y, z)
            let l = (n.0*n.0 + n.1*n.1 + n.2*n.2).squareRoot()
            if l > 1e-6 { n = (n.0/l, n.1/l, n.2/l) }
            return n
        }
    }

    /// Pearson-correlate each predicted channel against each world component of the
    /// mesh normals; pick the permutation maximizing Σ|corr|, signs from the corr signs.
    static func calibrate(predN: [Float], predValid: [UInt8], meshN: [Float],
                          covered: [Bool], G: Int) -> Calibration? {
        var n = 0
        var sumP = [Double](repeating: 0, count: 3), sumM = [Double](repeating: 0, count: 3)
        var sumPP = [Double](repeating: 0, count: 3), sumMM = [Double](repeating: 0, count: 3)
        var sumPM = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for p in 0 ..< G * G where covered[p] && predValid[p] == 1 {
            // world components reordered as (y, z, x) to match the calibration targets
            let m = [Double(meshN[p*3+1]), Double(meshN[p*3+2]), Double(meshN[p*3])]
            let pr = [Double(predN[p*3]), Double(predN[p*3+1]), Double(predN[p*3+2])]
            for i in 0 ..< 3 {
                sumP[i] += pr[i]; sumPP[i] += pr[i]*pr[i]
                sumM[i] += m[i]; sumMM[i] += m[i]*m[i]
                for j in 0 ..< 3 { sumPM[i][j] += pr[i]*m[j] }
            }
            n += 1
        }
        guard n > 256 else { return nil }
        let N = Double(n)
        var corr = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for i in 0 ..< 3 {
            for j in 0 ..< 3 {
                let cov = sumPM[i][j] - sumP[i]*sumM[j]/N
                let vp = sumPP[i] - sumP[i]*sumP[i]/N, vm = sumMM[j] - sumM[j]*sumM[j]/N
                corr[i][j] = vp > 1e-9 && vm > 1e-9 ? cov / (vp*vm).squareRoot() : 0
            }
        }
        var best: Calibration?
        for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)] {
            let cs = [corr[perm.0][0], corr[perm.1][1], corr[perm.2][2]]
            let total = Float(cs.map { abs($0) }.reduce(0, +))
            if total > (best?.corrSum ?? -1) {
                best = Calibration(perm: perm,
                                   sign: (Float(cs[0] >= 0 ? 1 : -1) as Float,
                                          Float(cs[1] >= 0 ? 1 : -1) as Float,
                                          Float(cs[2] >= 0 ? 1 : -1) as Float),
                                   corrSum: total)
            }
        }
        return best
    }

    // MARK: - Screened Poisson (coarse-to-fine, red-black SOR)

    private struct Level {
        var G: Int
        var covered: [Bool]
        var f0: [Float]
        var geU: [Float], geV: [Float]   // per-edge gradient targets (p → p+1u / p+1v)
    }

    static func solve(f0: [Float], covered: [Bool], gU: [Float], gV: [Float], gValid: [UInt8],
                      G: Int, lambda: Float, log: ((String) -> Void)?) -> [Float] {
        // Edge targets at full res: predicted gradient where both ends valid, else follow f0.
        // Edges spanning a depth discontinuity (self-occlusion: arm over torso) must follow
        // f0 regardless — integrating a smooth predicted gradient across the jump smears it
        // and the error spreads as global drift. Slopes ≤5 are representable (gradients are
        // clamped there), so anything jumping >6 px-units is a discontinuity, not a slope.
        func buildEdges(G: Int, covered: [Bool], f0: [Float], gU: [Float], gV: [Float],
                        gValid: [UInt8]?, h: Float) -> ([Float], [Float]) {
            let maxJump = 6 * h
            var geU = [Float](repeating: .nan, count: G * G)
            var geV = [Float](repeating: .nan, count: G * G)
            for y in 0 ..< G {
                for x in 0 ..< G {
                    let p = y * G + x
                    guard covered[p] else { continue }
                    if x + 1 < G, covered[p + 1] {
                        let jump = f0[p + 1] - f0[p]
                        let ok = abs(jump) <= maxJump
                            && (gValid == nil || (gValid![p] == 1 && gValid![p + 1] == 1))
                        geU[p] = ok ? 0.5 * (gU[p] + gU[p + 1]) : jump
                    }
                    if y + 1 < G, covered[p + G] {
                        let jump = f0[p + G] - f0[p]
                        let ok = abs(jump) <= maxJump
                            && (gValid == nil || (gValid![p] == 1 && gValid![p + G] == 1))
                        geV[p] = ok ? 0.5 * (gV[p] + gV[p + G]) : jump
                    }
                }
            }
            return (geU, geV)
        }

        // Pyramid: downsample coverage (OR), f0 (mean), per-pixel gradients (mean·2).
        var levels = [Level]()
        do {
            let (geU, geV) = buildEdges(G: G, covered: covered, f0: f0, gU: gU, gV: gV,
                                        gValid: gValid, h: 2 * worldHalf / Float(G))
            levels.append(Level(G: G, covered: covered, f0: f0, geU: geU, geV: geV))
        }
        var curCov = covered, curF0 = f0, curGU = gU, curGV = gV, curValid = gValid, curG = G
        while curG > 128 {
            let g2 = curG / 2
            var cov2 = [Bool](repeating: false, count: g2 * g2)
            var f02 = [Float](repeating: 0, count: g2 * g2)
            var gU2 = [Float](repeating: 0, count: g2 * g2), gV2 = [Float](repeating: 0, count: g2 * g2)
            var valid2 = [UInt8](repeating: 0, count: g2 * g2)
            for y in 0 ..< g2 {
                for x in 0 ..< g2 {
                    var nf = 0, ng = 0
                    var sf: Float = 0, su: Float = 0, sv: Float = 0
                    for dy in 0 ..< 2 {
                        for dx in 0 ..< 2 {
                            let p = (y * 2 + dy) * curG + (x * 2 + dx)
                            if curCov[p] { nf += 1; sf += curF0[p] }
                            if curValid[p] == 1 { ng += 1; su += curGU[p]; sv += curGV[p] }
                        }
                    }
                    let q = y * g2 + x
                    if nf > 0 { cov2[q] = true; f02[q] = sf / Float(nf) }
                    if ng > 0 { valid2[q] = 1; gU2[q] = su / Float(ng) * 2; gV2[q] = sv / Float(ng) * 2 }
                }
            }
            let (geU2, geV2) = buildEdges(G: g2, covered: cov2, f0: f02, gU: gU2, gV: gV2,
                                          gValid: valid2, h: 2 * worldHalf / Float(g2))
            levels.append(Level(G: g2, covered: cov2, f0: f02, geU: geU2, geV: geV2))
            curCov = cov2; curF0 = f02; curGU = gU2; curGV = gV2; curValid = valid2; curG = g2
        }

        // Solve coarsest → finest.
        var f = levels.last!.f0
        for li in stride(from: levels.count - 1, through: 0, by: -1) {
            let L = levels[li]
            if li < levels.count - 1 {      // prolongate previous (coarser) solution
                let C = levels[li + 1]
                var fNew = L.f0
                for y in 0 ..< L.G {
                    for x in 0 ..< L.G {
                        let p = y * L.G + x
                        guard L.covered[p] else { continue }
                        let cu = min(Float(C.G) - 1.001, max(0, (Float(x) + 0.5) / 2 - 0.5))
                        let cv = min(Float(C.G) - 1.001, max(0, (Float(y) + 0.5) / 2 - 0.5))
                        let x0 = Int(cu), y0 = Int(cv)
                        let fx = cu - Float(x0), fy = cv - Float(y0)
                        var wsum: Float = 0, vsum: Float = 0
                        for (dx, dy, wt) in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                                             (0, 1, (1-fx)*fy), (1, 1, fx*fy)] {
                            let q = (y0 + dy) * C.G + (x0 + dx)
                            guard q < C.G * C.G, C.covered[q] else { continue }
                            wsum += wt; vsum += wt * f[q]
                        }
                        if wsum > 1e-6 { fNew[p] = vsum / wsum }
                    }
                }
                f = fNew
            }
            let lam = lambda * Float(G * G) / Float(L.G * L.G)
            let iters = L.G <= 128 ? 400 : (L.G <= 256 ? 200 : (L.G <= 512 ? 120 : 80))
            sor(f: &f, level: L, lambda: lam, iterations: iters)
            log?("poisson level \(L.G)² ×\(iters)")
        }
        return f
    }

    private static func sor(f: inout [Float], level L: Level, lambda: Float, iterations: Int) {
        let G = L.G
        let omega: Float = 1.6
        L.covered.withUnsafeBufferPointer { cov in
            L.f0.withUnsafeBufferPointer { f0 in
                L.geU.withUnsafeBufferPointer { geU in
                    L.geV.withUnsafeBufferPointer { geV in
                        f.withUnsafeMutableBufferPointer { fb in
                            struct P: @unchecked Sendable {
                                let f: UnsafeMutableBufferPointer<Float>
                                let cov: UnsafeBufferPointer<Bool>
                                let f0: UnsafeBufferPointer<Float>
                                let geU: UnsafeBufferPointer<Float>
                                let geV: UnsafeBufferPointer<Float>
                            }
                            let ptrs = P(f: fb, cov: cov, f0: f0, geU: geU, geV: geV)
                            let chunks = max(1, min(ProcessInfo.processInfo.activeProcessorCount, G / 32))
                            let rowsPer = (G + chunks - 1) / chunks
                            for _ in 0 ..< iterations {
                                for parity in 0 ..< 2 {
                                    DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                                        let yStart = chunk * rowsPer, yEnd = min(G, yStart + rowsPer)
                                        for y in yStart ..< yEnd {
                                            let rowBase = y * G
                                            var x = (y + parity) & 1
                                            while x < G {
                                                let p = rowBase + x
                                                if ptrs.cov[p] {
                                                    var num = lambda * ptrs.f0[p]
                                                    var den = lambda
                                                    if x + 1 < G, !ptrs.geU[p].isNaN {
                                                        num += ptrs.f[p + 1] - ptrs.geU[p]; den += 1
                                                    }
                                                    if x > 0, !ptrs.geU[p - 1].isNaN {
                                                        num += ptrs.f[p - 1] + ptrs.geU[p - 1]; den += 1
                                                    }
                                                    if y + 1 < G, !ptrs.geV[p].isNaN {
                                                        num += ptrs.f[p + G] - ptrs.geV[p]; den += 1
                                                    }
                                                    if y > 0, !ptrs.geV[p - G].isNaN {
                                                        num += ptrs.f[p - G] + ptrs.geV[p - G]; den += 1
                                                    }
                                                    let fNew = num / den
                                                    ptrs.f[p] += omega * (fNew - ptrs.f[p])
                                                }
                                                x += 2
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Silhouette distance transform

    /// Two-pass chamfer EDT over the alpha mask: for every outside pixel, the index of
    /// the (approximately) nearest inside pixel. Inside pixels map to themselves.
    static func insideEDT(condRGBA: [Float], G: Int) -> ([Int32], [Float]) {
        var dist = [Float](repeating: .greatestFiniteMagnitude, count: G * G)
        var nearest = [Int32](repeating: -1, count: G * G)
        for p in 0 ..< G * G where condRGBA[p * 4 + 3] > 0.5 {
            dist[p] = 0; nearest[p] = Int32(p)
        }
        let w1: Float = 1, w2: Float = 1.4142135
        func relax(_ p: Int, _ q: Int, _ w: Float) {
            if dist[q] + w < dist[p] { dist[p] = dist[q] + w; nearest[p] = nearest[q] }
        }
        for y in 0 ..< G {
            for x in 0 ..< G {
                let p = y * G + x
                if x > 0 { relax(p, p - 1, w1) }
                if y > 0 {
                    relax(p, p - G, w1)
                    if x > 0 { relax(p, p - G - 1, w2) }
                    if x + 1 < G { relax(p, p - G + 1, w2) }
                }
            }
        }
        for y in stride(from: G - 1, through: 0, by: -1) {
            for x in stride(from: G - 1, through: 0, by: -1) {
                let p = y * G + x
                if x + 1 < G { relax(p, p + 1, w1) }
                if y + 1 < G {
                    relax(p, p + G, w1)
                    if x + 1 < G { relax(p, p + G + 1, w2) }
                    if x > 0 { relax(p, p + G - 1, w2) }
                }
            }
        }
        return (nearest, dist)
    }
}
