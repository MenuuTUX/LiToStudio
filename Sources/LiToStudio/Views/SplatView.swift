import SwiftUI
import MetalKit
import MetalSplatter
import SplatIO
import simd

/// True gaussian-splat viewer: renders the 3DGS `.ply` the engine exports with proper
/// alpha blending, per-splat extent and full SH3 view-dependent color — the high-fidelity
/// view (the SceneKit point cloud is only a preview fallback).
///
/// LiTo world space is +Z up (reference viewer: viewUp=[0,0,1]); the view matrix bakes a
/// −90° X rotation so models stand upright under the y-up camera. Drag orbits, scroll zooms.
struct SplatView: NSViewRepresentable {
    let url: URL
    var autoRotate: Bool

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SplatMTKView {
        let view = SplatMTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.layer?.isOpaque = false
        view.preferredFramesPerSecond = 60
        context.coordinator.attach(view: view)
        return view
    }

    func updateNSView(_ view: SplatMTKView, context: Context) {
        context.coordinator.autoRotate = autoRotate
        context.coordinator.load(url: url)
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        private var view: SplatMTKView?
        private var device: MTLDevice?
        private var queue: MTLCommandQueue?
        private var renderer: SplatRenderer?
        private var loadedURL: URL?
        private var loadTask: Task<Void, Never>?
        private var center = SIMD3<Float>(0, 0, 0)
        private var radius: Float = 1

        var autoRotate = true
        private var yaw: Float = 0          // turntable angle (radians)
        private var pitch: Float = 0        // user-controlled elevation
        private var distanceScale: Float = 1
        private var lastFrame = Date()
        private let inFlight = DispatchSemaphore(value: 3)
        private var drawableSize = CGSize(width: 1, height: 1)

        func attach(view: SplatMTKView) {
            self.view = view
            self.device = view.device
            self.queue = view.device?.makeCommandQueue()
            view.delegate = self
            view.onDrag = { [weak self] dx, dy in
                guard let self else { return }
                self.yaw += Float(dx) * 0.01
                self.pitch = max(-1.2, min(1.2, self.pitch + Float(dy) * 0.01))
            }
            view.onScroll = { [weak self] dz in
                guard let self else { return }
                self.distanceScale = max(0.3, min(4, self.distanceScale * (1 - Float(dz) * 0.02)))
            }
        }

        func load(url: URL) {
            guard url != loadedURL, let device else { return }
            loadedURL = url
            loadTask?.cancel()
            loadTask = Task { [weak self] in
                do {
                    let points = try await AutodetectSceneReader(url).readAll()
                    guard !Task.isCancelled, let self, let view = self.view else { return }
                    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
                    for p in points { lo = simd_min(lo, p.position); hi = simd_max(hi, p.position) }
                    self.center = (lo + hi) / 2
                    self.radius = max(simd_length(hi - lo) / 2, 1e-3)
                    let renderer = try SplatRenderer(device: device,
                                                     colorFormat: view.colorPixelFormat,
                                                     depthFormat: view.depthStencilPixelFormat,
                                                     sampleCount: view.sampleCount,
                                                     maxViewCount: 1,
                                                     maxSimultaneousRenders: 3)
                    let chunk = try SplatChunk(device: device, from: points)
                    _ = await renderer.addChunk(chunk)
                    self.renderer = renderer
                } catch {
                    NSLog("SplatView: failed to load \(url.lastPathComponent): \(error)")
                }
            }
        }

        private var viewport: SplatRenderer.ViewportDescriptor {
            let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
            let proj = perspective(fovyRadians: .pi * 35 / 180, aspect: aspect,
                                   nearZ: 0.02, farZ: radius * 60)
            // y-up camera orbiting the (z-up) model: T(-dist) · Rx(pitch) · Ry(yaw) · Rx(-90°) · T(-center)
            let dist = radius * 3.2 * distanceScale
            let m = translation(-center)
            let upCal = rotation(radians: -.pi / 2, axis: SIMD3(1, 0, 0))
            let spin = rotation(radians: yaw, axis: SIMD3(0, 1, 0))
            let elev = rotation(radians: pitch, axis: SIMD3(1, 0, 0))
            let viewM = translation(SIMD3(0, 0, -dist)) * elev * spin * upCal * m
            return SplatRenderer.ViewportDescriptor(
                viewport: MTLViewport(originX: 0, originY: 0,
                                      width: Double(drawableSize.width), height: Double(drawableSize.height),
                                      znear: 0, zfar: 1),
                projectionMatrix: proj, viewMatrix: viewM,
                screenSize: SIMD2(Int(drawableSize.width), Int(drawableSize.height)))
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { drawableSize = size }

        func draw(in view: MTKView) {
            let now = Date()
            if autoRotate { yaw += Float(now.timeIntervalSince(lastFrame)) * (.pi * 2 / 18) }
            lastFrame = now
            guard let renderer, renderer.isReadyToRender,
                  let drawable = view.currentDrawable, let queue else { return }
            inFlight.wait()
            guard let cmd = queue.makeCommandBuffer() else { inFlight.signal(); return }
            let sem = inFlight
            cmd.addCompletedHandler { _ in sem.signal() }
            var didRender = false
            do {
                didRender = try renderer.render(viewports: [viewport],
                                                colorTexture: drawable.texture,
                                                colorStoreAction: .store,
                                                depthTexture: view.depthStencilTexture,
                                                rasterizationRateMap: nil,
                                                renderTargetArrayLength: 0,
                                                to: cmd)
            } catch {
                NSLog("SplatView render error: \(error)")
            }
            if didRender { cmd.present(drawable) }
            cmd.commit()
        }
    }
}

/// MTKView with simple orbit (drag) + zoom (scroll) callbacks.
final class SplatMTKView: MTKView {
    var onDrag: ((CGFloat, CGFloat) -> Void)?
    var onScroll: ((CGFloat) -> Void)?

    override var acceptsFirstResponder: Bool { true }
    override func mouseDragged(with event: NSEvent) { onDrag?(event.deltaX, event.deltaY) }
    override func scrollWheel(with event: NSEvent) { onScroll?(event.scrollingDeltaY) }
}

// MARK: - matrix helpers (right-handed, from Apple sample math)

private func rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
    let u = normalize(axis), ct = cosf(radians), st = sinf(radians), ci = 1 - ct
    let (x, y, z) = (u.x, u.y, u.z)
    return simd_float4x4(columns: (
        SIMD4(ct + x*x*ci, y*x*ci + z*st, z*x*ci - y*st, 0),
        SIMD4(x*y*ci - z*st, ct + y*y*ci, z*y*ci + x*st, 0),
        SIMD4(x*z*ci + y*st, y*z*ci - x*st, ct + z*z*ci, 0),
        SIMD4(0, 0, 0, 1)))
}

private func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
                            SIMD4(0, 0, 1, 0), SIMD4(t.x, t.y, t.z, 1)))
}

private func perspective(fovyRadians fovy: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
    let ys = 1 / tanf(fovy * 0.5), xs = ys / aspect, zs = farZ / (nearZ - farZ)
    return simd_float4x4(columns: (SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0),
                                   SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * nearZ, 0)))
}
