import SwiftUI
import LiToKit

/// The single generation mode the app exposes — LiTo gaussian splat, fully on-device in Swift.
enum GenMode: String, CaseIterable, Identifiable {
    case objectSplat
    var id: String { rawValue }
    var title: String { "Object Splat" }
    var subtitle: String { "LiTo gaussian splat · 100% native, on-device" }
    var symbol: String { "cube.transparent" }
}

/// Where the pipeline currently is.
enum GenPhase: Equatable {
    case idle
    case running(stage: String, progress: Double)
    case done
    case failed(String)
}

/// A snapshot of an intermediate pipeline stage for the preview strip.
struct PipelinePreview: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let imageURL: URL
}

@MainActor
@Observable
final class AppModel {
    var inputImageURL: URL?

    var mode: GenMode = .objectSplat
    var samplingSteps: Double = 20         // Heun ODE steps (reference default 20; more = better geometry)
    var cfgScale: Double = 3.0             // classifier-free guidance scale; 1 = off, 3 = reference default
    var useRMBG: Bool = true               // RMBG 2.0 CoreML background removal (better than Vision)
    var useUpscaler: Bool = true           // Real-ESRGAN 4x CoreML upscale before pipeline
    var occupancyThreshold: Double = 0     // logit cutoff for occupied voxels (0 = sigmoid 0.5, reference)
    var opacityThreshold: Double = 0.10    // drop gaussians below this opacity (point cloud only)

    var extractMesh: Bool = true           // marching-cubes surface extraction after generation
    var seedCandidates: Double = 1         // best-of-N seed search (silhouette-IoU scored)

    var phase: GenPhase = .idle
    var log: [String] = []
    var resultURL: URL?                    // colored point cloud (SceneKit preview)
    var splatURL: URL?                     // full 3DGS gaussian splat (Metal viewer — high fidelity)
    var meshURL: URL?                      // extracted triangle mesh (geometry-first artifact)
    var previews: [PipelinePreview] = []
    var autoRotate: Bool = true
    var liveCloud: [Float] = []            // intermediate occupancy dots while sampling (z-up)
    var liveCloudGen = 0                   // bumped per update so the view knows to refresh

    var isRunning: Bool { if case .running = phase { return true }; return false }
    var canGenerate: Bool { inputImageURL != nil && !isRunning }

    var autoSettings: Bool = true

    func pickImage(_ url: URL) {
        inputImageURL = url; phase = .idle; resultURL = nil; splatURL = nil; meshURL = nil
        log = []; previews = []
        if autoSettings { applyRecommendedSettings(for: url) }
    }

    private func applyRecommendedSettings(for url: URL) {
        let rec = ImageAnalyzer.recommend(imageURL: url)
        samplingSteps = rec.samplingSteps
        cfgScale = rec.cfgScale
        useRMBG = rec.useRMBG
        useUpscaler = rec.useUpscaler
        occupancyThreshold = rec.occupancyThreshold
        opacityThreshold = rec.opacityThreshold
    }
    /// Show an existing model file in the viewer (manual open).
    /// 3DGS gaussian PLYs (our `_gs.ply` exports or any INRIA-format file) go to the
    /// Metal splat renderer; everything else to SceneKit.
    func showModel(_ url: URL) {
        splatURL = nil; resultURL = nil; meshURL = nil
        if url.pathExtension.lowercased() == "ply", Self.isGaussianSplatPLY(url) {
            splatURL = url
        } else {
            resultURL = url
        }
    }

    /// Sniff the PLY header for 3DGS attributes (f_dc_0 ⇒ gaussian splat).
    private static func isGaussianSplatPLY(_ url: URL) -> Bool {
        guard let fh = try? FileHandle(forReadingFrom: url),
              let head = try? fh.read(upToCount: 4096) else { return false }
        try? fh.close()
        return String(decoding: head, as: UTF8.self).contains("f_dc_0")
    }

    private var task: Task<Void, Never>?

    func generate() {
        guard canGenerate, let img = inputImageURL else { return }
        guard Config.ready else {
            phase = .failed("Model weights not found at \(Config.weightsDir.path)")
            return
        }
        phase = .running(stage: "Starting…", progress: 0.01)
        log = []
        resultURL = nil
        splatURL = nil
        meshURL = nil
        previews = []
        liveCloud = []
        liveCloudGen = 0

        let args = PipelineArgs(imagePath: img.path, steps: Int(samplingSteps),
                                cfgScale: Float(cfgScale),
                                useRMBG: useRMBG, useUpscaler: useUpscaler,
                                occupancyThreshold: Float(occupancyThreshold),
                                opacityThreshold: Float(opacityThreshold),
                                seedCandidates: Int(seedCandidates),
                                extractMesh: extractMesh)
        task?.cancel()
        task = Task { [weak self] in
            for await event in runPipeline(args) {
                guard let self else { return }
                switch event {
                case .progress(let stage, let frac):
                    self.phase = .running(stage: stage, progress: frac)
                case .preview(let label, let url):
                    self.previews.append(PipelinePreview(label: label, imageURL: url))
                case .line(let l):
                    self.log.append(l)
                    if self.log.count > 500 { self.log.removeFirst(self.log.count - 500) }
                case .cloud(let points, _, _):
                    self.liveCloud = points
                    self.liveCloudGen += 1
                case .result(let pointCloud, let splat, let mesh):
                    self.resultURL = pointCloud
                    self.splatURL = splat
                    self.meshURL = mesh
                    self.liveCloud = []
                case .failed(let msg):
                    self.phase = .failed(msg)
                }
            }
            guard let self else { return }
            if self.resultURL != nil {
                self.phase = .done
            } else if case .running = self.phase {
                self.phase = .failed("No output produced")
            }
        }
    }

    func cancel() {
        task?.cancel()
        if isRunning { phase = .idle }
    }
}
