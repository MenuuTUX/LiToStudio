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
    /// Stopped on user request without a result (a finish-candidate stop that
    /// produced a splat ends in `.done` instead).
    case cancelled
}

/// A snapshot of an intermediate pipeline stage for the preview strip.
struct PipelinePreview: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let imageURL: URL
}

/// One node of the run's progress tree (a view-branch stage or a trunk stage),
/// folded from `StageUpdate` events and kept after the run ends.
struct StageRecord: Identifiable, Equatable {
    let view: Int?            // nil = trunk
    let stage: String
    var label: String
    var status: StageStatus
    var thumbnail: URL?
    var dims: String?
    var detail: String?
    var updatedAt: Date
    var id: String { "\(view.map(String.init) ?? "trunk")/\(stage)" }
}

/// A thumbnail expanded to full-window view with its stage metadata.
struct LightboxItem: Identifiable, Equatable {
    let id = UUID()
    let url: URL
    let title: String
    let subtitle: String
}

@MainActor
@Observable
final class AppModel {
    /// One image = classic path. Several images (or one contact sheet, which the
    /// pipeline splits automatically) = multi-view conditioning of a single shape.
    var inputImageURLs: [URL] = []
    var inputImageURL: URL? { inputImageURLs.first }

    var mode: GenMode = .objectSplat
    var multiViewMode: MultiViewMode = .multidiffusion
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
    var autoRotate: Bool = false           // viewport turntable — off by default, manual orbit instead
    var buildShimmer: Bool = true          // subtle pulse on the live cloud while sampling
    var liveCloud: [Float] = []            // intermediate occupancy dots while sampling (z-up)
    var liveCloudGen = 0                   // bumped per update so the view knows to refresh
    var liveCloudProgress: Double = 0      // sampling fraction of the last cloud (drives shading)

    /// Progress tree: one stage row per input view plus the shared trunk. Persists
    /// after the run so earlier stages stay inspectable.
    var viewStages: [[StageRecord]] = []
    var trunkStages: [StageRecord] = []
    /// Latest sampling position (candidate i/N, step s/T) — drives the Stop logic.
    var sampling: (candidate: Int, candidates: Int, step: Int, total: Int)?
    /// Expanded thumbnail, when one is open.
    var lightboxItem: LightboxItem?

    /// Per-view view-label corrections (precedence: user > filename > pose estimate).
    var viewLabelOverrides: [Int: ViewLabel] = [:]
    /// The landmark conditioning package for the current selection — taxonomy priors
    /// + Vision pose features; real detections only once a grounding backend exists.
    private(set) var landmarkPackage: LandmarkPackage?
    /// Optional text guidance. NOT consumed by the pipeline (the checkpoint has no
    /// text conditioning) — recorded in run metadata only.
    var userPromptGuidance: String = ""

    /// Stop/cancel: the token is polled by the pipeline thread at safe points.
    private var cancelToken: GenCancelToken?
    /// A stop has been requested and the pipeline is unwinding (latency ≈ one step).
    private(set) var stopping = false
    /// The "finish candidate or stop now?" dialog should be shown.
    var showStopChoices = false

    var isRunning: Bool { if case .running = phase { return true }; return false }
    var canGenerate: Bool { !inputImageURLs.isEmpty && !isRunning }

    /// Re-runs the analysis the moment the toggle flips on (not just at image pick).
    var autoSettings: Bool = true {
        didSet { if autoSettings, !oldValue { applyRecommendedSettings() } }
    }
    /// The last image analysis — drives the "default → detected" panel in settings.
    private(set) var analysis: ImageAnalyzer.Analysis?

    func pickImage(_ url: URL) { pickImages([url]) }

    func pickImages(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        inputImageURLs = urls; phase = .idle; resultURL = nil; splatURL = nil; meshURL = nil
        log = []; previews = []
        viewStages = []; trunkStages = []; sampling = nil; lightboxItem = nil
        analysis = nil
        viewLabelOverrides = [:]
        rebuildLandmarkPackage()
        if autoSettings { applyRecommendedSettings() }
    }

    // MARK: - landmark package

    /// Correct a view's label by hand (nil reverts to the inferred label).
    func setViewLabel(_ label: ViewLabel?, for index: Int) {
        if let label { viewLabelOverrides[index] = label }
        else { viewLabelOverrides.removeValue(forKey: index) }
        rebuildLandmarkPackage()
    }

    /// The label the package would use for a view, with its source.
    func resolvedViewLabel(for index: Int) -> (label: ViewLabel, source: String) {
        if let o = viewLabelOverrides[index] { return (o, "user") }
        guard index < inputImageURLs.count else { return (.unknown, "none") }
        let name = inputImageURLs[index].deletingPathExtension().lastPathComponent
        if let f = ViewLabel.fromFilename(name) { return (f, "filename") }
        if let va = analysis?.views.first(where: { $0.index == index }),
           va.orientation != .unknown {
            return (ViewLabel(orientation: va.orientation), "pose-estimate")
        }
        return (.unknown, "none")
    }

    /// Rebuild the conditioning package from the current selection: view labels
    /// (user > filename > pose), Vision pose features, and taxonomy priors. Real
    /// detections only once a grounding backend exists — none is installed.
    func rebuildLandmarkPackage() {
        guard !inputImageURLs.isEmpty else { landmarkPackage = nil; return }
        var labels = [ViewLabel](), sources = [String](), pose = [PoseFeatures?]()
        for i in inputImageURLs.indices {
            let (label, source) = resolvedViewLabel(for: i)
            labels.append(label); sources.append(source)
            pose.append(analysis?.views.first(where: { $0.index == i })?.poseFeatures)
        }
        landmarkPackage = LandmarkPackage.build(imagePaths: inputImageURLs.map(\.path),
                                                labels: labels, labelSources: sources,
                                                pose: pose)
    }

    // MARK: - progress tree

    private static let viewStageOrder = ["original", "upscale", "background", "crop",
                                         "dino", "sapiens", "sam3", "token"]

    private static func viewOrder(_ stage: String) -> Int {
        viewStageOrder.firstIndex(of: stage) ?? 99
    }

    private static func trunkOrder(_ stage: String) -> Int {
        switch stage {
        case "merge": return 0
        case "select": return 100
        case "gauss": return 101
        case "texture": return 102
        case "mesh": return 103
        case "result": return 104
        default:  // sampling candidates sort between merge and select
            return stage.hasPrefix("cand") ? (Int(stage.dropFirst(4)) ?? 50) : 99
        }
    }

    func applyStage(_ u: StageUpdate) {
        if let v = u.view {
            while viewStages.count <= v { viewStages.append([]) }
            Self.upsert(&viewStages[v], u, order: Self.viewOrder)
        } else {
            Self.upsert(&trunkStages, u, order: Self.trunkOrder)
        }
    }

    /// Merge an update into a branch: status/label always win, optional fields only
    /// overwrite when the update carries them (a later status flip keeps the thumbnail).
    private static func upsert(_ branch: inout [StageRecord], _ u: StageUpdate,
                               order: (String) -> Int) {
        if let i = branch.firstIndex(where: { $0.stage == u.stage }) {
            branch[i].label = u.label
            branch[i].status = u.status
            if let t = u.thumbnail { branch[i].thumbnail = t }
            if let d = u.dims { branch[i].dims = d }
            if let d = u.detail { branch[i].detail = d }
            branch[i].updatedAt = Date()
        } else {
            branch.append(StageRecord(view: u.view, stage: u.stage, label: u.label,
                                      status: u.status, thumbnail: u.thumbnail,
                                      dims: u.dims, detail: u.detail, updatedAt: Date()))
            branch.sort { order($0.stage) < order($1.stage) }
        }
    }

    /// A run that ends mid-flight leaves its in-flight stages visibly failed (errors)
    /// or skipped (user stop) instead of spinning forever.
    private func markRunningStages(as status: StageStatus, detail: String? = nil) {
        for v in viewStages.indices {
            for i in viewStages[v].indices where viewStages[v][i].status == .running {
                viewStages[v][i].status = status
                if let d = detail { viewStages[v][i].detail = d }
            }
        }
        for i in trunkStages.indices where trunkStages[i].status == .running {
            trunkStages[i].status = status
            if let d = detail { trunkStages[i].detail = d }
        }
    }

    private var analysisTask: Task<Void, Never>?

    /// Analysis runs off-main (Vision body-pose per view is not free for 6 images);
    /// results are dropped if the selection changed or auto was toggled off meanwhile.
    private func applyRecommendedSettings() {
        guard !inputImageURLs.isEmpty else { return }
        let urls = inputImageURLs
        analysisTask?.cancel()
        analysisTask = Task { [weak self] in
            let a = await Task.detached(priority: .userInitiated) {
                ImageAnalyzer.analyze(imageURLs: urls)
            }.value
            guard !Task.isCancelled, let self,
                  self.inputImageURLs == urls, self.autoSettings else { return }
            self.analysis = a
            self.samplingSteps = a.settings.samplingSteps
            self.cfgScale = a.settings.cfgScale
            self.useRMBG = a.settings.useRMBG
            self.useUpscaler = a.settings.useUpscaler
            self.occupancyThreshold = a.settings.occupancyThreshold
            self.opacityThreshold = a.settings.opacityThreshold
            self.seedCandidates = a.settings.seedCandidates
            self.rebuildLandmarkPackage()   // analysis brings pose features + labels
        }
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
        guard canGenerate else { return }
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
        viewStages = []
        trunkStages = []
        sampling = nil
        lightboxItem = nil
        stopping = false
        showStopChoices = false
        let token = GenCancelToken()
        cancelToken = token

        let args = PipelineArgs(imagePaths: inputImageURLs.map(\.path), steps: Int(samplingSteps),
                                cfgScale: Float(cfgScale),
                                multiView: multiViewMode,
                                useRMBG: useRMBG, useUpscaler: useUpscaler,
                                occupancyThreshold: Float(occupancyThreshold),
                                opacityThreshold: Float(opacityThreshold),
                                seedCandidates: Int(seedCandidates),
                                extractMesh: extractMesh,
                                analysis: analysis,
                                landmarks: landmarkPackage,
                                userPrompt: userPromptGuidance.trimmingCharacters(in: .whitespacesAndNewlines))
        task?.cancel()
        task = Task { [weak self] in
            for await event in runPipeline(args, cancel: token) {
                guard let self else { return }
                switch event {
                case .progress(let stage, let frac):
                    self.phase = .running(stage: stage, progress: frac)
                case .preview(let label, let url):
                    self.previews.append(PipelinePreview(label: label, imageURL: url))
                case .line(let l):
                    self.log.append(l)
                    if self.log.count > 500 { self.log.removeFirst(self.log.count - 500) }
                case .cloud(let points, let step, let total):
                    self.liveCloud = points
                    self.liveCloudGen += 1
                    self.liveCloudProgress = total > 0 ? Double(step) / Double(total) : 0
                case .stage(let update):
                    self.applyStage(update)
                case .landmarks(let pkg):
                    // Real backend outputs replace the priors-only package.
                    self.landmarkPackage = pkg
                case .sampling(let candidate, let candidates, let step, let total):
                    self.sampling = (candidate, candidates, step, total)
                case .result(let pointCloud, let splat, let mesh):
                    self.resultURL = pointCloud
                    self.splatURL = splat
                    self.meshURL = mesh
                    self.liveCloud = []
                case .failed(let msg):
                    self.phase = .failed(msg)
                    self.markRunningStages(as: .failed)
                case .cancelled:
                    self.phase = .cancelled
                    self.markRunningStages(as: .skipped, detail: "stopped by user")
                }
            }
            guard let self else { return }
            if self.resultURL != nil {
                self.phase = .done
            } else if case .running = self.phase {
                self.phase = .failed("No output produced")
            }
            self.stopping = false
        }
    }

    /// Stop button entry point: when the current candidate is almost complete
    /// (≥ 80 % or ≤ 5 steps left), offer to finish it; otherwise stop immediately.
    func stopTapped() {
        guard isRunning, !stopping else { return }
        if let s = sampling, s.total > 0,
           Double(s.step) / Double(s.total) >= 0.8 || (s.total - s.step) <= 5 {
            showStopChoices = true
        } else {
            requestStop(.immediate)
        }
    }

    /// Cooperative stop: the pipeline/engine polls the token at safe points; an
    /// in-flight MLX eval can't be interrupted, so unwinding takes ≈ one step.
    func requestStop(_ mode: GenCancelToken.Mode) {
        guard isRunning, mode != .none else { return }
        stopping = true
        cancelToken?.request(mode)
        if case .running(_, let p) = phase {
            phase = .running(stage: mode == .afterCandidate
                                 ? "Finishing candidate, then stopping…" : "Stopping…",
                             progress: p)
        }
    }

    func cancel() {
        requestStop(.immediate)
    }
}
