import Foundation

/// Bridge to the Python model backend in `tools/backend/` (one uv venv, see its
/// README). The Swift app stays clean: workers are separate processes that take a
/// manifest JSON path and print exactly one JSON document; masks/keypoints land as
/// files. Workers exit between runs, so their memory never overlaps MLX sampling.
public enum PythonBackend {

    /// `tools/backend` — env override, else walk up from the executable, else cwd.
    public static var backendDir: URL? {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["LITO_BACKEND_DIR"], !env.isEmpty {
            return URL(filePath: env)
        }
        var dir = Bundle.main.bundleURL
        for _ in 0 ..< 8 {
            let cand = dir.appending(path: "tools/backend")
            if fm.fileExists(atPath: cand.appending(path: "setup.sh").path) { return cand }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }
        let cwd = URL(filePath: fm.currentDirectoryPath).appending(path: "tools/backend")
        return fm.fileExists(atPath: cwd.appending(path: "setup.sh").path) ? cwd : nil
    }

    /// The venv interpreter, when `tools/backend/setup.sh` has been run.
    public static var python: URL? {
        guard let dir = backendDir else { return nil }
        let p = dir.appending(path: ".venv/bin/python")
        return FileManager.default.fileExists(atPath: p.path) ? p : nil
    }

    /// Whether a HuggingFace repo is present in the local hub cache — the cheap
    /// availability proxy (a worker still reports real load errors if it isn't
    /// actually usable).
    public static func modelCached(_ repo: String) -> Bool {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".cache/huggingface/hub/models--\(repo.replacingOccurrences(of: "/", with: "--"))/snapshots")
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return false }
        return !items.isEmpty
    }

    public enum BackendError: Error, CustomStringConvertible {
        case notInstalled
        case timeout
        case cancelled
        case worker(String, instruction: String?)
        case badOutput(String)
        public var description: String {
            switch self {
            case .notInstalled: return "Python backend not installed — run tools/backend/setup.sh"
            case .timeout: return "backend worker timed out"
            case .cancelled: return "backend worker cancelled"
            case .worker(let e, let i): return i.map { "\(e) — \($0)" } ?? e
            case .badOutput(let s): return "backend worker produced no JSON: \(s.prefix(200))"
            }
        }
    }

    /// Run a worker script with an Encodable manifest; returns its stdout JSON.
    /// Polls `cancel` once a second and terminates the worker on an immediate stop.
    public static func run<M: Encodable>(script: String, manifest: M,
                                         timeout: TimeInterval = 1800,
                                         cancel: GenCancelToken? = nil) throws -> Data {
        guard let python, let dir = backendDir else { throw BackendError.notInstalled }
        let manifestURL = FileManager.default.temporaryDirectory
            .appending(path: "lito_manifest_\(UUID().uuidString).json")
        try JSONEncoder().encode(manifest).write(to: manifestURL)
        defer { try? FileManager.default.removeItem(at: manifestURL) }

        // stdout/stderr go to temp files: no pipe-buffer deadlocks on chatty model
        // loads, and nothing shared across threads.
        let fm = FileManager.default
        let outURL = fm.temporaryDirectory.appending(path: "lito_worker_out_\(UUID().uuidString)")
        let errURL = fm.temporaryDirectory.appending(path: "lito_worker_err_\(UUID().uuidString)")
        fm.createFile(atPath: outURL.path, contents: nil)
        fm.createFile(atPath: errURL.path, contents: nil)
        defer { try? fm.removeItem(at: outURL); try? fm.removeItem(at: errURL) }

        let proc = Process()
        proc.executableURL = python
        proc.arguments = [dir.appending(path: script).path, manifestURL.path]
        proc.standardOutput = try FileHandle(forWritingTo: outURL)
        proc.standardError = try FileHandle(forWritingTo: errURL)
        try proc.run()

        let deadline = Date().addingTimeInterval(timeout)
        while proc.isRunning {
            if cancel?.isImmediate == true {
                proc.terminate()
                throw BackendError.cancelled
            }
            if Date() > deadline {
                proc.terminate()
                throw BackendError.timeout
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        let data = (try? Data(contentsOf: outURL)) ?? Data()
        if data.isEmpty {
            let stderrText = (try? String(contentsOf: errURL, encoding: .utf8)) ?? ""
            throw BackendError.badOutput(stderrText)
        }
        return data
    }
}

// MARK: - CLIP tokenizer (for SAM 3.1 free-text concepts)

/// Tokenizes free-text concept phrases into CLIP-BPE ids for the SAM 3.1 CoreML text
/// encoder. The 12 taxonomy prompts are pre-baked (`prompt_tokens.json`); only user
/// free-text needs this, and only the venv (no model download).
public enum ClipTokenizer {
    public static var isAvailable: Bool { PythonBackend.python != nil }

    struct Manifest: Encodable { let phrases: [String] }
    struct Response: Decodable {
        let ok: Bool
        let error: String?
        let instruction: String?
        let tokens: [[Int]]?
    }

    /// Returns one 32-length id array per phrase. Throws if the venv is missing.
    public static func tokenize(_ phrases: [String]) throws -> [[Int]] {
        let data = try PythonBackend.run(script: "tokenize_clip.py",
                                         manifest: Manifest(phrases: phrases), timeout: 120)
        let resp = try JSONDecoder().decode(Response.self, from: data)
        guard resp.ok, let tokens = resp.tokens else {
            throw PythonBackend.BackendError.worker(resp.error ?? "tokenize failed",
                                                    instruction: resp.instruction)
        }
        return tokens
    }
}

// MARK: - RMBG worker (PyTorch fallback for background removal)

/// Background removal through the Python backend — the real fallback when the
/// CoreML conversion of RMBG-2.0 is unavailable (BiRefNet uses
/// `torchvision::deform_conv2d`, which coremltools cannot convert as of 9.x).
/// Same briaai/RMBG-2.0 weights, PyTorch on MPS, one process per batch.
public enum RMBGWorkerBackend {
    public static let repo = "briaai/RMBG-2.0"

    public static var isAvailable: Bool {
        PythonBackend.python != nil && PythonBackend.modelCached(repo)
    }

    public struct CutoutResult: Sendable {
        public let backend: String
        public let cutouts: [URL?]    // aligned with input images
    }

    struct Manifest: Encodable {
        struct View: Encodable { let index: Int; let image: String }
        let views: [View]
        let outDir: String
    }

    struct Response: Decodable {
        let ok: Bool
        let backend: String?
        let error: String?
        let instruction: String?
        let views: [V]?
        struct V: Decodable { let viewIndex: Int; let cutout: String? }
    }

    public static func cutouts(images: [URL], outDir: URL,
                               cancel: GenCancelToken? = nil) throws -> CutoutResult {
        let manifest = Manifest(
            views: images.enumerated().map { .init(index: $0, image: $1.path) },
            outDir: outDir.path)
        let data = try PythonBackend.run(script: "rmbg_worker.py", manifest: manifest, cancel: cancel)
        let resp = try JSONDecoder().decode(Response.self, from: data)
        guard resp.ok, let views = resp.views else {
            throw PythonBackend.BackendError.worker(resp.error ?? "unknown", instruction: resp.instruction)
        }
        var out = [URL?](repeating: nil, count: images.count)
        for v in views where v.viewIndex < images.count {
            if let c = v.cutout { out[v.viewIndex] = URL(filePath: c) }
        }
        return CutoutResult(backend: resp.backend ?? "RMBG-2.0 (python)", cutouts: out)
    }
}

// MARK: - SAM3 landmark grounding backend

/// What one SAM3 run produced for the matrix/package: per view, per token, a real
/// detected/not-detected/failed outcome (never a prior).
public struct Sam3RunResult: Sendable {
    public let backend: String
    /// Aligned with the run's views; each entry maps tokenID → outcome.
    public let perView: [[Sam3Finding]]
    public struct Sam3Finding: Sendable {
        public let id: String
        public let token: String
        public let status: String              // detected | not_detected | failed
        public let observation: LandmarkObservation?
    }
    public var detectionCount: Int {
        perView.reduce(0) { $0 + $1.filter { $0.status == "detected" }.count }
    }
}

public enum Sam3Backend {
    public static let repo = "facebook/sam3"

    /// Backend venv exists AND the gated checkpoint is in the HF cache.
    public static var isAvailable: Bool {
        PythonBackend.python != nil && PythonBackend.modelCached(repo)
    }
    /// Human-readable reason when unavailable (drives UI text).
    public static var unavailableReason: String {
        if PythonBackend.python == nil { return "Python backend not installed — run tools/backend/setup.sh" }
        return "facebook/sam3 not downloaded — request access at huggingface.co/facebook/sam3, then `hf download facebook/sam3`"
    }

    struct Manifest: Encodable {
        struct View: Encodable { let index: Int; let label: String; let image: String }
        struct Prompt: Encodable { let id: String; let token: String; let phrase: String }
        let views: [View]
        let prompts: [Prompt]
        let outDir: String
        let threshold: Double
    }

    struct Response: Decodable {
        let ok: Bool
        let backend: String?
        let error: String?
        let instruction: String?
        let views: [ViewResult]?
        struct ViewResult: Decodable {
            let viewIndex: Int
            let viewLabel: String
            let landmarks: [Landmark]
        }
        struct Landmark: Decodable {
            let id: String
            let label: String
            let status: String
            let bbox: [Double]?
            let maskPath: String?
            let confidence: Double?
        }
    }

    /// Ground the taxonomy core set in every view. One worker process for the whole
    /// batch (model loads once); masks land in `masksDir`.
    public static func detect(images: [URL], labels: [ViewLabel], masksDir: URL,
                              threshold: Double = 0.4,
                              cancel: GenCancelToken? = nil) throws -> Sam3RunResult {
        let manifest = Manifest(
            views: images.enumerated().map { i, url in
                .init(index: i, label: (i < labels.count ? labels[i] : .unknown).rawValue,
                      image: url.path)
            },
            prompts: LandmarkTaxonomy.coreSet.map { .init(id: $0.id, token: $0.token, phrase: $0.prompt) },
            outDir: masksDir.path,
            threshold: threshold)
        let data = try PythonBackend.run(script: "sam3_worker.py", manifest: manifest, cancel: cancel)
        let resp = try JSONDecoder().decode(Response.self, from: data)
        guard resp.ok, let viewResults = resp.views else {
            throw PythonBackend.BackendError.worker(resp.error ?? "unknown", instruction: resp.instruction)
        }
        var perView = [[Sam3RunResult.Sam3Finding]](repeating: [], count: images.count)
        for vr in viewResults {
            guard vr.viewIndex < images.count else { continue }
            perView[vr.viewIndex] = vr.landmarks.map { lm in
                var obs: LandmarkObservation?
                if lm.status == "detected", let box = lm.bbox, let conf = lm.confidence {
                    obs = LandmarkObservation(tokenID: lm.id, token: lm.label, box: box,
                                              maskPath: lm.maskPath, confidence: conf,
                                              viewIndex: vr.viewIndex,
                                              viewLabel: ViewLabel(rawValue: vr.viewLabel) ?? .unknown)
                }
                return .init(id: lm.id, token: lm.label, status: lm.status, observation: obs)
            }
        }
        return Sam3RunResult(backend: resp.backend ?? "sam3", perView: perView)
    }
}

// MARK: - Sapiens2 pose backend

/// Real Sapiens2 per-view pose record (summary; full keypoints in `keypointsFile`).
public struct HumanPoseRecord: Sendable, Codable {
    public struct GroupSummary: Sendable, Codable {
        public let visible: Int
        public let total: Int
        public let meanConfidence: Double
    }
    public let backend: String
    public let personBox: [Double]?     // normalized xywh crop used
    public let keypointCount: Int
    public let keypointsFile: String?
    public let groups: [String: GroupSummary]
    public let raisedHand: String?
}

public enum SapiensPoseBackend {
    public static let repo = "facebook/sapiens2-pose-0.4b"

    public static var isAvailable: Bool {
        PythonBackend.python != nil && PythonBackend.modelCached(repo)
    }
    public static var unavailableReason: String {
        if PythonBackend.python == nil { return "Python backend not installed — run tools/backend/setup.sh" }
        return "\(repo) not downloaded — run `hf download \(repo)`"
    }

    struct Manifest: Encodable {
        struct View: Encodable {
            let index: Int
            let label: String
            let image: String
            let subjectBox: [Double]?
        }
        let views: [View]
        let outDir: String
    }

    struct Response: Decodable {
        let ok: Bool
        let backend: String?
        let error: String?
        let instruction: String?
        let views: [ViewResult]?
        struct ViewResult: Decodable {
            let viewIndex: Int
            let personBox: [Double]?
            let keypointCount: Int
            let keypointsFile: String?
            let groups: [String: HumanPoseRecord.GroupSummary]
            let raisedHand: String?
        }
    }

    /// Extract pose for every view in one worker process. `subjectBoxes` are
    /// normalized xywh person-crop hints (from the real RMBG cutout / subject
    /// estimate) — the honest substitute for the upstream RTMDet detector.
    public static func extract(images: [URL], labels: [ViewLabel],
                               subjectBoxes: [[Double]?], outDir: URL,
                               cancel: GenCancelToken? = nil) throws -> [HumanPoseRecord?] {
        let manifest = Manifest(
            views: images.enumerated().map { i, url in
                .init(index: i, label: (i < labels.count ? labels[i] : .unknown).rawValue,
                      image: url.path,
                      subjectBox: i < subjectBoxes.count ? subjectBoxes[i] : nil)
            },
            outDir: outDir.path)
        let data = try PythonBackend.run(script: "sapiens2_worker.py", manifest: manifest, cancel: cancel)
        let resp = try JSONDecoder().decode(Response.self, from: data)
        guard resp.ok, let viewResults = resp.views else {
            throw PythonBackend.BackendError.worker(resp.error ?? "unknown", instruction: resp.instruction)
        }
        var out = [HumanPoseRecord?](repeating: nil, count: images.count)
        for vr in viewResults where vr.viewIndex < images.count {
            out[vr.viewIndex] = HumanPoseRecord(
                backend: resp.backend ?? "sapiens2",
                personBox: vr.personBox,
                keypointCount: vr.keypointCount,
                keypointsFile: vr.keypointsFile,
                groups: vr.groups,
                raisedHand: vr.raisedHand)
        }
        return out
    }
}
