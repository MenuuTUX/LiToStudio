import Foundation

/// Cooperative cancellation for a generation run, shared between the UI thread (which
/// requests) and the pipeline/engine thread (which polls at safe points: between
/// pipeline phases, between views, per seed candidate, and once per Heun half-step).
/// An in-flight MLX `eval()` cannot be interrupted, so stop latency is roughly one
/// sampling step or one decode.
public final class GenCancelToken: @unchecked Sendable {
    public enum Mode: Int, Sendable {
        case none = 0
        /// Finish the current sampling candidate, then decode it as the result.
        case afterCandidate
        /// Stop at the next safe point; no result is produced.
        case immediate
    }

    private let lock = NSLock()
    private var _mode: Mode = .none

    public init() {}

    public var mode: Mode {
        lock.lock(); defer { lock.unlock() }
        return _mode
    }

    /// Escalation-only: a later, weaker request never downgrades an immediate stop.
    public func request(_ m: Mode) {
        lock.lock(); defer { lock.unlock() }
        if m.rawValue > _mode.rawValue { _mode = m }
    }

    public var isRequested: Bool { mode != .none }
    public var isImmediate: Bool { mode == .immediate }
}

/// Typed progress events from `LiToEngine.generate` — the structured counterpart to
/// the human-readable `progress(fraction, stage)` strings. Drives the per-view
/// progress tree in the app; the CLI ignores them.
public enum EngineEvent: Sendable {
    /// Conditioning crop + normalization started for a view.
    case viewPreprocessing(view: Int)
    /// Crop done; DINOv2 forward started for a view.
    case viewEncoding(view: Int)
    /// DINOv2 tokens ready for a view (the view's conditioning token set).
    case viewEncoded(view: Int, tokens: Int, dim: Int)
    /// One Heun step finished inside a sampling candidate.
    case samplingStep(candidate: Int, candidates: Int, step: Int, total: Int)
    /// A candidate finished sampling + occupancy decode. `meanIoU` is nil when
    /// scoring was skipped (single candidate, single view).
    case candidateDone(candidate: Int, candidates: Int, meanIoU: Float?)
    /// Best candidate chosen — final voxel/occupancy decode running.
    case decoding
    /// Gaussian decode finished.
    case decodedGaussians(count: Int)
    /// Writing splat / point-cloud artifacts.
    case writingOutput
}
