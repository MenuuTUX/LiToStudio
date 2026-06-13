import Foundation
import CoreGraphics

// Landmark grounding scaffold. There is NO segmentation-grounding model (SAM3 or
// otherwise) in this project today — `UnavailableLandmarkSegmenter` is the only
// backend and it reports exactly that. What runs for real: view-label inference
// (user > filename > Vision pose estimate), taxonomy visibility priors, and pose
// features from Apple Vision. The exported package is clearly marked as priors,
// not detections, and as NOT consumed by the generator.
// Canonical taxonomy: docs/LANDMARK_TAXONOMY.txt (core set embedded below).

/// Canonical camera-relative view labels for a multi-view capture set.
public enum ViewLabel: String, CaseIterable, Sendable, Codable {
    case front
    case frontRightOblique = "front_right_oblique"
    case rightProfile = "right_profile"
    case back
    case frontLeftOblique = "front_left_oblique"
    case leftProfile = "left_profile"
    case unknown

    public var label: String {
        switch self {
        case .front: return "front"
        case .frontRightOblique: return "front-right oblique"
        case .rightProfile: return "right profile"
        case .back: return "back"
        case .frontLeftOblique: return "front-left oblique"
        case .leftProfile: return "left profile"
        case .unknown: return "unknown"
        }
    }

    /// Map the analyzer's Vision-based orientation estimate onto a view label.
    public init(orientation: ViewOrientation) {
        switch orientation {
        case .front: self = .front
        case .frontObliqueLeft: self = .frontLeftOblique
        case .frontObliqueRight: self = .frontRightOblique
        case .profileLeft: self = .leftProfile
        case .profileRight: self = .rightProfile
        case .back: self = .back
        case .unknown: self = .unknown
        }
    }

    /// Infer from an explicit filename token ("right_profile", "front_left", "back" …).
    /// Returns nil unless the name is unambiguous.
    public static func fromFilename(_ name: String) -> ViewLabel? {
        let n = name.lowercased()
        if n.contains("front_left") || n.contains("left_oblique") { return .frontLeftOblique }
        if n.contains("front_right") || n.contains("right_oblique") { return .frontRightOblique }
        if n.contains("left_profile") || n.contains("profile_left") { return .leftProfile }
        if n.contains("right_profile") || n.contains("profile_right") { return .rightProfile }
        if n.contains("back") || n.contains("rear") { return .back }
        if n.contains("front") { return .front }
        return nil
    }
}

/// One token of the core landmark set (docs/LANDMARK_TAXONOMY.txt § K).
public struct LandmarkToken: Sendable, Codable, Identifiable, Equatable {
    public let id: String        // taxonomy id, e.g. "L006"
    public let token: String     // neutral academic name, e.g. "chain_belt"
    public let category: String
    public let summary: String
    /// Short noun phrase sent to the grounding model (concept prompt); the UI/package
    /// always shows the taxonomy token, never this.
    public let prompt: String
}

/// The embedded core landmark set + per-view visibility priors, from the canonical
/// taxonomy file. Priors are *expectations by view label* (§ L of the taxonomy),
/// not detections — the UI and package must keep that distinction visible.
public enum LandmarkTaxonomy {

    public static let coreSet: [LandmarkToken] = [
        .init(id: "L001", token: "face_region", category: "face / identity",
              summary: "Face area, visible mostly in front and oblique views",
              prompt: "face"),
        .init(id: "L002", token: "hair_volume", category: "hair",
              summary: "Full hair mass; rear sheet / side profile in back and side views",
              prompt: "hair"),
        .init(id: "L003", token: "upper_torso_garment", category: "upper-torso garment",
              summary: "Upper-torso garment region (incl. rear strap in back views)",
              prompt: "swimsuit top"),
        .init(id: "L004", token: "abdomen_navel_region", category: "body region",
              summary: "Abdomen region surrounding the navel",
              prompt: "abdomen"),
        .init(id: "L005", token: "navel_piercing", category: "accessories",
              summary: "Small high-detail navel jewelry landmark",
              prompt: "navel piercing"),
        .init(id: "L006", token: "chain_belt", category: "waist accessories",
              summary: "Chain belt assembly — strong cross-view waist anchor",
              prompt: "chain belt"),
        .init(id: "L007", token: "hip_strap_line", category: "waist accessories",
              summary: "Hip strap line — side/back waist landmark",
              prompt: "hip strap"),
        .init(id: "L008", token: "gloves", category: "handwear",
              summary: "Fingerless handwear, visible across views",
              prompt: "fingerless gloves"),
        .init(id: "L009", token: "cargo_pants", category: "lower garment",
              summary: "Lower-body garment anchor",
              prompt: "cargo pants"),
        .init(id: "L010", token: "pants_pockets", category: "lower garment",
              summary: "Back/side garment pocket details",
              prompt: "pants pocket"),
        .init(id: "L011", token: "belt_charms", category: "waist accessories",
              summary: "Dangling charm details on the chain belt",
              prompt: "charm pendant"),
        .init(id: "L012", token: "fingernails", category: "handwear",
              summary: "Fine hand-detail landmark (nail tips)",
              prompt: "fingernails"),
    ]

    /// Expected-visible tokens per view label (taxonomy § L, mapped onto the core
    /// set: e.g. "rear_hair_sheet"/"side_hair_profile" ⊂ hair_volume,
    /// "back_garment_strap" ⊂ upper_torso_garment, "back_pockets" ⊂ pants_pockets).
    public static func expectedTokens(for label: ViewLabel) -> Set<String> {
        let allViews: Set<String> = ["hair_volume", "upper_torso_garment", "chain_belt",
                                     "gloves", "cargo_pants"]
        switch label {
        case .front:
            return allViews.union(["face_region", "abdomen_navel_region", "navel_piercing",
                                   "belt_charms", "fingernails"])
        case .frontRightOblique, .frontLeftOblique:
            return allViews.union(["face_region", "abdomen_navel_region", "navel_piercing",
                                   "hip_strap_line", "belt_charms", "fingernails"])
        case .rightProfile, .leftProfile:
            return allViews.union(["abdomen_navel_region", "navel_piercing",
                                   "hip_strap_line", "pants_pockets", "belt_charms"])
        case .back:
            return allViews.union(["hip_strap_line", "pants_pockets"])
        case .unknown:
            return []
        }
    }
}

/// Pose/framing features for one view — produced by Apple Vision body pose (real,
/// on-device). Explicitly NOT Sapiens output; `source` says so.
public struct PoseFeatures: Sendable, Codable {
    public let source: String                 // "apple-vision-body-pose"
    public let framing: ViewFraming
    public let orientation: ViewOrientation
    public let orientationConfidence: Double
    /// Which hand (if any) is raised to/above head level.
    public let raisedHand: String?            // "left" / "right" / "both" / nil
    /// What a human-feature model (Sapiens2, once installed) should prioritize for
    /// this framing — a recommendation, not an output.
    public let suggestedFocus: String

    public init(framing: ViewFraming, orientation: ViewOrientation,
                orientationConfidence: Double, raisedHand: String?) {
        self.source = "apple-vision-body-pose"
        self.framing = framing
        self.orientation = orientation
        self.orientationConfidence = orientationConfidence
        self.raisedHand = raisedHand
        var focus: String
        switch framing {
        case .faceCrop: focus = "facial structure and identity detail"
        case .upperBody: focus = "head, shoulder, arm, and hand structure"
        case .torso: focus = "torso pose, arm/hand structure, hip orientation"
        case .headToKnees: focus = "body pose, torso, arms, hands, hip and knee structure"
        case .fullBody: focus = "full-body pose and proportions"
        case .unknown: focus = "general geometry"
        }
        switch orientation {
        case .back: focus += "; rear hair and garment structure"
        case .profileLeft, .profileRight: focus += "; side hair and garment contours"
        default: break
        }
        self.suggestedFocus = focus
    }
}

/// One real detection from a grounding backend. Only ever produced by an actual
/// model adapter — nothing in this project fabricates these today.
public struct LandmarkObservation: Sendable, Codable {
    public let tokenID: String
    public let token: String
    /// Normalized [x, y, w, h], top-left origin, in the view image.
    public let box: [Double]
    /// Binary mask PNG (data artifact for consumers).
    public let maskPath: String?
    /// Region highlighted over the original photo (what the UI shows — far more
    /// legible than the raw mask).
    public var overlayPath: String?
    public let confidence: Double
    public let viewIndex: Int
    public let viewLabel: ViewLabel
    /// Mask area as a fraction of the person silhouette (0 when no person mask was
    /// available) — surfaced so a large region reads as intentional, not a bug.
    public var personCoverage: Double?

    public init(tokenID: String, token: String, box: [Double], maskPath: String?,
                overlayPath: String? = nil, confidence: Double, viewIndex: Int,
                viewLabel: ViewLabel, personCoverage: Double? = nil) {
        self.tokenID = tokenID; self.token = token; self.box = box
        self.maskPath = maskPath; self.overlayPath = overlayPath
        self.confidence = confidence; self.viewIndex = viewIndex
        self.viewLabel = viewLabel; self.personCoverage = personCoverage
    }
}

/// The grounding backend seam. A SAM3 adapter implements this once a checkpoint +
/// CoreML/MLX conversion exists; until then `UnavailableLandmarkSegmenter` is the
/// only implementation.
public protocol LandmarkSegmenter {
    var backendName: String { get }
    var isAvailable: Bool { get }
    func detect(imageURL: URL, tokens: [LandmarkToken],
                viewIndex: Int, viewLabel: ViewLabel) throws -> [LandmarkObservation]
}

public struct UnavailableLandmarkSegmenter: LandmarkSegmenter {
    public enum SegmenterError: Error, CustomStringConvertible {
        case modelNotInstalled
        public var description: String {
            "No landmark-grounding model installed — SAM3 checkpoint + on-device conversion pending."
        }
    }

    public let backendName = "unavailable (no SAM3 model installed)"
    public let isAvailable = false
    public init() {}
    public func detect(imageURL: URL, tokens: [LandmarkToken],
                       viewIndex: Int, viewLabel: ViewLabel) throws -> [LandmarkObservation] {
        throw SegmenterError.modelNotInstalled
    }
}

/// The per-run conditioning package: per-view entries (x_i: image, view label, pose
/// features, expected tokens, real observations when a backend exists) plus the
/// cross-view visibility matrix. Exported as `<base>_landmarks.json`.
///
/// NOT consumed by generation: the DiT cross-attends DINOv2 image tokens only — there
/// is no auxiliary conditioning channel. Integration point: `DiT.sample(conds:)`
/// already takes an arbitrary list of token streams, so a *projected* landmark/text
/// stream could join after a conditioning fine-tune (see
/// docs/LITO_PROMPT_GUIDANCE_RESEARCH.md). `consumedByGenerator` stays false until
/// that exists.
public struct LandmarkPackage: Sendable, Codable {
    public struct ViewEntry: Sendable, Codable {
        public let index: Int
        public let imagePath: String
        public let viewLabel: ViewLabel
        public let labelSource: String        // "user" | "filename" | "pose-estimate" | "none"
        public let poseFeatures: PoseFeatures?
        public let expectedTokens: [String]   // taxonomy § L priors for this label
        public var observations: [LandmarkObservation]  // real backend detections only
        /// Real Sapiens2 pose record (nil until the pose backend has run).
        public var sapiensPose: HumanPoseRecord?
    }

    /// One row of the cross-view visibility matrix. Cell values:
    /// "detected:<confidence>" / "not_detected" / "failed" (real backend outcomes) ·
    /// "expected" / "not_expected" (taxonomy prior, backend absent) ·
    /// "unknown" (view label unknown).
    public struct TokenRow: Sendable, Codable {
        public let id: String
        public let token: String
        public let category: String
        public let perView: [String]
    }

    /// The user's optional free-text guidance, segmented by SAM 3.1 across views.
    /// Real text→region (not a taxonomy prior); like the rest of the package it is
    /// recorded and shown but NOT consumed by DiT geometry (the checkpoint has no
    /// text pathway).
    public struct UserConcept: Sendable, Codable {
        public let phrase: String
        public let backend: String
        public let perView: [LandmarkObservation?]   // nil = not present in that view
        public var detectionCount: Int { perView.lazy.filter { $0 != nil }.count }
        public init(phrase: String, backend: String, perView: [LandmarkObservation?]) {
            self.phrase = phrase; self.backend = backend; self.perView = perView
        }
    }

    public var schema = 1
    public let createdAt: Date
    public let backend: String
    public let backendAvailable: Bool
    public let consumedByGenerator: Bool
    public let consumptionNote: String
    public let taxonomyFile: String
    public let views: [ViewEntry]
    public let visibilityMatrix: [TokenRow]
    public var userConcept: UserConcept? = nil

    /// Assemble the package from real inputs. `observations` come from an actual
    /// backend run (empty today); priors fill the matrix where no detection exists.
    public static func build(imagePaths: [String],
                             labels: [ViewLabel],
                             labelSources: [String],
                             pose: [PoseFeatures?],
                             observations: [[LandmarkObservation]] = [],
                             segmenter: LandmarkSegmenter = UnavailableLandmarkSegmenter()) -> LandmarkPackage {
        let n = imagePaths.count
        var entries = [ViewEntry]()
        for i in 0 ..< n {
            let label = i < labels.count ? labels[i] : .unknown
            entries.append(ViewEntry(
                index: i, imagePath: imagePaths[i], viewLabel: label,
                labelSource: i < labelSources.count ? labelSources[i] : "none",
                poseFeatures: i < pose.count ? pose[i] : nil,
                expectedTokens: LandmarkTaxonomy.expectedTokens(for: label).sorted(),
                observations: i < observations.count ? observations[i] : [],
                sapiensPose: nil))
        }
        return LandmarkPackage(
            createdAt: Date(),
            backend: segmenter.backendName,
            backendAvailable: segmenter.isAvailable,
            consumedByGenerator: false,
            consumptionNote: "Generated but NOT consumed: the DiT conditions on DINOv2 image tokens only (no auxiliary channel). Matrix cells are taxonomy priors unless marked detected:<conf> / not_detected / failed. Integration point: DiT.sample(conds:) after a conditioning fine-tune.",
            taxonomyFile: "docs/LANDMARK_TAXONOMY.txt",
            views: entries,
            visibilityMatrix: matrix(for: entries, findings: nil))
    }

    /// Inject REAL backend outcomes (a SAM3 run and/or Sapiens2 pose records) into a
    /// priors-only package: observations land on their views, matrix cells flip from
    /// priors to detected/not_detected/failed, and the backend string reflects what
    /// actually ran.
    public func applying(sam3: Sam3RunResult?, pose: [HumanPoseRecord?]?) -> LandmarkPackage {
        var entries = views
        if let sam3 {
            for i in entries.indices where i < sam3.perView.count {
                entries[i].observations = sam3.perView[i].compactMap(\.observation)
            }
        }
        if let pose {
            for i in entries.indices where i < pose.count {
                entries[i].sapiensPose = pose[i]
            }
        }
        return LandmarkPackage(
            schema: schema,
            createdAt: Date(),
            backend: sam3?.backend ?? backend,
            backendAvailable: sam3 != nil || backendAvailable,
            consumedByGenerator: false,
            consumptionNote: consumptionNote,
            taxonomyFile: taxonomyFile,
            views: entries,
            visibilityMatrix: Self.matrix(for: entries, findings: sam3?.perView),
            userConcept: userConcept)
    }

    /// Matrix cells: real outcomes when a grounding run exists for the view,
    /// taxonomy priors otherwise.
    private static func matrix(for entries: [ViewEntry],
                               findings: [[Sam3RunResult.Sam3Finding]]?) -> [TokenRow] {
        var matrix = [TokenRow]()
        for tok in LandmarkTaxonomy.coreSet {
            var cells = [String]()
            for (i, e) in entries.enumerated() {
                if let f = findings, i < f.count, !f[i].isEmpty,
                   let finding = f[i].first(where: { $0.id == tok.id }) {
                    switch finding.status {
                    case "detected":
                        cells.append(String(format: "detected:%.2f",
                                            finding.observation?.confidence ?? 0))
                    case "failed":
                        cells.append("failed")
                    default:
                        cells.append("not_detected")
                    }
                } else if let det = e.observations.first(where: { $0.tokenID == tok.id }) {
                    cells.append(String(format: "detected:%.2f", det.confidence))
                } else if e.viewLabel == .unknown {
                    cells.append("unknown")
                } else {
                    cells.append(e.expectedTokens.contains(tok.token) ? "expected" : "not_expected")
                }
            }
            matrix.append(TokenRow(id: tok.id, token: tok.token,
                                   category: tok.category, perView: cells))
        }
        return matrix
    }
}
