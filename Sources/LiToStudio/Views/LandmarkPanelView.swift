import SwiftUI
import LiToKit

/// Semantic landmark / object panel: the taxonomy core set as a cross-view
/// visibility matrix, with per-view label correction and per-token inspection.
///
/// Honesty contract: with no grounding backend installed, every status here is a
/// *taxonomy expectation* for the view's label (docs/LANDMARK_TAXONOMY.txt § L) or
/// an Apple Vision pose feature — never a claimed detection. Detected states (filled
/// markers + confidence) can only appear once a real SAM3 adapter exists.
struct LandmarkPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let pkg = model.landmarkPackage {
            VStack(alignment: .leading, spacing: 10) {
                Text("SEMANTIC LANDMARKS")
                    .font(.caption2.weight(.semibold)).foregroundStyle(Theme.dim).tracking(1)

                if pkg.backendAvailable {
                    Label {
                        Text("\(pkg.backend) — detections are real model outputs; remaining cells are expectations. Not consumed by generation.")
                            .font(.system(size: 9))
                            .foregroundStyle(Theme.dim)
                            .fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: "checkmark.seal")
                            .font(.system(size: 10)).foregroundStyle(Theme.good)
                    }
                } else {
                    Label {
                        Text("No grounding backend (SAM3) installed — statuses are taxonomy expectations per view label, not detections. The package is exported with each run but not consumed by generation.")
                            .font(.system(size: 9))
                            .foregroundStyle(Theme.dim)
                            .fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 10)).foregroundStyle(.orange)
                    }
                }

                // Per-view labels with correction menus.
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(pkg.views, id: \.index) { entry in
                        ViewLabelRow(entry: entry)
                    }
                }

                Divider().overlay(Theme.stroke)

                // Legend + matrix header.
                HStack(spacing: 8) {
                    Text("● detected  ◌ not detected  ○ expected  – not expected  ✕ failed  ? unknown")
                        .font(.system(size: 8)).foregroundStyle(Theme.dim)
                    Spacer()
                    HStack(spacing: 2) {
                        ForEach(pkg.views, id: \.index) { e in
                            Text("V\(e.index + 1)")
                                .font(.system(size: 8, weight: .semibold))
                                .foregroundStyle(Theme.dim)
                                .frame(width: 16)
                        }
                    }
                }

                let categories = orderedCategories(pkg)
                ForEach(categories, id: \.self) { category in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(category.uppercased())
                            .font(.system(size: 8, weight: .semibold))
                            .foregroundStyle(Theme.dim.opacity(0.8)).tracking(0.5)
                        ForEach(pkg.visibilityMatrix.filter { $0.category == category }, id: \.id) { row in
                            TokenMatrixRow(row: row, package: pkg)
                        }
                    }
                }

                if let concept = pkg.userConcept {
                    Divider().overlay(Theme.stroke)
                    UserConceptRow(concept: concept, package: pkg)
                }

                Divider().overlay(Theme.stroke)

                // Backend diagnostics — what is actually producing features. Apple
                // Vision output is never labeled as Sapiens2 output.
                VStack(alignment: .leading, spacing: 2) {
                    Label {
                        Text("Apple Vision body pose — active (orientation, framing, raised-hand estimates)")
                            .font(.system(size: 9)).foregroundStyle(Theme.dim)
                    } icon: {
                        Image(systemName: "checkmark.circle")
                            .font(.system(size: 9)).foregroundStyle(Theme.good)
                    }
                    if let pose = pkg.views.compactMap(\.sapiensPose).first {
                        Label {
                            Text("Sapiens2 — \(pose.backend): real keypoints per view (see view rows)")
                                .font(.system(size: 9)).foregroundStyle(Theme.dim)
                                .fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "checkmark.seal")
                                .font(.system(size: 9)).foregroundStyle(Theme.good)
                        }
                    } else if SapiensPoseBackend.isAvailable {
                        Label {
                            Text("Sapiens2 — installed (facebook/sapiens2-pose-0.4b); runs during generation")
                                .font(.system(size: 9)).foregroundStyle(Theme.dim)
                        } icon: {
                            Image(systemName: "circle.dashed")
                                .font(.system(size: 9)).foregroundStyle(Theme.accent)
                        }
                    } else {
                        Label {
                            Text("Sapiens2 pose — \(SapiensPoseBackend.unavailableReason)")
                                .font(.system(size: 9)).foregroundStyle(Theme.dim)
                                .fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "nosign")
                                .font(.system(size: 9)).foregroundStyle(.orange)
                        }
                    }
                    if !pkg.backendAvailable {
                        if Config.sam3CoreMLDir != nil {
                            Label {
                                Text("SAM 3.1 CoreML — installed (weights/sam3-coreml); grounds during generation")
                                    .font(.system(size: 9)).foregroundStyle(Theme.dim)
                            } icon: {
                                Image(systemName: "circle.dashed")
                                    .font(.system(size: 9)).foregroundStyle(Theme.accent)
                            }
                        } else if !Sam3Backend.isAvailable {
                            Label {
                                Text("SAM3 — \(Sam3Backend.unavailableReason)")
                                    .font(.system(size: 9)).foregroundStyle(Theme.dim)
                                    .fixedSize(horizontal: false, vertical: true)
                            } icon: {
                                Image(systemName: "nosign")
                                    .font(.system(size: 9)).foregroundStyle(.orange)
                            }
                        }
                    }
                }
            }
            .padding(14).card()
        }
    }

    private func orderedCategories(_ pkg: LandmarkPackage) -> [String] {
        var seen = [String]()
        for row in pkg.visibilityMatrix where !seen.contains(row.category) {
            seen.append(row.category)
        }
        return seen
    }
}

/// "V3 · right profile (filename)" with a correction menu.
struct ViewLabelRow: View {
    @Environment(AppModel.self) private var model
    let entry: LandmarkPackage.ViewEntry

    var body: some View {
        HStack(spacing: 6) {
            Text("V\(entry.index + 1)")
                .font(.system(size: 10, weight: .semibold)).foregroundStyle(Theme.accent)
            Menu {
                Button("Auto (inferred)") { model.setViewLabel(nil, for: entry.index) }
                Divider()
                ForEach(ViewLabel.allCases.filter { $0 != .unknown }, id: \.self) { label in
                    Button(label.label) { model.setViewLabel(label, for: entry.index) }
                }
            } label: {
                Text("\(entry.viewLabel.label) (\(entry.labelSource))")
                    .font(.system(size: 10))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            if let sp = entry.sapiensPose {
                // Real Sapiens2 keypoints for this view.
                let hands = sp.groups["hands"].map { "hands \($0.visible)/\($0.total)" }
                let face = sp.groups["face"].map { "face \($0.visible)/\($0.total)" }
                Text("· sapiens2: \([face, hands, sp.raisedHand.map { "raised \($0)" }].compactMap { $0 }.joined(separator: " "))")
                    .font(.system(size: 9)).foregroundStyle(Theme.good.opacity(0.9))
                    .lineLimit(1)
            } else if let pose = entry.poseFeatures {
                Text("· vision: \(pose.framing.label)\(pose.raisedHand.map { " · raised hand: \($0)" } ?? "")")
                    .font(.system(size: 9)).foregroundStyle(Theme.dim)
                    .lineLimit(1)
            }
            Spacer()
        }
    }
}

/// One taxonomy token row: id, neutral label, per-view status cells; expands to the
/// full per-view breakdown.
/// The user's free-text guidance segmented by SAM 3.1 — region overlays per view,
/// explicitly marked as text→region (not a DiT geometry conditioner).
struct UserConceptRow: View {
    @Environment(AppModel.self) private var model
    let concept: LandmarkPackage.UserConcept
    let package: LandmarkPackage

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "text.viewfinder").font(.system(size: 10)).foregroundStyle(Theme.accent)
                Text("Text guidance — “\(concept.phrase)”")
                    .font(.system(size: 11, weight: .medium))
                Spacer()
                Text("\(concept.detectionCount)/\(concept.perView.count)")
                    .font(.system(size: 10).monospacedDigit()).foregroundStyle(Theme.dim)
            }
            let hits = detections
            if hits.isEmpty {
                Text("not found in any view")
                    .font(.system(size: 9)).foregroundStyle(.orange)
            } else {
                HStack(spacing: 4) {
                    ForEach(hits, id: \.view) { d in
                        if let img = NSImage(contentsOf: d.url) {
                            Image(nsImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 30, height: 40)
                                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .strokeBorder(Theme.accent.opacity(0.8)))
                                .onTapGesture {
                                    model.lightboxItem = LightboxItem(
                                        url: d.url, title: "V\(d.view + 1) — “\(concept.phrase)”",
                                        subtitle: String(format: "SAM 3.1 text concept · confidence %.2f", d.confidence))
                                }
                        }
                    }
                }
            }
            Text("Segmented by SAM 3.1 from your text. Recorded in the landmark package; does not alter generated geometry (the LiTo checkpoint has no text pathway).")
                .font(.system(size: 9)).foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var detections: [(view: Int, url: URL, confidence: Double)] {
        concept.perView.enumerated().compactMap { i, obs in
            guard let obs, let p = obs.overlayPath ?? obs.maskPath,
                  FileManager.default.fileExists(atPath: p) else { return nil }
            return (i, URL(filePath: p), obs.confidence)
        }
    }
}

struct TokenMatrixRow: View {
    @Environment(AppModel.self) private var model
    let row: LandmarkPackage.TokenRow
    let package: LandmarkPackage
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 2) {
                Button { expanded.toggle() } label: {
                    HStack(spacing: 4) {
                        Image(systemName: expanded ? "chevron.down" : "chevron.right")
                            .font(.system(size: 7)).foregroundStyle(Theme.dim)
                        Text(row.id).font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(Theme.dim)
                        Text(row.token).font(.system(size: 10))
                            .lineLimit(1).truncationMode(.middle)
                    }
                }
                .buttonStyle(.plain)
                Spacer(minLength: 4)
                HStack(spacing: 2) {
                    ForEach(Array(row.perView.enumerated()), id: \.offset) { _, cell in
                        cellView(cell).frame(width: 16)
                    }
                }
            }
            if expanded {
                VStack(alignment: .leading, spacing: 1) {
                    if let tok = LandmarkTaxonomy.coreSet.first(where: { $0.id == row.id }) {
                        Text(tok.summary).font(.system(size: 9)).foregroundStyle(Theme.dim)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(Array(row.perView.enumerated()), id: \.offset) { i, cell in
                        Text("V\(i + 1) (\(viewLabel(i))): \(describe(cell))")
                            .font(.system(size: 9)).foregroundStyle(Theme.dim)
                    }
                    // Real detections — the region highlighted over the photo,
                    // expandable to the lightbox.
                    let masks = detectionMasks
                    if !masks.isEmpty {
                        HStack(spacing: 4) {
                            ForEach(masks, id: \.view) { m in
                                if let img = NSImage(contentsOf: m.url) {
                                    Image(nsImage: img)
                                        .resizable().scaledToFill()
                                        .frame(width: 30, height: 40)
                                        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                                        .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous)
                                            .strokeBorder(Theme.good.opacity(0.7)))
                                        .onTapGesture {
                                            model.lightboxItem = LightboxItem(
                                                url: m.url,
                                                title: "V\(m.view + 1) — \(row.token)",
                                                subtitle: String(format: "SAM 3.1 · confidence %.2f%@",
                                                                 m.confidence,
                                                                 m.personCoverage.map { String(format: " · %.0f%% of subject", $0 * 100) } ?? ""))
                                        }
                                }
                            }
                        }
                        .padding(.top, 2)
                    } else if !package.backendAvailable {
                        Text("detections: none — grounding backend unavailable")
                            .font(.system(size: 9)).foregroundStyle(.orange)
                    }
                }
                .padding(.leading, 14)
            }
        }
    }

    /// Region-over-photo thumbnails for this token across views. Prefers the overlay
    /// (photo with the region highlighted) and falls back to the raw mask only if no
    /// overlay was written.
    private var detectionMasks: [(view: Int, url: URL, confidence: Double, personCoverage: Double?)] {
        package.views.compactMap { entry in
            guard let obs = entry.observations.first(where: { $0.tokenID == row.id }) else { return nil }
            let path = obs.overlayPath ?? obs.maskPath
            guard let p = path, FileManager.default.fileExists(atPath: p) else { return nil }
            return (entry.index, URL(filePath: p), obs.confidence, obs.personCoverage)
        }
    }

    private func viewLabel(_ i: Int) -> String {
        package.views.first(where: { $0.index == i })?.viewLabel.label ?? "?"
    }

    private func describe(_ cell: String) -> String {
        if cell.hasPrefix("detected:") {
            return "detected (confidence \(cell.dropFirst("detected:".count)))"
        }
        switch cell {
        case "not_detected": return "not detected (model ran, nothing found above threshold)"
        case "failed": return "failed (grounding error for this view/token)"
        case "expected": return "expected from taxonomy — no detection (backend unavailable)"
        case "not_expected": return "not expected in this view"
        default: return "unknown (view label unknown)"
        }
    }

    @ViewBuilder
    private func cellView(_ cell: String) -> some View {
        if cell.hasPrefix("detected:") {
            Image(systemName: "circle.fill")
                .font(.system(size: 7)).foregroundStyle(Theme.good)
                .help("detected — confidence \(cell.dropFirst("detected:".count))")
        } else if cell == "not_detected" {
            Image(systemName: "circle.dotted")
                .font(.system(size: 7)).foregroundStyle(.orange)
                .help("not detected — the model ran and found nothing above threshold")
        } else if cell == "failed" {
            Image(systemName: "xmark")
                .font(.system(size: 7)).foregroundStyle(Theme.bad)
                .help("failed — grounding error for this view/token")
        } else if cell == "expected" {
            Image(systemName: "circle")
                .font(.system(size: 7)).foregroundStyle(Theme.accent.opacity(0.7))
                .help("expected from taxonomy — not detected (no backend)")
        } else if cell == "not_expected" {
            Image(systemName: "minus")
                .font(.system(size: 7)).foregroundStyle(Theme.dim.opacity(0.6))
                .help("not expected in this view")
        } else {
            Image(systemName: "questionmark")
                .font(.system(size: 7)).foregroundStyle(.orange)
                .help("unknown — set a view label to get expectations")
        }
    }
}
