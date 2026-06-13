import SwiftUI
import AppKit
import LiToKit

/// The run's progress tree: one branch of stage chips per input view, merging into
/// the shared trunk (conditioning → sampling candidates → decode → outputs).
/// Every stage is always visible — skipped and unavailable stages show as such
/// instead of disappearing. Chips with a thumbnail expand to a lightbox on click.
struct ProgressTreeView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("PIPELINE")
                .font(.caption2.weight(.semibold)).foregroundStyle(Theme.dim).tracking(1)
            ForEach(model.viewStages.indices, id: \.self) { v in
                ViewBranchRow(index: v, stages: model.viewStages[v])
            }
            if !model.trunkStages.isEmpty {
                Divider().overlay(Theme.stroke)
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(model.trunkStages) { record in
                        TrunkStageRow(record: record, progress: candidateProgress(record))
                    }
                }
            }
        }
    }

    /// Fractional progress for the currently sampling candidate's row.
    private func candidateProgress(_ record: StageRecord) -> Double? {
        guard record.status == .running, let s = model.sampling,
              record.stage == "cand\(s.candidate)", s.total > 0 else { return nil }
        return Double(s.step) / Double(s.total)
    }
}

struct ViewBranchRow: View {
    let index: Int
    let stages: [StageRecord]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("VIEW \(index + 1)")
                .font(.system(size: 9, weight: .semibold)).foregroundStyle(Theme.dim).tracking(1)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(stages) { StageChip(record: $0) }
                }
            }
        }
    }
}

struct StageChip: View {
    @Environment(AppModel.self) private var model
    let record: StageRecord

    private static let shortLabels: [String: String] = [
        "original": "orig", "upscale": "4×", "background": "bg", "crop": "crop",
        "dino": "dino", "sapiens": "sapiens", "sam3": "sam3", "token": "token",
    ]

    var body: some View {
        VStack(spacing: 2) {
            ZStack {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Theme.card)
                if let thumb = record.thumbnail, let img = NSImage(contentsOf: thumb) {
                    Image(nsImage: img)
                        .resizable().scaledToFill()
                        .frame(width: 38, height: 38)
                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                        .opacity(record.status == .pending ? 0.4 : 1)
                } else {
                    Image(systemName: symbol)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(color)
                }
                if record.status == .running {
                    ProgressView().controlSize(.small).scaleEffect(0.55)
                }
            }
            .frame(width: 38, height: 38)
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(color.opacity(record.status == .pending ? 0.35 : 0.9),
                                  lineWidth: record.status == .running ? 1.5 : 1)
            )
            Text(Self.shortLabels[record.stage] ?? record.stage)
                .font(.system(size: 8))
                .foregroundStyle(record.status == .pending ? Theme.dim.opacity(0.6) : Theme.dim)
                .lineLimit(1)
        }
        .frame(width: 42)
        .help(helpText)
        .contentShape(Rectangle())
        .onTapGesture {
            guard let thumb = record.thumbnail else { return }
            model.lightboxItem = LightboxItem(
                url: thumb,
                title: "View \((record.view ?? 0) + 1) — \(record.label)",
                subtitle: [record.dims, record.status.rawValue, record.detail]
                    .compactMap { $0 }.joined(separator: " · "))
        }
    }

    private var color: Color {
        switch record.status {
        case .pending: return Theme.dim
        case .running: return Theme.accent
        case .done: return Theme.good
        case .failed: return Theme.bad
        case .skipped, .unavailable: return .orange
        }
    }

    private var symbol: String {
        switch record.status {
        case .pending: return "circle.dotted"
        case .running: return "circle"
        case .done: return "checkmark"
        case .failed: return "xmark"
        case .skipped: return "arrow.right.to.line"
        case .unavailable: return "nosign"
        }
    }

    private var helpText: String {
        var parts = ["\(record.label) — \(record.status.rawValue)"]
        if let d = record.dims { parts.append(d) }
        if let d = record.detail { parts.append(d) }
        parts.append(record.updatedAt.formatted(date: .omitted, time: .standard))
        return parts.joined(separator: "\n")
    }
}

struct TrunkStageRow: View {
    let record: StageRecord
    var progress: Double?

    var body: some View {
        HStack(spacing: 8) {
            Group {
                if record.status == .running {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: symbol)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(color)
                }
            }
            .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(record.label)
                    .font(.system(size: 11, weight: record.status == .running ? .semibold : .regular))
                    .foregroundStyle(record.status == .pending ? Theme.dim : .primary)
                if let d = record.detail {
                    Text(d).font(.system(size: 9)).foregroundStyle(Theme.dim).lineLimit(1)
                }
                if let p = progress {
                    ProgressView(value: p).controlSize(.small).tint(Theme.accent)
                }
            }
            Spacer()
        }
        .help("\(record.label) — \(record.status.rawValue) · \(record.updatedAt.formatted(date: .omitted, time: .standard))")
    }

    private var color: Color {
        switch record.status {
        case .pending: return Theme.dim
        case .running: return Theme.accent
        case .done: return Theme.good
        case .failed: return Theme.bad
        case .skipped, .unavailable: return .orange
        }
    }

    private var symbol: String {
        switch record.status {
        case .pending: return "circle.dotted"
        case .running: return "circle"
        case .done: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        case .skipped: return "arrow.right.to.line.circle"
        case .unavailable: return "nosign"
        }
    }
}

// MARK: - Lightbox

/// Full-window expansion of any pipeline thumbnail: fit-to-screen with pinch /
/// scroll-wheel-modifier zoom, drag pan when zoomed, double-click reset, Esc or the
/// close button (or clicking the backdrop) to dismiss.
struct LightboxView: View {
    let item: LightboxItem
    let onClose: () -> Void

    @State private var zoom: CGFloat = 1
    @State private var lastZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero

    var body: some View {
        ZStack {
            Color.black.opacity(0.93)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture(perform: onClose)

            if let img = NSImage(contentsOf: item.url) {
                Image(nsImage: img)
                    .resizable().scaledToFit()
                    .scaleEffect(zoom)
                    .offset(offset)
                    .padding(44)
                    .gesture(
                        MagnificationGesture()
                            .onChanged { zoom = min(8, max(1, lastZoom * $0)) }
                            .onEnded { _ in
                                lastZoom = zoom
                                if zoom <= 1 { offset = .zero; lastOffset = .zero }
                            }
                    )
                    .simultaneousGesture(
                        DragGesture()
                            .onChanged { g in
                                guard zoom > 1 else { return }
                                offset = CGSize(width: lastOffset.width + g.translation.width,
                                                height: lastOffset.height + g.translation.height)
                            }
                            .onEnded { _ in lastOffset = offset }
                    )
                    .onTapGesture(count: 2) {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            zoom = 1; lastZoom = 1; offset = .zero; lastOffset = .zero
                        }
                    }
            } else {
                Text("Could not load image").foregroundStyle(.white)
            }

            VStack {
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 22))
                            .foregroundStyle(.white.opacity(0.85))
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.cancelAction)
                    .padding(14)
                }
                Spacer()
                VStack(spacing: 2) {
                    Text(item.title)
                        .font(.system(size: 13, weight: .semibold)).foregroundStyle(.white)
                    if !item.subtitle.isEmpty {
                        Text(item.subtitle)
                            .font(.system(size: 11)).foregroundStyle(.white.opacity(0.7))
                    }
                    Text("pinch to zoom · drag to pan · double-click to reset · esc to close")
                        .font(.system(size: 9)).foregroundStyle(.white.opacity(0.45))
                        .padding(.top, 2)
                }
                .padding(12)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .padding(.bottom, 18)
            }
        }
    }
}
