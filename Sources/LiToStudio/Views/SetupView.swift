import SwiftUI

/// First-run experience: shown instead of the main UI while required model files
/// are missing. One click downloads everything (and converts Apple's checkpoint),
/// then hands off to the normal app.
struct SetupView: View {
    @State private var installer = WeightsInstaller()
    var onComplete: () -> Void

    private var totalLabel: String {
        ByteCountFormatter.string(
            fromByteCount: WeightsInstaller.missing.reduce(0) { $0 + $1.approxBytes },
            countStyle: .file)
    }

    var body: some View {
        ZStack {
            Theme.sceneGradient.ignoresSafeArea()
            VStack(spacing: 22) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 44, weight: .light))
                    .foregroundStyle(Theme.accentGradient)
                Text("Set up LiTo Studio")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(.white)
                Text("The models that power on-device 3D generation aren't bundled with the app. One click downloads them (~\(totalLabel), one time) and converts Apple's LiTo checkpoint locally — nothing ever leaves your Mac.")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.dim)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)

                if let warn = installer.diskWarning {
                    Label(warn, systemImage: "externaldrive.badge.exclamationmark")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Theme.bad)
                        .frame(maxWidth: 440)
                }

                if !installer.items.isEmpty {
                    VStack(spacing: 10) {
                        ForEach(installer.items) { item in
                            SetupRow(item: item)
                        }
                    }
                    .padding(16)
                    .frame(width: 480)
                    .card()
                }

                if installer.finished {
                    Button {
                        onComplete()
                    } label: {
                        Text("Start creating")
                            .font(.system(size: 14, weight: .semibold))
                            .padding(.horizontal, 28).padding(.vertical, 10)
                            .background(Theme.accentGradient, in: Capsule())
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.plain)
                } else {
                    Button {
                        installer.start()
                    } label: {
                        Text(installer.running ? "Setting up…"
                             : installer.items.contains(where: { if case .failed = $0.phase { return true }; return false })
                               ? "Retry" : "Download & set up")
                            .font(.system(size: 14, weight: .semibold))
                            .padding(.horizontal, 28).padding(.vertical, 10)
                            .background(installer.running ? AnyShapeStyle(Theme.card)
                                                          : AnyShapeStyle(Theme.accentGradient),
                                        in: Capsule())
                            .foregroundStyle(installer.running ? Theme.dim : .white)
                    }
                    .buttonStyle(.plain)
                    .disabled(installer.running)
                }

                VStack(spacing: 4) {
                    Text("Installing to \(Config.installDir.path)")
                    Text("Optional extras (RMBG 2.0 background removal, Sapiens refinement) have licenses that don't allow bundling — see SETUP.md. The app works without them.")
                        .multilineTextAlignment(.center)
                }
                .font(.system(size: 11))
                .foregroundStyle(Theme.dim.opacity(0.7))
                .frame(maxWidth: 460)
            }
            .padding(40)
        }
    }
}

private struct SetupRow: View {
    let item: WeightsInstaller.Item

    var body: some View {
        HStack(spacing: 10) {
            statusIcon
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(item.title)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                    if !item.required {
                        Text("optional")
                            .font(.system(size: 9, weight: .semibold))
                            .padding(.horizontal, 5).padding(.vertical, 1.5)
                            .background(Theme.cardHi, in: Capsule())
                            .foregroundStyle(Theme.dim)
                    }
                    Spacer()
                    Text(detail)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.dim)
                }
                if let frac = barFraction {
                    ProgressView(value: max(frac, 0))
                        .progressViewStyle(.linear)
                        .tint(Theme.accent)
                }
            }
        }
    }

    private var barFraction: Double? {
        switch item.phase {
        case .downloading(let f): return f >= 0 ? f : 0
        case .converting(let f): return f
        default: return nil
        }
    }

    private var detail: String {
        switch item.phase {
        case .pending: return "waiting"
        case .downloading(let f): return f >= 0 ? "downloading \(Int(f * 100))%" : "downloading"
        case .verifying: return "verifying"
        case .converting(let f): return "converting \(Int(f * 100))%"
        case .done: return "done"
        case .failed(let msg): return msg
        }
    }

    @ViewBuilder private var statusIcon: some View {
        switch item.phase {
        case .pending:
            Image(systemName: "circle.dotted").foregroundStyle(Theme.dim)
        case .downloading, .converting, .verifying:
            ProgressView().controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.good)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.bad)
        }
    }
}
