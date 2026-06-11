import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ContentView: View {
    // First run on a machine without the models: show setup instead of the app.
    @State private var needsSetup = WeightsInstaller.needsSetup

    var body: some View {
        if needsSetup {
            SetupView { needsSetup = false }
        } else {
            HStack(spacing: 0) {
                Sidebar()
                    .frame(width: 384)
                Divider().overlay(Theme.stroke)
                ViewerPane()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
        }
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HeaderBar()
                DropZone()
                ModeGrid()
                SettingsPanel()
                GenerateButton()
                StatusArea()
                Spacer(minLength: 0)
            }
            .padding(20)
        }
        .scrollIndicators(.never)
        .background(Theme.bg)
    }
}

struct HeaderBar: View {
    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .fill(Theme.accentGradient)
                .frame(width: 40, height: 40)
                .overlay(Image(systemName: "cube.transparent.fill")
                    .font(.system(size: 19, weight: .bold))
                    .foregroundStyle(.white))
            VStack(alignment: .leading, spacing: 1) {
                Text("LiTo Studio").font(.system(size: 19, weight: .bold))
                Text("photo → 3D · on-device").font(.caption).foregroundStyle(Theme.dim)
            }
            Spacer()
        }
    }
}

// MARK: - Drop zone

struct DropZone: View {
    @Environment(AppModel.self) private var model
    @State private var importing = false
    @State private var targeted = false

    var body: some View {
        VStack(spacing: 10) {
            if let url = model.inputImageURL, let img = NSImage(contentsOf: url) {
                Image(nsImage: img)
                    .resizable().scaledToFit()
                    .frame(maxHeight: 196)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                Text(url.lastPathComponent)
                    .font(.caption).foregroundStyle(Theme.dim).lineLimit(1)
            } else {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 34)).foregroundStyle(Theme.accent)
                Text("Drop a photo").font(.headline)
                Text("or click to choose").font(.caption).foregroundStyle(Theme.dim)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 26)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Theme.card)
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                        .foregroundStyle(targeted ? Theme.accent : Theme.stroke)
                )
        )
        .contentShape(Rectangle())
        .onTapGesture { importing = true }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            model.pickImage(url)
            return true
        } isTargeted: { targeted = $0 }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result { model.pickImage(url) }
        }
        .animation(.easeInOut(duration: 0.15), value: targeted)
    }
}

// MARK: - Mode grid

struct ModeGrid: View {
    @Environment(AppModel.self) private var model
    private let cols = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("MODE").font(.caption2.weight(.semibold)).foregroundStyle(Theme.dim).tracking(1)
            LazyVGrid(columns: cols, spacing: 10) {
                ForEach(GenMode.allCases) { mode in
                    ModeCard(mode: mode, selected: model.mode == mode)
                        .onTapGesture { model.mode = mode }
                }
            }
        }
    }
}

struct ModeCard: View {
    let mode: GenMode
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: mode.symbol)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(selected ? .white : Theme.accent)
                Spacer()
            }
            Text(mode.title).font(.system(size: 13, weight: .semibold))
            Text(mode.subtitle).font(.system(size: 10)).foregroundStyle(Theme.dim)
                .lineLimit(2).fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, minHeight: 78, alignment: .topLeading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(selected ? AnyShapeStyle(Theme.accentGradient) : AnyShapeStyle(Theme.card))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(selected ? Color.clear : Theme.stroke)
        )
        .contentShape(Rectangle())
        .animation(.easeInOut(duration: 0.15), value: selected)
    }
}

// MARK: - Settings

struct SettingsPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 14) {
            Toggle("Auto-detect settings from image", isOn: $model.autoSettings)
                .font(.system(size: 13, weight: .medium))
            Divider().overlay(Theme.stroke)
            SliderRow(title: "Sampling steps", value: $model.samplingSteps,
                      range: 4...60, step: 1, format: "%.0f")
            Text("Heun ODE steps — the dominant quality knob. 20 = reference default, 25–40 = recommended (cleaner geometry, fewer ghost volumes), diminishing returns past ~50.")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
            SliderRow(title: "Guidance (CFG)", value: $model.cfgScale,
                      range: 1...7, step: 0.5, format: "%.1f")
            Text("Classifier-free guidance — how strongly the shape follows the image. 1 = off (mushy blob), 3 = reference default. Higher can over-sharpen / add artifacts.")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
            SliderRow(title: "Occupancy cutoff", value: $model.occupancyThreshold,
                      range: 0...3, step: 0.25, format: "%.2f")
            SliderRow(title: "Seed search (best of N)", value: $model.seedCandidates,
                      range: 1...5, step: 1, format: "%.0f")
            Text("Samples N different seeds and keeps the one whose silhouette best matches the photo — strong for tricky poses/occlusions, N× the sampling time.")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
            SliderRow(title: "Opacity cutoff", value: $model.opacityThreshold,
                      range: 0.05...0.6, step: 0.05, format: "%.2f")
            Text("Higher cutoffs prune ghost voxels / floaters (the stray wisps & spikes) — at the cost of some real detail.")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
            Divider().overlay(Theme.stroke)
            Toggle("RMBG 2.0 background removal", isOn: $model.useRMBG)
                .font(.system(size: 13))
            Toggle("Real-ESRGAN 4x upscale", isOn: $model.useUpscaler)
                .font(.system(size: 13))
            Text("Requires .mlpackage models in the weights folder.")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
            Toggle("Extract mesh (marching cubes)", isOn: $model.extractMesh)
                .font(.system(size: 13))
            Text("Also writes a triangle-mesh .ply/.obj surface (geometry-first artifact, opens in Blender etc.).")
                .font(.system(size: 10)).foregroundStyle(Theme.dim)
        }
        .toggleStyle(.switch)
        .tint(Theme.accent)
        .padding(16)
        .card()
    }

    private func label(_ t: String, _ sym: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: sym).font(.system(size: 12)).foregroundStyle(Theme.accent).frame(width: 16)
            Text(t).font(.system(size: 13))
        }
    }
}

struct SliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.system(size: 13))
                Spacer()
                Text(String(format: format, value))
                    .font(.system(size: 13, weight: .semibold).monospacedDigit())
                    .foregroundStyle(Theme.accent)
            }
            Slider(value: $value, in: range, step: step).tint(Theme.accent)
        }
    }
}

// MARK: - Generate

struct GenerateButton: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Button(action: { model.generate() }) {
            HStack(spacing: 8) {
                if model.isRunning {
                    ProgressView().controlSize(.small).tint(.white)
                    Text("Generating…")
                } else {
                    Image(systemName: "sparkles")
                    Text("Generate 3D")
                }
            }
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity).frame(height: 46)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(model.canGenerate || model.isRunning ? AnyShapeStyle(Theme.accentGradient)
                                                               : AnyShapeStyle(Theme.card))
            )
            .opacity(model.canGenerate || model.isRunning ? 1 : 0.5)
        }
        .buttonStyle(.plain)
        .disabled(!model.canGenerate)
    }
}

// MARK: - Status

struct StatusArea: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch model.phase {
            case .idle:
                EmptyView()
            case .running(let stage, let progress):
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(stage).font(.system(size: 12)).foregroundStyle(Theme.dim)
                        Spacer()
                        Text("\(Int(progress * 100))%").font(.system(size: 12, weight: .semibold).monospacedDigit())
                    }
                    ProgressView(value: progress).tint(Theme.accent)
                }
                .padding(14).card()
            case .done:
                Label("Done — see the viewer", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.good)
                    .padding(14).card()
            case .failed(let msg):
                Label(msg, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.bad)
                    .padding(14).card()
            }

            if !model.previews.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("PIPELINE STAGES").font(.caption2.weight(.semibold)).foregroundStyle(Theme.dim).tracking(1)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(model.previews) { preview in
                                VStack(spacing: 3) {
                                    if let img = NSImage(contentsOf: preview.imageURL) {
                                        Image(nsImage: img)
                                            .resizable()
                                            .scaledToFit()
                                            .frame(width: 64, height: 64)
                                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous)
                                                .strokeBorder(Theme.stroke))
                                    }
                                    Text(preview.label)
                                        .font(.system(size: 9))
                                        .foregroundStyle(Theme.dim)
                                        .lineLimit(1)
                                        .frame(width: 64)
                                }
                            }
                        }
                    }
                }
                .padding(14).card()
            }
        }
    }
}

// MARK: - Viewer (placeholder until SceneKit is wired)

enum ViewerMode: String, CaseIterable {
    case splat = "Splat", mesh = "Mesh", points = "Points"
    var symbol: String {
        switch self {
        case .splat: return "sparkles"
        case .mesh: return "square.3.layers.3d"
        case .points: return "circle.grid.3x3.fill"
        }
    }
}

struct ViewerPane: View {
    @Environment(AppModel.self) private var model
    @State private var opening = false
    @State private var mode: ViewerMode = .splat

    private var modelTypes: [UTType] {
        ["obj", "ply", "usdz", "usd", "stl"].compactMap { UTType(filenameExtension: $0) }
    }

    private var running: (String, Double)? {
        if case .running(let s, let f) = model.phase { return (s, f) }
        return nil
    }

    /// Which modes have an artifact to show.
    private var available: [ViewerMode] {
        var m = [ViewerMode]()
        if model.splatURL != nil, SplatView.isSupported { m.append(.splat) }
        if model.meshURL != nil { m.append(.mesh) }
        if model.resultURL != nil { m.append(.points) }
        return m
    }

    private func url(for mode: ViewerMode) -> URL? {
        switch mode {
        case .splat: return model.splatURL
        case .mesh: return model.meshURL
        case .points: return model.resultURL
        }
    }

    var body: some View {
        ZStack {
            Theme.sceneGradient.ignoresSafeArea()

            if !available.isEmpty {
                let active = available.contains(mode) ? mode : available[0]
                let activeURL = url(for: active)!

                if active == .splat {
                    SplatView(url: activeURL, autoRotate: model.autoRotate)
                        .ignoresSafeArea()
                } else {
                    SceneKitView(url: activeURL, autoRotate: model.autoRotate)
                        .ignoresSafeArea()
                }
                VStack {
                    HStack {
                        Spacer()
                        if available.count > 1 {
                            HStack(spacing: 2) {
                                ForEach(available, id: \.self) { m in
                                    Button { mode = m } label: {
                                        Label(m.rawValue, systemImage: m.symbol)
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundStyle(m == active ? Color.white : .white.opacity(0.55))
                                            .padding(.horizontal, 10).padding(.vertical, 7)
                                            .background(m == active ? AnyShapeStyle(Theme.accent.opacity(0.55))
                                                                    : AnyShapeStyle(.clear),
                                                        in: Capsule())
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(3)
                            .background(.ultraThinMaterial, in: Capsule())
                            .padding(.top, 16)
                        }
                        Button { model.autoRotate.toggle() } label: {
                            Image(systemName: model.autoRotate ? "rotate.3d" : "pause.circle")
                                .font(.system(size: 16, weight: .medium))
                                .foregroundStyle(.white.opacity(0.85))
                                .padding(8)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .help(model.autoRotate ? "Pause rotation" : "Resume rotation")
                        .padding(16)
                    }
                    Spacer()
                    HStack(spacing: 10) {
                        Label(activeURL.lastPathComponent, systemImage: "cube.fill")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.white.opacity(0.85))
                            .lineLimit(1)
                        Spacer()
                        pill("Reveal", "magnifyingglass") {
                            NSWorkspace.shared.activateFileViewerSelecting([activeURL])
                        }
                        pill("Open file…", "folder") { opening = true }
                    }
                    .padding(12)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .padding(16)
                }
            } else if running == nil {
                VStack(spacing: 16) {
                    Image(systemName: "rotate.3d")
                        .font(.system(size: 56, weight: .thin))
                        .foregroundStyle(Theme.dim)
                    Text("Drop a photo, pick a mode,\nthen hit Generate 3D")
                        .multilineTextAlignment(.center)
                        .font(.title3).foregroundStyle(Theme.dim)
                    Button { opening = true } label: {
                        Label("…or open an existing .obj / .ply", systemImage: "folder")
                            .font(.system(size: 12))
                    }
                    .buttonStyle(.bordered).tint(Theme.dim)
                }
            }

            if let (stage, frac) = running {
                // The shape assembling live: intermediate occupancy decodes as white dots.
                if !model.liveCloud.isEmpty {
                    LiveCloudView(points: model.liveCloud, generation: model.liveCloudGen)
                        .ignoresSafeArea()
                        .transition(.opacity)
                }
                VStack {
                    Spacer()
                    VStack(spacing: 12) {
                        Text(stage).font(.headline).foregroundStyle(.white)
                        ProgressView(value: frac).frame(width: 240).tint(Theme.accent)
                        Text("\(Int(frac * 100))%")
                            .font(.caption.monospacedDigit()).foregroundStyle(Theme.dim)
                        if model.liveCloud.isEmpty, !model.previews.isEmpty {
                            PreviewStrip()
                        }
                    }
                    .padding(22)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .padding(.bottom, 28)
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: model.isRunning)
        .fileImporter(isPresented: $opening, allowedContentTypes: modelTypes) { result in
            if case .success(let url) = result { model.showModel(url) }
        }
    }

    private func pill(_ title: String, _ symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol).font(.system(size: 12, weight: .medium))
        }
        .buttonStyle(.bordered).tint(Theme.accent)
    }
}

// MARK: - Preview strip (intermediate pipeline outputs)

struct PreviewStrip: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(model.previews) { preview in
                    VStack(spacing: 4) {
                        if let img = NSImage(contentsOf: preview.imageURL) {
                            Image(nsImage: img)
                                .resizable()
                                .scaledToFit()
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        } else {
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(Theme.card)
                                .frame(width: 80, height: 80)
                        }
                        Text(preview.label)
                            .font(.system(size: 9))
                            .foregroundStyle(Theme.dim)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.top, 8)
        }
        .frame(maxWidth: 320)
    }
}
