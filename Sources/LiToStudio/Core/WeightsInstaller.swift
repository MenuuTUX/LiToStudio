import Foundation
import CryptoKit
import LiToConvertCore

/// First-run model installer: downloads every weight the engine needs into
/// `Config.installDir`, verifies pinned sha256s, and converts Apple's torch
/// checkpoint to safetensors in-process — so a fresh clone (or a bare .app) goes
/// from nothing to generating with one click.
@MainActor
@Observable
final class WeightsInstaller {

    // MARK: manifest

    struct Spec: Sendable {
        let file: String            // destination filename in the weights dir
        let url: URL
        let sha256: String?         // nil = trust TLS (Apple CDN ckpt — too big to pre-pin)
        let approxBytes: Int64      // for disk preflight + progress before headers arrive
        let required: Bool          // gates setup completion; optional failures just warn
        var convertsTo: String? = nil   // torch-zip ckpt → this safetensors file
    }

    /// Everything the downloader can fetch. RMBG 2.0 and Sapiens are not here on
    /// purpose: their licenses don't allow redistribution, so they stay optional
    /// manual installs (the pipeline has fallbacks for both).
    static let manifest: [Spec] = {
        let trellis = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts"
        let release = "https://github.com/MenuuTUX/LiToStudio/releases/download/weights-v1"
        return [
            Spec(file: "ss_dec_conv3d_16l8_fp16.json",
                 url: URL(string: "\(trellis)/ss_dec_conv3d_16l8_fp16.json")!,
                 sha256: "646781293f1cda74720de85d1cef50a957fb4aebd9a4bd014e454e32f2330ac5",
                 approxBytes: 245, required: true),
            Spec(file: "ss_enc_conv3d_16l8_fp16.json",
                 url: URL(string: "\(trellis)/ss_enc_conv3d_16l8_fp16.json")!,
                 sha256: "12efe92a0d7dcd790f251acb94a6950957ea4398268e2838ef7319c3f20b071e",
                 approxBytes: 244, required: false),
            Spec(file: "ss_dec_conv3d_16l8_fp16.safetensors",
                 url: URL(string: "\(trellis)/ss_dec_conv3d_16l8_fp16.safetensors")!,
                 sha256: "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
                 approxBytes: 147_591_972, required: true),
            Spec(file: "ss_enc_conv3d_16l8_fp16.safetensors",
                 url: URL(string: "\(trellis)/ss_enc_conv3d_16l8_fp16.safetensors")!,
                 sha256: "107874eeaa0feb82f51b19db5da7db534fb7e7f19e5a122b9ff1bc2e258bfc6d",
                 approxBytes: 119_068_016, required: false),
            Spec(file: "mlx.metallib",
                 url: URL(string: "\(release)/mlx.metallib")!,
                 sha256: "b258f6f95490b819a217d01654b04cd0a4c53219ad7949185989f23f1ca4f6aa",
                 approxBytes: 104_692_148, required: true),
            Spec(file: "RealESRGAN_x4.mlmodel",
                 url: URL(string: "\(release)/RealESRGAN_x4.mlmodel")!,
                 sha256: "6107dc417de87bf974e5b225a2632e2c78f2849265dc897981f482e922050ec9",
                 approxBytes: 66_857_221, required: false),
            Spec(file: "lito_dit_rgba.ckpt",
                 url: URL(string: "https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt")!,
                 sha256: nil,
                 approxBytes: 7_400_000_000, required: true,
                 convertsTo: "lito.safetensors"),
        ]
    }()

    /// Files that must exist before the engine can run at all.
    static var missingRequired: [Spec] {
        manifest.filter { spec in
            guard spec.required else { return false }
            let target = spec.convertsTo ?? spec.file
            return !FileManager.default.fileExists(atPath: Config.weightsDir.appending(path: target).path)
        }
    }

    /// Anything in the manifest (required or optional) whose end product is absent.
    static var missing: [Spec] {
        manifest.filter { spec in
            let target = spec.convertsTo ?? spec.file
            return !FileManager.default.fileExists(atPath: Config.weightsDir.appending(path: target).path)
        }
    }

    static var needsSetup: Bool { !missingRequired.isEmpty }

    // MARK: observable state

    struct Item: Identifiable {
        enum Phase: Equatable {
            case pending
            case downloading(Double)    // 0…1 (-1 while size unknown)
            case verifying
            case converting(Double)
            case done
            case failed(String)
        }
        let id: String                  // destination filename
        let title: String
        let required: Bool
        var phase: Phase = .pending
    }

    private(set) var items: [Item] = []
    private(set) var running = false
    private(set) var finished = false
    var diskWarning: String?

    private func setPhase(_ id: String, _ phase: Item.Phase) {
        if let i = items.firstIndex(where: { $0.id == id }) { items[i].phase = phase }
    }

    // MARK: orchestration

    func start() {
        guard !running else { return }
        let todo = Self.missing
        guard !todo.isEmpty else { finished = true; return }
        running = true
        items = todo.map { spec in
            Item(id: spec.convertsTo ?? spec.file,
                 title: spec.convertsTo.map { "\($0)  (from \(spec.file))" } ?? spec.file,
                 required: spec.required)
        }
        preflightDisk(for: todo)

        Task {
            let dir = Config.installDir
            for spec in todo {
                let id = spec.convertsTo ?? spec.file
                do {
                    let dest = dir.appending(path: spec.file)
                    setPhase(id, .downloading(-1))
                    try await Self.download(spec: spec, to: dest) { [weak self] done, total in
                        Task { @MainActor [weak self] in
                            self?.setPhase(id, .downloading(total > 0 ? Double(done) / Double(total) : -1))
                        }
                    }
                    if spec.sha256 != nil { setPhase(id, .verifying) }
                    if let product = spec.convertsTo {
                        setPhase(id, .converting(0))
                        let out = dir.appending(path: product)
                        try await Self.convert(ckpt: dest, to: out) { [weak self] done, total in
                            Task { @MainActor [weak self] in
                                self?.setPhase(id, .converting(total > 0 ? Double(done) / Double(total) : 0))
                            }
                        }
                        // the 7.4 GB ckpt has served its purpose — reclaim the space
                        try? FileManager.default.removeItem(at: dest)
                    }
                    setPhase(id, .done)
                } catch {
                    setPhase(id, .failed("\(error)"))
                }
            }
            // make MLX able to start without run.sh's copy step
            Config.ensureMetallibColocated()
            running = false
            finished = !Self.needsSetup
        }
    }

    private func preflightDisk(for todo: [Spec]) {
        // the ckpt and its converted safetensors coexist briefly, so count it twice
        let need = todo.reduce(Int64(0)) { $0 + $1.approxBytes * ($1.convertsTo != nil ? 2 : 1) }
        let values = try? Config.installDir.deletingLastPathComponent()
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let free = values?.volumeAvailableCapacityForImportantUsage, free < need {
            let f = ByteCountFormatter.string(fromByteCount: free, countStyle: .file)
            let n = ByteCountFormatter.string(fromByteCount: need, countStyle: .file)
            diskWarning = "About \(n) of free space is needed during setup; only \(f) is available."
        }
    }

    // MARK: download (streaming hash + HTTP-range resume)

    enum InstallError: Error, CustomStringConvertible {
        case http(Int, String)
        case checksum(String)
        var description: String {
            switch self {
            case .http(let code, let file): return "HTTP \(code) while fetching \(file)"
            case .checksum(let file): return "checksum mismatch for \(file) — partial file removed, retry setup"
            }
        }
    }

    nonisolated static func download(spec: Spec, to dest: URL,
                                     progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let part = dest.appendingPathExtension("part")
        var hasher = SHA256()
        var have: Int64 = 0

        // resume: hash the bytes we already have so the final digest covers the whole file
        if let fh = try? FileHandle(forReadingFrom: part) {
            while let chunk = try fh.read(upToCount: 8 << 20), !chunk.isEmpty {
                hasher.update(data: chunk); have += Int64(chunk.count)
            }
            try fh.close()
        } else {
            fm.createFile(atPath: part.path, contents: nil)
        }

        var req = URLRequest(url: spec.url)
        if have > 0 { req.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }
        let (bytes, resp) = try await URLSession.shared.bytes(for: req)
        guard let http = resp as? HTTPURLResponse else { throw InstallError.http(-1, spec.file) }
        if have > 0 && http.statusCode == 200 {
            // server ignored the Range header — start over
            hasher = SHA256(); have = 0
            try? fm.removeItem(at: part); fm.createFile(atPath: part.path, contents: nil)
        }
        guard http.statusCode == 200 || http.statusCode == 206 else {
            throw InstallError.http(http.statusCode, spec.file)
        }
        let total = have + max(http.expectedContentLength, 0)

        let fh = try FileHandle(forWritingTo: part)
        try fh.seekToEnd()
        defer { try? fh.close() }
        var buf = Data(); buf.reserveCapacity(4 << 20)
        var lastReport = Date.distantPast
        for try await byte in bytes {
            buf.append(byte)
            if buf.count >= 4 << 20 {
                try fh.write(contentsOf: buf)
                hasher.update(data: buf)
                have += Int64(buf.count)
                buf.removeAll(keepingCapacity: true)
                let now = Date()
                if now.timeIntervalSince(lastReport) > 0.2 { progress(have, total); lastReport = now }
            }
        }
        if !buf.isEmpty { try fh.write(contentsOf: buf); hasher.update(data: buf); have += Int64(buf.count) }
        progress(have, max(total, have))

        if let want = spec.sha256 {
            let got = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard got == want else {
                try? fm.removeItem(at: part)
                throw InstallError.checksum(spec.file)
            }
        }
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: part, to: dest)
    }

    /// ckpt → safetensors off the cooperative pool (it's minutes of blocking I/O).
    nonisolated static func convert(ckpt: URL, to out: URL,
                                    progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let thread = Thread {
                do {
                    try CkptConverter.convert(ckpt: ckpt, to: out) { done, total in
                        progress(Int64(done), Int64(total))
                    }
                    cont.resume()
                } catch {
                    try? FileManager.default.removeItem(at: out)   // never leave a torso behind
                    cont.resume(throwing: error)
                }
            }
            thread.stackSize = 8 << 20
            thread.start()
        }
    }
}
