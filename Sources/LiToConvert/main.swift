import Foundation

// LiToConvert — one-time, dev-only weight converter (Foundation-only; not shipped).
//
//   LiToConvert <input.ckpt|.pth> [output.safetensors] [--list] [--limit N] [--grep S]
//
// Reads a torch-zip checkpoint and either lists its tensors (--list) or writes a
// standard .safetensors file. No Python, no MLX — pure Swift.

func fail(_ msg: String) -> Never { FileHandle.standardError.write(Data((msg + "\n").utf8)); exit(1) }

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    fail("usage: LiToConvert <input.ckpt|.pth> [output.safetensors] [--list] [--limit N] [--grep S]")
}
let input = URL(filePath: argv[1])
let listOnly = argv.contains("--list")
var limit = 30
if let i = argv.firstIndex(of: "--limit"), i + 1 < argv.count { limit = Int(argv[i + 1]) ?? 30 }
var grep: String? = nil
if let i = argv.firstIndex(of: "--grep"), i + 1 < argv.count { grep = argv[i + 1] }
let output: URL? = argv.dropFirst(2).first(where: { !$0.hasPrefix("-") }).map { URL(filePath: $0) }

func human(_ bytes: Int) -> String {
    let u = ["B", "KB", "MB", "GB", "TB"]; var v = Double(bytes); var i = 0
    while v >= 1024 && i < u.count - 1 { v /= 1024; i += 1 }
    return String(format: "%.2f %@", v, u[i])
}

do {
    let t0 = Date()
    FileHandle.standardError.write(Data("[convert] opening \(input.lastPathComponent)…\n".utf8))
    let zip = try TorchZip(url: input)
    let (pkl, prefix) = try zip.pickle()
    let root = try TorchUnpickler(pkl).load()
    var tensors: [(String, TorchTensor)] = []
    root.stateDict.flattenTensors(into: &tensors)
    guard !tensors.isEmpty else { fail("[convert] no tensors found in state_dict") }

    let totalElems = tensors.reduce(0) { $0 + $1.1.numel }
    let totalBytes = tensors.reduce(0) { $0 + $1.1.numel * $1.1.storage.dtype.itemSize }

    if listOnly {
        print("tensors: \(tensors.count)   params: \(totalElems)   raw: \(human(totalBytes))   prefix: '\(prefix)'")
        var shown = 0
        for (name, t) in tensors {
            if let g = grep, !name.contains(g) { continue }
            if shown < limit {
                let contig = t.isContiguous ? "" : "  [strided off=\(t.offset)]"
                print("  \(t.storage.dtype.safetensors)\t\(t.shape)\t\(name)\(contig)")
            }
            shown += 1
        }
        if shown > limit { print("  … and \(shown - limit) more") }
        exit(0)
    }

    guard let out = output else { fail("[convert] need an output path (or pass --list)") }
    FileHandle.standardError.write(Data("[convert] writing \(tensors.count) tensors (\(human(totalBytes))) → \(out.lastPathComponent)…\n".utf8))
    let (n, bytes) = try Safetensors.write(tensors, zip: zip, prefix: prefix, to: out)
    let dt = Date().timeIntervalSince(t0)
    print("[convert] ✓ \(n) tensors, \(human(bytes)) in \(String(format: "%.1f", dt))s → \(out.path)")
} catch {
    fail("[convert] ERROR: \(error)")
}
