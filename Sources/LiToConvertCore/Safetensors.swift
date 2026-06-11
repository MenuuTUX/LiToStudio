import Foundation

/// Streaming **safetensors** writer. Header is computed up front from shapes+dtypes
/// (so we never need the bytes to lay out offsets), then each tensor's raw storage
/// bytes are copied straight through — keeping peak memory to one tensor at a time.
public enum Safetensors {
    enum Err: Error, CustomStringConvertible {
        case io(String), noncontig(String)
        var description: String {
            switch self {
            case .io(let s): return "safetensors: \(s)"
            case .noncontig(let s): return "safetensors: \(s)"
            }
        }
    }

    /// Write `tensors` (name, descriptor) reading storage bytes from `zip`.
    /// Returns (tensorCount, totalDataBytes). `progress` reports (bytesWritten, bytesTotal)
    /// after each tensor — the first-run setup UI feeds a progress bar from it.
    @discardableResult
    public static func write(_ tensors: [(String, TorchTensor)], zip: TorchZip, prefix: String,
                             to url: URL, progress: ((Int, Int) -> Void)? = nil) throws -> (Int, Int) {
        // 1) lay out the data section: each tensor gets a contiguous [start,end) range
        var header = [String: Any]()
        var offsets: [(String, TorchTensor, Int, Int)] = []   // name, t, start, end
        var cursor = 0
        for (name, t) in tensors {
            let nbytes = t.numel * t.storage.dtype.itemSize
            header[name] = ["dtype": t.storage.dtype.safetensors, "shape": t.shape,
                            "data_offsets": [cursor, cursor + nbytes]]
            offsets.append((name, t, cursor, cursor + nbytes))
            cursor += nbytes
        }

        // 2) header JSON (padded to an 8-byte boundary with spaces, per convention)
        var headerData = try JSONSerialization.data(withJSONObject: header,
                                                    options: [.sortedKeys, .withoutEscapingSlashes])
        while (8 + headerData.count) % 8 != 0 { headerData.append(0x20) }

        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = FileHandle(forWritingAtPath: url.path) else { throw Err.io("cannot open \(url.path)") }
        defer { try? fh.close() }

        var lenLE = UInt64(headerData.count).littleEndian
        try fh.write(contentsOf: Data(bytes: &lenLE, count: 8))
        try fh.write(contentsOf: headerData)

        // 3) stream tensor bytes in declared order
        for (name, t, _, end) in offsets {
            guard let storageData = zip.storage(prefix: prefix, key: t.storage.key) else {
                throw Err.io("missing storage data/\(t.storage.key) for '\(name)'")
            }
            try fh.write(contentsOf: materialize(t, from: storageData))
            progress?(end, cursor)
        }
        return (tensors.count, cursor)
    }

    /// Extract a tensor's bytes from its storage (contiguous fast-path; gather otherwise).
    static func materialize(_ t: TorchTensor, from storage: Data) throws -> Data {
        let isz = t.storage.dtype.itemSize
        if t.isContiguous {
            let start = t.offset * isz
            let end = start + t.numel * isz
            guard end <= storage.count else { throw Err.io("storage too small (\(storage.count) < \(end))") }
            return storage.subdata(in: start..<end)
        }
        // general strided gather → row-major contiguous output
        let n = t.numel
        var out = Data(count: n * isz)
        let shape = t.shape, stride = t.stride
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            storage.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                var idx = [Int](repeating: 0, count: shape.count)
                for lin in 0..<n {
                    var srcEl = t.offset
                    for d in shape.indices { srcEl += idx[d] * stride[d] }
                    let s = srcEl * isz, dPos = lin * isz
                    for k in 0..<isz { dst[dPos + k] = src[s + k] }
                    // increment row-major multi-index
                    var d = shape.count - 1
                    while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
                }
            }
        }
        return out
    }
}
