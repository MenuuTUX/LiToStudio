import Foundation

/// Minimal **zip64-aware** reader for PyTorch `.ckpt`/`.pth` archives.
///
/// `torch.save` emits a ZIP where every entry is **Stored** (uncompressed), so we
/// never need zlib — we just locate each entry's raw byte range. The 6.9 GB LiTo
/// checkpoint exceeds 32-bit offsets, so zip64 (EOCD64 locator/record + 0x0001
/// extra fields) is mandatory. The file is memory-mapped; slices are zero-copy views.
public struct TorchZip {
    enum ZipError: Error, CustomStringConvertible {
        case notFound(String)
        case unsupported(String)
        var description: String {
            switch self {
            case .notFound(let s): return "zip: \(s)"
            case .unsupported(let s): return "zip: unsupported — \(s)"
            }
        }
    }

    struct Entry { let dataOffset: UInt64; let size: UInt64 }

    let data: Data                 // memory-mapped archive bytes
    private(set) var entries: [String: Entry] = [:]

    public init(url: URL) throws {
        self.data = try Data(contentsOf: url, options: .alwaysMapped)
        try index()
    }

    // MARK: little-endian primitive reads at an absolute offset
    private func u16(_ o: UInt64) -> UInt32 {
        let i = Int(o)
        return UInt32(data[i]) | (UInt32(data[i + 1]) << 8)
    }
    private func u32(_ o: UInt64) -> UInt64 {
        let i = Int(o)
        return UInt64(data[i]) | (UInt64(data[i + 1]) << 8)
             | (UInt64(data[i + 2]) << 16) | (UInt64(data[i + 3]) << 24)
    }
    private func u64(_ o: UInt64) -> UInt64 {
        u32(o) | (u32(o + 4) << 32)
    }

    /// Build `entries` by walking the central directory (zip64 where signalled).
    private mutating func index() throws {
        let n = UInt64(data.count)
        guard n >= 22 else { throw ZipError.notFound("file too small") }

        // --- locate End Of Central Directory (0x06054b50), scanning back over any comment
        let eocdSig: UInt64 = 0x0605_4b50
        var eocd: UInt64? = nil
        let lo = n >= (22 + 65_536) ? n - (22 + 65_536) : 0
        var p = n - 22
        while true {
            if u32(p) == eocdSig { eocd = p; break }
            if p == lo { break }
            p -= 1
        }
        guard let eo = eocd else { throw ZipError.notFound("EOCD not found") }

        var totalEntries = UInt64(u16(eo + 10))
        var cdOffset = u32(eo + 16)

        // --- zip64 promotion when EOCD fields are saturated
        if cdOffset == 0xFFFF_FFFF || totalEntries == 0xFFFF {
            let locSig: UInt64 = 0x0706_4b50          // zip64 EOCD locator, 20 bytes before EOCD
            let loc = eo - 20
            guard loc < n, u32(loc) == locSig else {
                throw ZipError.unsupported("zip64 locator missing for a >4GB archive")
            }
            let z64 = u64(loc + 8)                     // offset of zip64 EOCD record
            let z64Sig: UInt64 = 0x0606_4b50
            guard u32(z64) == z64Sig else { throw ZipError.unsupported("bad zip64 EOCD record") }
            totalEntries = u64(z64 + 32)
            cdOffset = u64(z64 + 48)
        }

        // --- walk central directory records (0x02014b50)
        let cenSig: UInt64 = 0x0201_4b50
        var rec = cdOffset
        var count: UInt64 = 0
        while count < totalEntries {
            guard u32(rec) == cenSig else { throw ZipError.unsupported("bad central dir record @\(rec)") }
            let method = u16(rec + 10)
            let fnameLen = UInt64(u16(rec + 28))
            let extraLen = UInt64(u16(rec + 30))
            let commentLen = UInt64(u16(rec + 32))
            var compSize = u32(rec + 20)
            var localOffset = u32(rec + 42)
            let nameStart = rec + 46
            let name = String(decoding: data.subdata(in: Int(nameStart)..<Int(nameStart + fnameLen)),
                              as: UTF8.self)

            // zip64 extra (id 0x0001) supplies any field that was set to 0xFFFFFFFF.
            if compSize == 0xFFFF_FFFF || localOffset == 0xFFFF_FFFF {
                var ep = nameStart + fnameLen
                let extraEnd = ep + extraLen
                while ep + 4 <= extraEnd {
                    let id = u16(ep); let sz = UInt64(u16(ep + 2)); var f = ep + 4
                    if id == 0x0001 {
                        // order: [uncompressed][compressed][localHeaderOffset][diskStart]
                        if u32(rec + 24) == 0xFFFF_FFFF { f += 8 }            // skip uncompressed
                        if compSize == 0xFFFF_FFFF { compSize = u64(f); f += 8 }
                        if localOffset == 0xFFFF_FFFF { localOffset = u64(f); f += 8 }
                    }
                    ep += 4 + sz
                }
            }
            guard method == 0 else { throw ZipError.unsupported("entry '\(name)' is compressed (method \(method)); torch uses Stored") }

            // data starts after the *local* file header (its extra len can differ from central)
            let lfhSig: UInt64 = 0x0403_4b50
            guard u32(localOffset) == lfhSig else { throw ZipError.unsupported("bad local header for '\(name)'") }
            let lfnLen = UInt64(u16(localOffset + 26))
            let lexLen = UInt64(u16(localOffset + 28))
            let dataOffset = localOffset + 30 + lfnLen + lexLen
            entries[name] = Entry(dataOffset: dataOffset, size: compSize)

            rec = nameStart + fnameLen + extraLen + commentLen
            count += 1
        }
    }

    /// The single `*/data.pkl` entry (the pickle stream) and the archive's name prefix.
    public func pickle() throws -> (bytes: Data, prefix: String) {
        guard let name = entries.keys.first(where: { $0.hasSuffix("/data.pkl") }) ?? entries.keys.first(where: { $0 == "data.pkl" }) else {
            throw ZipError.notFound("data.pkl")
        }
        let prefix = name.hasSuffix("/data.pkl") ? String(name.dropLast("data.pkl".count)) : ""
        return (slice(name)!, prefix)
    }

    /// Zero-copy view of an entry's raw bytes (nil if absent).
    func slice(_ name: String) -> Data? {
        guard let e = entries[name] else { return nil }
        return data.subdata(in: Int(e.dataOffset)..<Int(e.dataOffset + e.size))
    }

    /// Raw bytes of storage key `k` (i.e. `<prefix>data/<k>`).
    func storage(prefix: String, key: String) -> Data? {
        slice("\(prefix)data/\(key)")
    }
}
