import Foundation
import CryptoKit

/// Resumable, checksum-verified file download — the transport under the app's
/// first-run setup. Streams to `<dest>.part` with a rolling SHA-256 (resume keeps
/// hashing across the seam via HTTP Range), then verifies and renames into place.
public enum WeightsDownload {
    public enum FetchError: Error, CustomStringConvertible {
        case http(Int, String)
        case checksum(String)
        public var description: String {
            switch self {
            case .http(let code, let file): return "HTTP \(code) while fetching \(file)"
            case .checksum(let file): return "checksum mismatch for \(file) — partial file removed, retry"
            }
        }
    }

    /// Download `url` to `dest`. `sha256` (lowercase hex) gates the final rename when
    /// given. `progress` reports (bytesSoFar, bytesTotal); total is -1 until known.
    public static func fetch(url: URL, sha256: String?, to dest: URL,
                             progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws {
        let fm = FileManager.default
        let name = dest.lastPathComponent
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

        var req = URLRequest(url: url)
        if have > 0 { req.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }
        let (bytes, resp) = try await URLSession.shared.bytes(for: req)
        guard let http = resp as? HTTPURLResponse else { throw FetchError.http(-1, name) }
        if http.statusCode == 416 && have > 0 {
            // the .part is already the complete file (a previous run died before the
            // rename) — fall through to checksum + move
            try finalize(part: part, dest: dest, hasher: hasher, sha256: sha256)
            return
        }
        if have > 0 && http.statusCode == 200 {
            // server ignored the Range header — start over
            hasher = SHA256(); have = 0
            try? fm.removeItem(at: part); fm.createFile(atPath: part.path, contents: nil)
        }
        guard http.statusCode == 200 || http.statusCode == 206 else {
            throw FetchError.http(http.statusCode, name)
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
        try finalize(part: part, dest: dest, hasher: hasher, sha256: sha256)
    }

    private static func finalize(part: URL, dest: URL, hasher: SHA256, sha256: String?) throws {
        let fm = FileManager.default
        if let want = sha256 {
            let got = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard got == want else {
                try? fm.removeItem(at: part)
                throw FetchError.checksum(dest.lastPathComponent)
            }
        }
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: part, to: dest)
    }
}
