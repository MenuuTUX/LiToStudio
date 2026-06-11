import Foundation

/// One-call torch-checkpoint → safetensors conversion, shared by the LiToConvert CLI
/// and the app's first-run setup. Foundation-only — no Python, no MLX.
public enum CkptConverter {
    public enum ConvertError: Error, CustomStringConvertible {
        case noTensors
        public var description: String { "convert: no tensors found in state_dict" }
    }

    /// Convert `ckpt` (a torch-zip .ckpt/.pth) into a standard .safetensors file at `out`.
    /// `progress` reports (bytesWritten, bytesTotal) as tensors stream through.
    @discardableResult
    public static func convert(ckpt: URL, to out: URL,
                               progress: ((Int, Int) -> Void)? = nil) throws -> (tensors: Int, bytes: Int) {
        let zip = try TorchZip(url: ckpt)
        let (pkl, prefix) = try zip.pickle()
        let root = try TorchUnpickler(pkl).load()
        var tensors: [(String, TorchTensor)] = []
        root.stateDict.flattenTensors(into: &tensors)
        guard !tensors.isEmpty else { throw ConvertError.noTensors }
        return try Safetensors.write(tensors, zip: zip, prefix: prefix, to: out, progress: progress)
    }
}
