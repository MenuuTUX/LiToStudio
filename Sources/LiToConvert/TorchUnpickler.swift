import Foundation

/// Tensor element types we may meet in a torch checkpoint, with their safetensors
/// spelling and byte width. We preserve the on-disk dtype verbatim (no casting) —
/// both torch and safetensors store little-endian, so conversion is a byte copy.
enum TorchDType {
    case f64, f32, f16, bf16, i64, i32, i16, i8, u8, bool

    init?(storageClass name: String) {
        switch name {
        case "DoubleStorage": self = .f64
        case "FloatStorage": self = .f32
        case "HalfStorage": self = .f16
        case "BFloat16Storage": self = .bf16
        case "LongStorage": self = .i64
        case "IntStorage": self = .i32
        case "ShortStorage": self = .i16
        case "CharStorage": self = .i8
        case "ByteStorage": self = .u8
        case "BoolStorage": self = .bool
        default: return nil
        }
    }
    var itemSize: Int {
        switch self {
        case .f64, .i64: return 8
        case .f32, .i32: return 4
        case .f16, .bf16, .i16: return 2
        case .i8, .u8, .bool: return 1
        }
    }
    var safetensors: String {
        switch self {
        case .f64: return "F64"; case .f32: return "F32"; case .f16: return "F16"
        case .bf16: return "BF16"; case .i64: return "I64"; case .i32: return "I32"
        case .i16: return "I16"; case .i8: return "I8"; case .u8: return "U8"
        case .bool: return "BOOL"
        }
    }
}

final class TorchStorage { let dtype: TorchDType; let key: String; let numel: Int
    init(_ dtype: TorchDType, _ key: String, _ numel: Int) { self.dtype = dtype; self.key = key; self.numel = numel } }

/// A tensor view into a storage: shape + C-order check come from size/stride.
struct TorchTensor { let storage: TorchStorage; let offset: Int; let shape: [Int]; let stride: [Int]
    var numel: Int { shape.reduce(1, *) }
    /// True when the view is plain row-major over the storage start (the common case).
    var isContiguous: Bool {
        if offset != 0 { return false }
        var expected = 1
        for d in stride.indices.reversed() {
            if shape[d] == 1 { continue }              // size-1 dims may carry any stride
            if stride[d] != expected { return false }
            expected *= shape[d]
        }
        return true
    }
}

final class PyDict { var items: [(PyVal, PyVal)] = [] }
final class PyList { var items: [PyVal] = [] }

indirect enum PyVal {
    case none, bool(Bool), int(Int64), double(Double), str(String), bytes(Data)
    case tuple([PyVal]), list(PyList), dict(PyDict)
    case global(String, String), storage(TorchStorage), tensor(TorchTensor), object
}

/// A tiny unpickler covering the opcode subset `torch.save` produces (protocols 2–5),
/// enough to recover a `state_dict` of tensor descriptors. Storage bytes are NOT read
/// here — `persistent_load` only records (dtype, key, numel); the caller reads bytes lazily.
final class TorchUnpickler {
    enum Err: Error, CustomStringConvertible {
        case opcode(UInt8, Int), truncated, shape(String)
        var description: String {
            switch self {
            case .opcode(let b, let p): return String(format: "unpickle: unhandled opcode 0x%02x @%d", b, p)
            case .truncated: return "unpickle: truncated stream"
            case .shape(let s): return "unpickle: \(s)"
            }
        }
    }

    private let b: [UInt8]
    private var pos = 0
    private var stack: [PyVal] = []
    private var marks: [Int] = []
    private var memo: [Int: PyVal] = [:]

    init(_ data: Data) { self.b = [UInt8](data) }

    // MARK: byte helpers
    private func byte() throws -> UInt8 { guard pos < b.count else { throw Err.truncated }; defer { pos += 1 }; return b[pos] }
    private func bytes(_ n: Int) throws -> [UInt8] { guard pos + n <= b.count else { throw Err.truncated }; defer { pos += n }; return Array(b[pos..<pos + n]) }
    private func u16() throws -> Int { let a = try bytes(2); return Int(a[0]) | Int(a[1]) << 8 }
    private func u32() throws -> Int { let a = try bytes(4); return Int(a[0]) | Int(a[1]) << 8 | Int(a[2]) << 16 | Int(a[3]) << 24 }
    private func i32() throws -> Int { let v = try u32(); return v >= 0x8000_0000 ? v - 0x1_0000_0000 : v }
    private func u64() throws -> Int { let lo = try u32(); let hi = try u32(); return lo | (hi << 32) }
    private func line() throws -> String {
        var out = [UInt8]()
        while true { let c = try byte(); if c == 0x0A { break }; out.append(c) }
        return String(decoding: out, as: UTF8.self)
    }
    private func long(_ n: Int) throws -> Int64 {              // little-endian two's-complement
        if n == 0 { return 0 }
        let a = try bytes(n); var v: UInt64 = 0
        for i in 0..<n { v |= UInt64(a[i]) << (8 * i) }
        let signBit: UInt64 = 1 << (8 * n - 1)
        if n < 8 && (v & signBit) != 0 { v |= ~((signBit << 1) - 1) }   // sign-extend
        return Int64(bitPattern: v)
    }
    private func str(_ n: Int) throws -> String { String(decoding: try bytes(n), as: UTF8.self) }

    private func popMark() -> [PyVal] {
        let m = marks.removeLast()
        let slice = Array(stack[m...]); stack.removeSubrange(m...); return slice
    }
    private func asInt(_ v: PyVal) -> Int { if case .int(let i) = v { return Int(i) }; return 0 }

    /// Resolve a torch persistent-id tuple `('storage', <type>, key, location, numel)`.
    private func persistentLoad(_ pid: PyVal) throws -> PyVal {
        guard case .tuple(let t) = pid, t.count >= 5 else { throw Err.shape("bad persistent id") }
        var typeName = ""
        if case .global(_, let n) = t[1] { typeName = n }
        else if case .str(let s) = t[1] { typeName = s }
        guard let dt = TorchDType(storageClass: typeName) else { throw Err.shape("unknown storage \(typeName)") }
        var key = ""; if case .str(let s) = t[2] { key = s }
        let numel = asInt(t[4])
        return .storage(TorchStorage(dt, key, numel))
    }

    /// Apply a REDUCE: `callable(*args)` for the handful of torch/collections callables.
    private func reduce(_ callable: PyVal, _ argsV: PyVal) throws -> PyVal {
        guard case .global(let mod, let name) = callable else { return .object }
        let args: [PyVal] = { if case .tuple(let t) = argsV { return t }; return [] }()
        switch (mod, name) {
        case ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"):
            guard args.count >= 4, case .storage(let st) = args[0] else { throw Err.shape("rebuild_tensor args") }
            let offset = asInt(args[1])
            let shape: [Int] = { if case .tuple(let t) = args[2] { return t.map(asInt) }; return [] }()
            let stride: [Int] = { if case .tuple(let t) = args[3] { return t.map(asInt) }; return [] }()
            return .tensor(TorchTensor(storage: st, offset: offset, shape: shape, stride: stride))
        case ("torch._utils", "_rebuild_parameter"):
            return args.first ?? .none
        case ("collections", "OrderedDict"):
            return .dict(PyDict())
        default:
            return .object                                   // hyperparameters etc. — never read
        }
    }

    func load() throws -> PyVal {
        while true {
            let op = try byte()
            switch op {
            case 0x80: _ = try byte()                                   // PROTO
            case 0x95: _ = try bytes(8)                                 // FRAME
            case 0x2E: return stack.removeLast()                        // STOP
            case 0x28: marks.append(stack.count)                       // MARK
            case 0x4E: stack.append(.none)                             // NONE
            case 0x88: stack.append(.bool(true))                      // NEWTRUE
            case 0x89: stack.append(.bool(false))                     // NEWFALSE
            case 0x4B: stack.append(.int(Int64(try byte())))          // BININT1
            case 0x4D: stack.append(.int(Int64(try u16())))           // BININT2
            case 0x4A: stack.append(.int(Int64(try i32())))           // BININT
            case 0x8A: stack.append(.int(try long(Int(try byte()))))  // LONG1
            case 0x8B: stack.append(.int(try long(try u32())))        // LONG4
            case 0x47:                                                 // BINFLOAT (big-endian f64)
                let a = try bytes(8); var be: UInt64 = 0; for x in a { be = (be << 8) | UInt64(x) }
                stack.append(.double(Double(bitPattern: be)))
            case 0x58: stack.append(.str(try str(try u32())))         // BINUNICODE
            case 0x8C: stack.append(.str(try str(Int(try byte()))))   // SHORT_BINUNICODE
            case 0x8D: stack.append(.str(try str(try u64())))         // BINUNICODE8
            case 0x55: stack.append(.str(try str(Int(try byte()))))   // SHORT_BINSTRING
            case 0x54: stack.append(.str(try str(try u32())))         // BINSTRING
            case 0x42: stack.append(.bytes(Data(try bytes(try u32())))) // BINBYTES
            case 0x43: stack.append(.bytes(Data(try bytes(Int(try byte()))))) // SHORT_BINBYTES
            case 0x8E: stack.append(.bytes(Data(try bytes(try u64())))) // BINBYTES8
            case 0x7D: stack.append(.dict(PyDict()))                  // EMPTY_DICT
            case 0x5D: stack.append(.list(PyList()))                  // EMPTY_LIST
            case 0x29: stack.append(.tuple([]))                       // EMPTY_TUPLE
            case 0x85: stack.append(.tuple([stack.removeLast()]))     // TUPLE1
            case 0x86: let b2 = stack.removeLast(); let a2 = stack.removeLast(); stack.append(.tuple([a2, b2])) // TUPLE2
            case 0x87: let c = stack.removeLast(); let b3 = stack.removeLast(); let a3 = stack.removeLast(); stack.append(.tuple([a3, b3, c])) // TUPLE3
            case 0x74: stack.append(.tuple(popMark()))               // TUPLE
            case 0x71: memo[Int(try byte())] = stack.last!           // BINPUT
            case 0x72: memo[try u32()] = stack.last!                 // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last!                // MEMOIZE
            case 0x68: stack.append(memo[Int(try byte())]!)         // BINGET
            case 0x6A: stack.append(memo[try u32()]!)               // LONG_BINGET
            case 0x63: let m = try line(); let n = try line(); stack.append(.global(m, n)) // GLOBAL
            case 0x93: let n = stack.removeLast(); let m = stack.removeLast()              // STACK_GLOBAL
                if case .str(let ms) = m, case .str(let ns) = n { stack.append(.global(ms, ns)) } else { stack.append(.object) }
            case 0x51: let pid = stack.removeLast(); stack.append(try persistentLoad(pid)) // BINPERSID
            case 0x52: let a = stack.removeLast(); let c = stack.removeLast(); stack.append(try reduce(c, a)) // REDUCE
            case 0x81: let a = stack.removeLast(); let c = stack.removeLast(); stack.append(try newObj(c, a)) // NEWOBJ
            case 0x62: try build()                                    // BUILD
            case 0x73: let v = stack.removeLast(); let k = stack.removeLast()              // SETITEM
                if case .dict(let d) = stack.last! { d.items.append((k, v)) }
            case 0x75:                                                // SETITEMS
                let pairs = popMark()
                if case .dict(let d) = stack.last! { var i = 0; while i + 1 < pairs.count { d.items.append((pairs[i], pairs[i + 1])); i += 2 } }
            case 0x61: let v = stack.removeLast(); if case .list(let l) = stack.last! { l.items.append(v) } // APPEND
            case 0x65: let xs = popMark(); if case .list(let l) = stack.last! { l.items.append(contentsOf: xs) } // APPENDS
            case 0x30: _ = stack.popLast()                            // POP
            case 0x31: _ = popMark()                                  // POP_MARK
            default: throw Err.opcode(op, pos - 1)
            }
        }
    }

    private func newObj(_ cls: PyVal, _ args: PyVal) throws -> PyVal {
        if case .global(_, let n) = cls, n == "OrderedDict" || n == "dict" { return .dict(PyDict()) }
        return .object
    }
    private func build() throws {
        let state = stack.removeLast()
        guard let top = stack.last else { return }
        if case .dict(let d) = top {                                  // __dict__/OrderedDict update
            if case .dict(let s) = state { d.items.append(contentsOf: s.items) }
        }
        // tensors/objects: no settable state we need — leave `top` as-is
    }
}

extension PyVal {
    /// Flatten a (possibly nested) dict of tensors into `dotted.name → TorchTensor`.
    func flattenTensors(prefix: String = "", into out: inout [(String, TorchTensor)]) {
        guard case .dict(let d) = self else { return }
        for (k, v) in d.items {
            guard case .str(let key) = k else { continue }
            let name = prefix.isEmpty ? key : "\(prefix).\(key)"
            switch v {
            case .tensor(let t): out.append((name, t))
            case .dict: v.flattenTensors(prefix: name, into: &out)
            default: break
            }
        }
    }
    /// The `state_dict` sub-dict if present, else self.
    var stateDict: PyVal {
        if case .dict(let d) = self {
            for (k, v) in d.items { if case .str("state_dict") = k, case .dict = v { return v } }
        }
        return self
    }
}
