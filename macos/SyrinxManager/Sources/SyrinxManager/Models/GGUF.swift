// Reads the header of a GGUF file: its metadata and the list of its tensors, without the
// weights. Enough to say what a model is (architecture, size, quantization, context length)
// and whether Syrinx's engine runs it. The format is documented in ggml's docs/gguf.md; the
// engine's own reader is crates/syrinx-engine/src/gguf.rs. The Windows app's copy is
// windows/SyrinxManager/Models/GgufHeader.cs: keep the two in step.
import Foundation

/// ggml's tensor types, by their number in the file.
enum GGUFTensorTypes {
    private static let names: [UInt32: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
        16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
        23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
        34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    ]

    static func name(_ type: UInt32) -> String { names[type] ?? "type \(type)" }
}

/// llama.cpp's general.file_type: the quantization a file was made with.
enum GGUFFileTypes {
    private static let names: [UInt32: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
        16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
        22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
        28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
        38: "MXFP4",
    ]

    static func name(_ type: UInt32) -> String? { names[type] }
}

struct GGUFTensor: Hashable {
    let name: String
    let dimensions: [UInt64]
    let type: UInt32

    var elements: UInt64 { dimensions.reduce(1, *) }
}

/// A metadata value other than an array.
enum GGUFValue: Hashable {
    case integer(Int64)
    case unsigned(UInt64)
    case float(Double)
    case bool(Bool)
    case string(String)
}

/// What a GGUF file's header says about the model in it.
struct GGUFInfo: Hashable {
    let version: UInt32
    /// Metadata values other than arrays (which are skipped: the tokenizer's run to hundreds
    /// of thousands of entries).
    let metadata: [String: GGUFValue]
    let tensors: [GGUFTensor]

    var architecture: String? { text("general.architecture") }
    var name: String? { text("general.name") }

    /// The context length the model was trained for.
    var contextLength: Int64? { architecture.flatMap { number("\($0).context_length") } }

    /// Every weight in the file, counted from the tensors' shapes.
    var parameterCount: Int64 { Int64(clamping: tensors.reduce(UInt64(0)) { $0 &+ $1.elements }) }

    /// The tensor types the file uses, by name (F32, Q4_K…).
    var tensorTypes: Set<String> { Set(tensors.map { GGUFTensorTypes.name($0.type) }) }

    /// The quantization's usual name (Q4_K_M), from general.file_type, or else the type most
    /// of the weights use.
    var quantization: String {
        if let fileType = number("general.file_type"), let name = GGUFFileTypes.name(UInt32(clamping: fileType)) {
            return name
        }
        var weights: [UInt32: UInt64] = [:]
        for tensor in tensors { weights[tensor.type, default: 0] &+= tensor.elements }
        guard let most = weights.max(by: { $0.value < $1.value }) else { return "unknown" }
        return GGUFTensorTypes.name(most.key)
    }

    /// One of several files a model was split into (split.count above 1).
    var splitCount: Int { Int(number("split.count") ?? 1) }

    /// A vision projector (mmproj), not a model that chats on its own.
    var isVisionProjector: Bool { architecture == "clip" }

    func text(_ key: String) -> String? {
        if case .string(let value) = metadata[key] { return value }
        return nil
    }

    func number(_ key: String) -> Int64? {
        switch metadata[key] {
        case .integer(let value): return value
        case .unsigned(let value): return Int64(clamping: value)
        default: return nil
        }
    }
}

struct GGUFFormatError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum GGUFReader {
    // Sanity limits, so a damaged file can't ask for gigabytes of strings or tensors.
    private static let maxCount: UInt64 = 1_000_000
    private static let maxString: UInt64 = 64 * 1024 * 1024

    static func read(_ url: URL) throws -> GGUFInfo {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var reader = Reader(handle: handle)
        return try read(&reader)
    }

    static func read(_ data: Data) throws -> GGUFInfo {
        var reader = Reader(data: data)
        return try read(&reader)
    }

    private static func read(_ reader: inout Reader) throws -> GGUFInfo {
        do {
            guard try reader.uint32() == 0x4655_4747 else { throw GGUFFormatError(message: "Not a GGUF file.") } // "GGUF"
            let version = try reader.uint32()
            guard (2...3).contains(version) else { throw GGUFFormatError(message: "GGUF version \(version) isn't supported.") }
            let tensorCount = try count(try reader.uint64())
            let metadataCount = try count(try reader.uint64())

            var metadata: [String: GGUFValue] = [:]
            for _ in 0..<metadataCount {
                let key = try string(&reader)
                let type = try reader.uint32()
                if let value = try value(&reader, type: type) { metadata[key] = value }
            }

            var tensors: [GGUFTensor] = []
            tensors.reserveCapacity(Int(min(tensorCount, 4096)))
            for _ in 0..<tensorCount {
                let name = try string(&reader)
                let dimensions = try reader.uint32()
                guard dimensions <= 8 else { throw GGUFFormatError(message: "Tensor \(name) has \(dimensions) dimensions.") }
                var shape: [UInt64] = []
                for _ in 0..<dimensions { shape.append(try reader.uint64()) }
                let type = try reader.uint32()
                _ = try reader.uint64() // offset of its data
                tensors.append(GGUFTensor(name: name, dimensions: shape, type: type))
            }
            return GGUFInfo(version: version, metadata: metadata, tensors: tensors)
        } catch is EndOfFile {
            throw GGUFFormatError(message: "The file ends inside its header: it may be incomplete.")
        }
    }

    private static func count(_ count: UInt64) throws -> UInt64 {
        guard count <= maxCount else { throw GGUFFormatError(message: "The header claims \(count) entries.") }
        return count
    }

    private static func string(_ reader: inout Reader) throws -> String {
        let length = try reader.uint64()
        guard length <= maxString else { throw GGUFFormatError(message: "A string in the header is too long.") }
        return String(decoding: try reader.bytes(Int(length)), as: UTF8.self)
    }

    /// A metadata value; arrays are skipped and give nil.
    private static func value(_ reader: inout Reader, type: UInt32) throws -> GGUFValue? {
        switch type {
        case 0: return .unsigned(UInt64(try reader.bytes(1)[0]))
        case 1: return .integer(Int64(Int8(bitPattern: try reader.bytes(1)[0])))
        case 2: return .unsigned(UInt64(try reader.uint16()))
        case 3: return .integer(Int64(Int16(bitPattern: try reader.uint16())))
        case 4: return .unsigned(UInt64(try reader.uint32()))
        case 5: return .integer(Int64(Int32(bitPattern: try reader.uint32())))
        case 6: return .float(Double(Float(bitPattern: try reader.uint32())))
        case 7: return .bool(try reader.bytes(1)[0] != 0)
        case 8: return .string(try string(&reader))
        case 9:
            let elementType = try reader.uint32()
            let count = try reader.uint64()
            guard count <= 100 * maxCount else { throw GGUFFormatError(message: "An array in the header is too long.") }
            try skipArray(&reader, elementType: elementType, count: count)
            return nil
        case 10: return .unsigned(try reader.uint64())
        case 11: return .integer(Int64(bitPattern: try reader.uint64()))
        case 12: return .float(Double(bitPattern: try reader.uint64()))
        default: throw GGUFFormatError(message: "Unknown metadata type \(type).")
        }
    }

    private static func skipArray(_ reader: inout Reader, elementType: UInt32, count: UInt64) throws {
        let size: UInt64
        switch elementType {
        case 0, 1, 7: size = 1
        case 2, 3: size = 2
        case 4, 5, 6: size = 4
        case 10, 11, 12: size = 8
        default: size = 0
        }
        if size > 0 {
            try reader.skip(count * size)
            return
        }
        for _ in 0..<count {
            switch elementType {
            case 8:
                let length = try reader.uint64()
                guard length <= maxString else { throw GGUFFormatError(message: "A string in the header is too long.") }
                try reader.skip(length)
            case 9:
                let nested = try reader.uint32()
                try skipArray(&reader, elementType: nested, count: try reader.uint64())
            default:
                throw GGUFFormatError(message: "Unknown array type \(elementType).")
            }
        }
    }

    private struct EndOfFile: Error {}

    /// Reads from a file in 1 MB pieces, or from bytes in memory.
    private struct Reader {
        private let handle: FileHandle?
        private var buffer: Data
        private var position = 0

        init(handle: FileHandle) {
            self.handle = handle
            buffer = Data()
        }

        init(data: Data) {
            handle = nil
            buffer = data
        }

        private mutating func fill(_ needed: Int) throws {
            if buffer.count - position >= needed { return }
            guard let handle else { throw EndOfFile() }
            buffer = buffer.subdata(in: buffer.startIndex + position ..< buffer.endIndex)
            position = 0
            while buffer.count < needed {
                guard let more = try handle.read(upToCount: max(1 << 20, needed - buffer.count)), !more.isEmpty else {
                    throw EndOfFile()
                }
                buffer.append(more)
            }
        }

        mutating func bytes(_ count: Int) throws -> Data {
            try fill(count)
            let start = buffer.startIndex + position
            position += count
            return buffer.subdata(in: start ..< start + count)
        }

        mutating func uint16() throws -> UInt16 {
            try bytes(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian
        }

        mutating func uint32() throws -> UInt32 {
            try bytes(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        }

        mutating func uint64() throws -> UInt64 {
            try bytes(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        }

        mutating func skip(_ count: UInt64) throws {
            let inBuffer = UInt64(buffer.count - position)
            if count <= inBuffer {
                position += Int(count)
                return
            }
            guard let handle else { throw EndOfFile() }
            let target = try handle.offset() + (count - inBuffer)
            // Seeking past the end succeeds; reading there then finds nothing.
            let end = try handle.seekToEnd()
            guard target <= end else { throw EndOfFile() }
            try handle.seek(toOffset: target)
            buffer = Data()
            position = 0
        }
    }
}

/// Writes small GGUF files, for tests.
struct GGUFWriter {
    private var metadata: [(key: String, type: UInt32, bytes: Data)] = []
    private var tensors: [(name: String, shape: [UInt64], type: UInt32)] = []

    func adding(_ key: String, _ value: String) -> GGUFWriter { put(key, 8, Self.string(value)) }
    func adding(_ key: String, _ value: UInt32) -> GGUFWriter { put(key, 4, Self.little(value)) }
    func adding(_ key: String, _ value: UInt64) -> GGUFWriter { put(key, 10, Self.little(value)) }

    func adding(_ key: String, _ values: [String]) -> GGUFWriter {
        var bytes = Self.little(UInt32(8))
        bytes.append(Self.little(UInt64(values.count)))
        for value in values { bytes.append(Self.string(value)) }
        return put(key, 9, bytes)
    }

    func adding(_ key: String, _ values: [Int32]) -> GGUFWriter {
        var bytes = Self.little(UInt32(5))
        bytes.append(Self.little(UInt64(values.count)))
        for value in values { bytes.append(Self.little(UInt32(bitPattern: value))) }
        return put(key, 9, bytes)
    }

    func tensor(_ name: String, type: UInt32, _ shape: UInt64...) -> GGUFWriter {
        var copy = self
        copy.tensors.append((name, shape, type))
        return copy
    }

    private func put(_ key: String, _ type: UInt32, _ bytes: Data) -> GGUFWriter {
        var copy = self
        copy.metadata.append((key, type, bytes))
        return copy
    }

    func data() -> Data {
        var out = Self.little(UInt32(0x4655_4747))
        out.append(Self.little(UInt32(3)))
        out.append(Self.little(UInt64(tensors.count)))
        out.append(Self.little(UInt64(metadata.count)))
        for entry in metadata {
            out.append(Self.string(entry.key))
            out.append(Self.little(entry.type))
            out.append(entry.bytes)
        }
        var offset: UInt64 = 0
        for tensor in tensors {
            out.append(Self.string(tensor.name))
            out.append(Self.little(UInt32(tensor.shape.count)))
            for d in tensor.shape { out.append(Self.little(d)) }
            out.append(Self.little(tensor.type))
            out.append(Self.little(offset))
            offset += 32
        }
        return out
    }

    private static func little<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }

    private static func string(_ value: String) -> Data {
        var data = little(UInt64(value.utf8.count))
        data.append(Data(value.utf8))
        return data
    }
}
