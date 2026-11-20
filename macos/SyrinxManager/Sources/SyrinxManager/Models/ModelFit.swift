// Whether a model of a given size fits this Mac: in the memory the GPU may use, in the rest
// of memory (on the CPU, slower), or not at all. A rough rule from the file size, since the
// weights are most of what a model needs; the rest is the context's cache and the engine's
// working memory. On a Mac the GPU shares the system's memory: Metal's recommended working
// set is what it may use, about two thirds to three quarters of it.
import Foundation
import Metal

enum FitLevel: Equatable {
    /// Fits in the GPU's share of memory with room for the context.
    case gpu
    /// Too big for the GPU's share, but fits in memory (slower).
    case systemMemory
    /// Too big for this Mac.
    case tooLarge
    /// The size or the Mac's memory isn't known.
    case unknown
}

struct Fit: Equatable {
    let level: FitLevel
    let summary: String
}

/// The memory this Mac has.
struct MemoryBudget: Equatable {
    var gpuBytes: Int64
    var systemBytes: Int64

    static let unknown = MemoryBudget(gpuBytes: 0, systemBytes: 0)
}

enum ModelFit {
    /// The context's cache and the engine's buffers.
    static let headroom: Int64 = 1 << 30

    /// The share of memory a model may take on the CPU while macOS and apps keep running.
    static let systemShare = 0.7

    static func assess(_ modelBytes: Int64?, _ memory: MemoryBudget) -> Fit {
        guard let size = modelBytes, size > 0 else { return Fit(level: .unknown, summary: "Size unknown") }
        let needed = size + headroom
        if memory.gpuBytes > 0, needed <= memory.gpuBytes { return Fit(level: .gpu, summary: "Fits in the memory the GPU can use") }
        if memory.systemBytes > 0, Double(needed) <= Double(memory.systemBytes) * systemShare {
            return Fit(level: .systemMemory, summary: memory.gpuBytes > 0
                ? "Too big for the GPU's share of memory; runs on the CPU, slower"
                : "Fits in memory")
        }
        if memory.gpuBytes == 0, memory.systemBytes == 0 { return Fit(level: .unknown, summary: "This Mac's memory is unknown") }
        return Fit(level: .tooLarge, summary: "Too large for this Mac")
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let value = Double(bytes)
        switch bytes {
        case (1 << 30)...: return String(format: "%.1f GB", value / Double(1 << 30))
        case (1 << 20)...: return String(format: "%.0f MB", value / Double(1 << 20))
        case (1 << 10)...: return String(format: "%.0f KB", value / Double(1 << 10))
        default: return "\(bytes) bytes"
        }
    }

    /// 1,235,814,400 → "1.2B"; from a GGUF's tensor shapes.
    static func formatParameters(_ count: Int64) -> String {
        switch count {
        case 1_000_000_000...:
            let billions = Double(count) / 1e9
            return billions.rounded() == (billions * 10).rounded() / 10 ? String(format: "%.0fB", billions) : String(format: "%.1fB", billions)
        case 1_000_000...: return String(format: "%.0fM", Double(count) / 1e6)
        default: return count.formatted()
        }
    }
}

/// What this Mac has for running models: its GPU, the memory the GPU may use, and the
/// system's memory.
enum HardwareInfo {
    private static let device = MTLCreateSystemDefaultDevice()

    static let gpuName: String = device?.name ?? "this Mac's GPU"

    static let systemMemoryBytes = Int64(clamping: ProcessInfo.processInfo.physicalMemory)

    /// Metal's recommended working set: the most the GPU should use.
    static let gpuMemoryBytes: Int64 = {
        if let device, device.recommendedMaxWorkingSetSize > 0 {
            return Int64(clamping: device.recommendedMaxWorkingSetSize)
        }
        return Int64(Double(systemMemoryBytes) * 0.75)
    }()

    static var budget: MemoryBudget { MemoryBudget(gpuBytes: gpuMemoryBytes, systemBytes: systemMemoryBytes) }
}
