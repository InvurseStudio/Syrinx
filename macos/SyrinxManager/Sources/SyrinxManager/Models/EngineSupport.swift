// What Syrinx's engine (crates/syrinx-engine) runs today, so the app can say so before a
// model is chosen or downloaded, and search Hugging Face for only what it runs. Keep it in
// step with the engine: the architectures its models accept (models/llama.rs, qwen35.rs,
// gemma4.rs, which also runs Muse Glimmer, inkling2_cuda.rs) and the tensor types it computes with (quant.rs on
// the CPU, metal/mod.rs's KINDS on the GPU; BF16 is turned into another type as it's
// uploaded, models/gpu.rs). The Windows app's copy is windows/SyrinxManager/Models/
// EngineSupport.cs: keep the two in step. Inkling 2 is the exception: its GPU model is Metal
// only (CUDA builds get a stand-in), so only this copy lists it.
import Foundation

enum SupportLevel: Equatable {
    /// Runs on the GPU (and the CPU).
    case gpu
    /// Runs on the CPU only, for now.
    case cpuOnly
    /// The engine can't run it yet.
    case unsupported
}

struct SupportVerdict: Equatable {
    let level: SupportLevel
    let summary: String

    var canRun: Bool { level != .unsupported }
}

enum EngineSupport {
    /// GGUF architectures ("general.architecture") with a forward pass in the engine.
    static let architectures: Set<String> = ["llama", "qwen35", "gemma4", "muse-glimmer", "inkling2", "inkling2_loop"]

    /// Architectures whose forward pass also runs on the GPU.
    static let gpuArchitectures: Set<String> = ["llama", "qwen35", "gemma4", "muse-glimmer", "inkling2", "inkling2_loop"]

    /// Tensor types the CPU path can read.
    static let tensorTypes: Set<String> = ["F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]

    /// Tensor types the GPU path runs: those with a Metal kernel, and BF16.
    static let gpuTensorTypes: Set<String> = ["F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]

    /// Tokenizers the engine reads ("tokenizer.ggml.model"), and for "gpt2" the
    /// pre-tokenizers it knows ("tokenizer.ggml.pre", llama.cpp's names). The llama
    /// architecture is shared by many models with other tokenizers (Llama 2, Mistral); only
    /// Llama 3's is read.
    static let preTokenizers: Set<String> = ["llama3", "llama-v3", "llama-bpe", "qwen2", "deepseek-r1-qwen", "qwen35", "llama4", "inkling2"]

    /// The families' names, for the architectures Hugging Face reports.
    static func familyName(_ architecture: String?) -> String {
        switch architecture {
        case "llama": return "Llama"
        case "qwen35": return "Qwen 3.5 – 3.8"
        case "gemma4": return "Gemma 4"
        case "muse-glimmer": return "Muse Glimmer"
        case "inkling2", "inkling2_loop": return "Inkling 2"
        case nil, "": return "Unknown"
        case let other?: return other
        }
    }

    /// Whether the engine runs a repository's models, from the architecture Hugging Face
    /// read from its GGUF files (nil when it gave none).
    static func runsArchitecture(_ architecture: String?) -> Bool {
        architecture.map(architectures.contains) ?? false
    }

    /// What a search result can't show: a llama-architecture repository may be any of
    /// several tokenizers until a file is downloaded and its header read.
    static func searchCaveat(_ architecture: String?) -> String? {
        architecture == "llama"
            ? "Built on Llama: Syrinx runs Llama 3 models; others on this architecture (Llama 2, Mistral) may not load"
            : nil
    }

    static func check(_ info: GGUFInfo) -> SupportVerdict {
        if info.isVisionProjector {
            return SupportVerdict(level: .unsupported, summary: "Vision add-on for another model; images aren't supported yet")
        }
        if info.splitCount > 1 {
            return SupportVerdict(level: .unsupported, summary: "Split into \(info.splitCount) files, which Syrinx can't load yet")
        }
        let arch = info.architecture ?? "unknown"
        if !architectures.contains(arch) {
            return SupportVerdict(level: .unsupported, summary: "The \(arch) architecture isn't supported yet")
        }
        let tokenizer = info.text("tokenizer.ggml.model") ?? "unknown"
        if tokenizer == "gpt2" {
            let pre = info.text("tokenizer.ggml.pre") ?? "default"
            if !preTokenizers.contains(pre) {
                return SupportVerdict(level: .unsupported, summary: "Its tokenizer (\"\(pre)\") isn't supported yet")
            }
        } else if tokenizer != "gemma4" {
            return SupportVerdict(level: .unsupported, summary: "Its tokenizer (\"\(tokenizer)\") isn't supported yet")
        }
        let missing = info.tensorTypes.subtracting(tensorTypes).sorted()
        if !missing.isEmpty {
            return SupportVerdict(level: .unsupported, summary: "Uses \(missing.joined(separator: ", ")), which Syrinx can't read yet")
        }
        if !gpuArchitectures.contains(arch) {
            return SupportVerdict(level: .cpuOnly, summary: "Runs on the CPU; GPU support for this architecture is coming")
        }
        let cpuOnly = info.tensorTypes.subtracting(gpuTensorTypes).sorted()
        if !cpuOnly.isEmpty {
            return SupportVerdict(level: .cpuOnly, summary: "Runs on the CPU; \(cpuOnly.joined(separator: ", ")) has no GPU kernel yet")
        }
        return SupportVerdict(level: .gpu, summary: "Runs on the GPU")
    }

    /// A guess from a file name alone, for search results before anything is downloaded:
    /// the quantization's tensor types are known by name.
    static func quantizationLikelySupported(_ quantization: String) -> Bool {
        switch quantization.uppercased() {
        case "F32", "F16", "BF16", "Q4_0", "Q8_0", "Q6_K": return true
        // K-quant mixes: Q4_K_M and Q5_K_M use Q4_K/Q5_K with Q6_K. The _S mixes stay in their type.
        case "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S": return true
        // bartowski's _L mixes add Q8_0 embeddings and output to the same types.
        case "Q4_K_L", "Q5_K_L", "Q6_K_L": return true
        default: return false
        }
    }
}
