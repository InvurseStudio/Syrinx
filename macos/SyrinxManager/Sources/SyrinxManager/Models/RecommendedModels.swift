// The models Discover suggests first: small ones of each family the engine runs. Sizes and
// checksums come from Hugging Face when one is downloaded; the sizes here are for showing
// the list before that. The Windows app's list is windows/SyrinxManager/Models/
// RecommendedModels.cs: keep the two in step.
import Foundation

struct RecommendedModel: Identifiable, Hashable {
    let title: String
    let summary: String
    let repository: String
    let fileName: String
    let approximateSize: Int64
    let architecture: String

    var id: String { "\(repository)/\(fileName)" }

    /// Whether the engine runs this architecture on the GPU today.
    var support: SupportLevel {
        EngineSupport.gpuArchitectures.contains(architecture) ? .gpu
            : EngineSupport.architectures.contains(architecture) ? .cpuOnly
            : .unsupported
    }
}

enum RecommendedModels {
    static let all: [RecommendedModel] = [
        RecommendedModel(title: "Qwen 3.5 0.8B", summary: "The smallest Qwen 3.5. Thinks before answering; quick on any Mac.",
                         repository: "unsloth/Qwen3.5-0.8B-GGUF", fileName: "Qwen3.5-0.8B-Q4_K_M.gguf",
                         approximateSize: 532_517_120, architecture: "qwen35"),
        RecommendedModel(title: "Llama 3.2 1B", summary: "Meta's small instruct model, and the first the engine ran. No thinking; quick replies.",
                         repository: "bartowski/Llama-3.2-1B-Instruct-GGUF", fileName: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                         approximateSize: 807_694_464, architecture: "llama"),
        RecommendedModel(title: "Qwen 3.5 2B", summary: "A capable small Qwen for everyday chats.",
                         repository: "unsloth/Qwen3.5-2B-GGUF", fileName: "Qwen3.5-2B-Q4_K_M.gguf",
                         approximateSize: 1_280_835_840, architecture: "qwen35"),
        RecommendedModel(title: "Qwen 3.5 4B", summary: "The strongest small Qwen 3.5; better at tools and reasoning.",
                         repository: "unsloth/Qwen3.5-4B-GGUF", fileName: "Qwen3.5-4B-Q4_K_M.gguf",
                         approximateSize: 2_740_937_888, architecture: "qwen35"),
        RecommendedModel(title: "Gemma 4 E2B", summary: "Google's quantization-aware build; strong for its size.",
                         repository: "google/gemma-4-E2B-it-qat-q4_0-gguf", fileName: "gemma-4-E2B_q4_0-it.gguf",
                         approximateSize: 3_349_516_256, architecture: "gemma4"),
        RecommendedModel(title: "Gemma 4 E4B", summary: "The larger Gemma 4 edge model, for a Mac with 16 GB.",
                         repository: "google/gemma-4-E4B-it-qat-q4_0-gguf", fileName: "gemma-4-E4B_q4_0-it.gguf",
                         approximateSize: 5_154_941_280, architecture: "gemma4"),
    ]
}
