// The models Settings › Models suggests first: small ones of each family the engine runs,
// that fit a laptop GPU. Sizes and checksums come from Hugging Face when the list is shown;
// the sizes here are for showing the list before that answer arrives.
namespace Syrinx.Manager.Models;

public sealed record RecommendedModel(
    string Title,
    string Summary,
    string Repository,
    string FileName,
    long ApproximateSize,
    ModelFamily Family,
    string Architecture);

public static class RecommendedModels
{
    public static IReadOnlyList<RecommendedModel> All { get; } =
    [
        new("Qwen 3.5 0.8B", "The smallest Qwen 3.5. Thinks before answering; quick on any PC.",
            "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", 532_517_120, ModelFamily.Qwen, "qwen35"),
        new("Llama 3.2 1B", "Meta's small instruct model, and the first the engine ran. No thinking; quick replies.",
            "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 807_694_464, ModelFamily.Other, "llama"),
        new("Qwen 3.5 2B", "A capable small Qwen for everyday chats.",
            "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1_280_835_840, ModelFamily.Qwen, "qwen35"),
        new("Qwen 3.5 4B", "The strongest small Qwen 3.5; better at tools and reasoning.",
            "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2_740_937_888, ModelFamily.Qwen, "qwen35"),
        new("Gemma 4 E2B", "Google's quantization-aware build; strong for its size.",
            "google/gemma-4-E2B-it-qat-q4_0-gguf", "gemma-4-E2B_q4_0-it.gguf", 3_349_516_256, ModelFamily.Gemma4, "gemma4"),
        new("Gemma 4 E4B", "The larger Gemma 4 edge model, for a GPU with 8 GB.",
            "google/gemma-4-E4B-it-qat-q4_0-gguf", "gemma-4-E4B_q4_0-it.gguf", 5_154_941_280, ModelFamily.Gemma4, "gemma4"),
    ];

    /// <summary>Whether the engine runs this architecture on the GPU today.</summary>
    public static SupportLevel Support(RecommendedModel model) =>
        EngineSupport.GpuArchitectures.Contains(model.Architecture) ? SupportLevel.Gpu
        : EngineSupport.Architectures.Contains(model.Architecture) ? SupportLevel.CpuOnly
        : SupportLevel.Unsupported;
}
