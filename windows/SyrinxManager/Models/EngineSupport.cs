// What Syrinx's engine (crates/syrinx-engine) runs today, so the app can say so before a
// model is chosen or downloaded, and search Hugging Face for only what it runs. Keep it in
// step with the engine: the architectures its models accept (models/llama.rs, qwen35.rs,
// gemma4.rs, which also runs Muse Glimmer) and the tensor types it computes with (quant.rs on
// the CPU, cuda/mod.rs on the GPU). Vireo's Windows app has its own copy
// (Vireo.Core/Engine/EngineSupport.cs), as do the rest of this folder's files, which came
// from there: keep the two in step.
namespace Syrinx.Manager.Models;

public enum SupportLevel
{
    /// <summary>Runs on the GPU (and the CPU).</summary>
    Gpu,
    /// <summary>Runs on the CPU only, for now.</summary>
    CpuOnly,
    /// <summary>The engine can't run it yet.</summary>
    Unsupported,
}

public sealed record SupportVerdict(SupportLevel Level, string Summary)
{
    public bool CanRun => Level != SupportLevel.Unsupported;
}

public static class EngineSupport
{
    /// <summary>GGUF architectures ("general.architecture") with a forward pass in the engine.</summary>
    public static readonly IReadOnlySet<string> Architectures = new HashSet<string> { "llama", "qwen35", "gemma4", "muse-glimmer" };

    /// <summary>Architectures whose forward pass also runs on the GPU.</summary>
    public static readonly IReadOnlySet<string> GpuArchitectures = new HashSet<string> { "llama", "qwen35", "gemma4", "muse-glimmer" };

    /// <summary>Tensor types the CPU path can read.</summary>
    public static readonly IReadOnlySet<string> TensorTypes =
        new HashSet<string> { "F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K" };

    /// <summary>Tensor types the GPU path runs: those with a kernel, and BF16, which is turned
    /// into 8-bit as it's uploaded (GpuMatrix::upload_q8_0; Gemma 4's Q6_K files have a BF16
    /// per-layer projection and run on the GPU). Vireo's copy still lists BF16 as CPU only.</summary>
    public static readonly IReadOnlySet<string> GpuTensorTypes = new HashSet<string> { "F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K" };

    /// <summary>The families' names, for the architectures Hugging Face reports.</summary>
    public static string FamilyName(string? architecture) => architecture switch
    {
        "llama" => "Llama",
        "qwen35" => "Qwen 3.5 – 3.8",
        "gemma4" => "Gemma 4",
        "muse-glimmer" => "Muse Glimmer",
        null or "" => "Unknown",
        _ => architecture,
    };

    /// <summary>Whether the engine runs a repository's models, from the architecture Hugging
    /// Face read from its GGUF files (null when it gave none).</summary>
    public static bool RunsArchitecture(string? architecture) => architecture is not null && Architectures.Contains(architecture);

    /// <summary>Tokenizers the engine reads ("tokenizer.ggml.model"), and for "gpt2" the
    /// pre-tokenizers it knows ("tokenizer.ggml.pre", llama.cpp's names: tokenizer/mod.rs and
    /// pretokenize.rs). The llama architecture is shared by many models with other
    /// tokenizers (Llama 2, Mistral); only Llama 3's is read.</summary>
    public static readonly IReadOnlySet<string> PreTokenizers = new HashSet<string>
        { "llama3", "llama-v3", "llama-bpe", "qwen2", "deepseek-r1-qwen", "qwen35", "llama4" };

    /// <summary>What a search result can't show: a llama-architecture repository may be any
    /// of several tokenizers until a file is downloaded and its header read.</summary>
    public static string? SearchCaveat(string? architecture) => architecture == "llama"
        ? "Built on Llama: Syrinx runs Llama 3 models; others on this architecture (Llama 2, Mistral) may not load"
        : null;

    public static SupportVerdict Check(GgufInfo info)
    {
        if (info.IsVisionProjector) return new(SupportLevel.Unsupported, "Vision add-on for another model; images aren't supported yet");
        if (info.SplitCount > 1) return new(SupportLevel.Unsupported, $"Split into {info.SplitCount} files, which Syrinx can't load yet");
        string arch = info.Architecture ?? "unknown";
        if (!Architectures.Contains(arch)) return new(SupportLevel.Unsupported, $"The {arch} architecture isn't supported yet");
        string tokenizer = info.Text("tokenizer.ggml.model") ?? "unknown";
        if (tokenizer == "gpt2")
        {
            string pre = info.Text("tokenizer.ggml.pre") ?? "default";
            if (!PreTokenizers.Contains(pre)) return new(SupportLevel.Unsupported, $"Its tokenizer (\"{pre}\") isn't supported yet");
        }
        else if (tokenizer != "gemma4")
        {
            return new(SupportLevel.Unsupported, $"Its tokenizer (\"{tokenizer}\") isn't supported yet");
        }
        var missing = info.TensorTypes.Where(t => !TensorTypes.Contains(t)).Order().ToList();
        if (missing.Count > 0) return new(SupportLevel.Unsupported, $"Uses {string.Join(", ", missing)}, which Syrinx can't read yet");
        if (!GpuArchitectures.Contains(arch)) return new(SupportLevel.CpuOnly, "Runs on the CPU; GPU support for this architecture is coming");
        var cpuOnlyTypes = info.TensorTypes.Where(t => !GpuTensorTypes.Contains(t)).Order().ToList();
        if (cpuOnlyTypes.Count > 0) return new(SupportLevel.CpuOnly, $"Runs on the CPU; {string.Join(", ", cpuOnlyTypes)} has no GPU kernel yet");
        return new(SupportLevel.Gpu, "Runs on the GPU");
    }

    /// <summary>A guess from a file name alone, for search results before anything is
    /// downloaded: the quantization's tensor types are known by name.</summary>
    public static bool QuantizationLikelySupported(string quantization) => quantization.ToUpperInvariant() switch
    {
        "F32" or "F16" or "BF16" or "Q4_0" or "Q8_0" or "Q6_K" => true,
        // K-quant mixes: Q4_K_M and Q5_K_M use Q4_K/Q5_K with Q6_K. The _S mixes stay in their type.
        "Q4_K_M" or "Q4_K_S" or "Q5_K_M" or "Q5_K_S" => true,
        // bartowski's _L mixes add Q8_0 embeddings and output to the same types.
        "Q4_K_L" or "Q5_K_L" or "Q6_K_L" => true,
        _ => false,
    };
}
