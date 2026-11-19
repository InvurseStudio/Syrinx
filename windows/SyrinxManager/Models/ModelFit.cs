// Whether a model of a given size fits this PC: in the GPU's memory, in system memory, or
// not at all. A rough rule from the file size, since the weights are most of what a model
// needs; the rest is the context's cache and the engine's working memory.
namespace Syrinx.Manager.Models;

public enum FitLevel
{
    /// <summary>Fits in GPU memory with room for the context.</summary>
    Gpu,
    /// <summary>Too big for the GPU, but fits in system memory (slower).</summary>
    SystemMemory,
    /// <summary>Too big for this PC.</summary>
    TooLarge,
    /// <summary>The size or the PC's memory isn't known.</summary>
    Unknown,
}

public sealed record Fit(FitLevel Level, string Summary);

/// <summary>The memory this PC has, as Settings detected it.</summary>
public sealed record MemoryBudget(long GpuBytes, long SystemBytes)
{
    public static readonly MemoryBudget Unknown = new(0, 0);
}

public static class ModelFit
{
    /// <summary>The context's cache, buffers and the display's own share of the GPU.</summary>
    public const long Headroom = 1L << 30;

    /// <summary>The share of system memory a model may take while Windows and apps keep running.</summary>
    public const double SystemShare = 0.7;

    public static Fit Assess(long? modelBytes, MemoryBudget memory)
    {
        if (modelBytes is not { } size || size <= 0) return new(FitLevel.Unknown, "Size unknown");
        long needed = size + Headroom;
        if (memory.GpuBytes > 0 && needed <= memory.GpuBytes) return new(FitLevel.Gpu, "Fits in GPU memory");
        if (memory.SystemBytes > 0 && needed <= (long)(memory.SystemBytes * SystemShare))
        {
            return new(FitLevel.SystemMemory, memory.GpuBytes > 0 ? "Too big for the GPU; runs from system memory, slower" : "Fits in system memory");
        }
        if (memory.GpuBytes == 0 && memory.SystemBytes == 0) return new(FitLevel.Unknown, "This PC's memory is unknown");
        return new(FitLevel.TooLarge, "Too large for this PC");
    }

    public static string FormatBytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):0.0} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):0} MB",
        >= 1L << 10 => $"{bytes / (double)(1L << 10):0} KB",
        _ => $"{bytes} bytes",
    };

    /// <summary>1,235,814,400 → "1.2B"; from a GGUF's tensor shapes.</summary>
    public static string FormatParameters(long count) => count switch
    {
        >= 1_000_000_000 => $"{count / 1e9:0.#}B",
        >= 1_000_000 => $"{count / 1e6:0}M",
        _ => $"{count:N0}",
    };
}
