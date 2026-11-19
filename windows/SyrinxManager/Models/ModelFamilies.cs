// The model families Vireo tunes for, with their makers' recommended settings. The values
// follow server/model_profiles.py, which the Mac's server uses; keep the two in step.
using System.Text.RegularExpressions;

namespace Syrinx.Manager.Models;

public enum ModelFamily
{
    Gemma4,
    Qwen,
    Muse,
    /// <summary>Any other model (Llama, for example).</summary>
    Other,
}

/// <summary>How a family's replies are sampled, and how it thinks.</summary>
public sealed record FamilyBehaviour
{
    public double Temperature { get; init; }
    public double TopP { get; init; }
    /// <summary>0 turns top-k off.</summary>
    public int TopK { get; init; }
    public double MinP { get; init; }
    /// <summary>Taken off the likelihood of every token the reply has already used. Keeps
    /// small thinking models from repeating themselves.</summary>
    public double PresencePenalty { get; init; }
    /// <summary>Whether the model reasons before answering. Families that can't think ignore it.</summary>
    public bool Thinking { get; init; }
    /// <summary>The most tokens one step may spend thinking; null for no cap.</summary>
    public int? ThinkingBudget { get; init; }
}

public sealed record FamilyInfo(
    ModelFamily Family,
    string Key,
    string Title,
    string Source,
    FamilyBehaviour Recommended,
    bool CanThink,
    /// <summary>Muse Glimmer always reasons: "off" is its low strength, and it takes no budget.</summary>
    bool HasThinkingBudget,
    string? ThinkingNote);

public static class ModelFamilies
{
    public static readonly FamilyInfo Gemma4 = new(ModelFamily.Gemma4, "gemma4", "Gemma 4", "Google's recommended settings",
        new FamilyBehaviour { Temperature = 1.0, TopP = 0.95, TopK = 64, MinP = 0, Thinking = true, ThinkingBudget = 2000 },
        CanThink: true, HasThinkingBudget: true, ThinkingNote: null);

    // Qwen 3.5 and later ship temperature 1.0 in generation_config.json, which the Mac's
    // server reads over model_profiles.py's 0.6; a GGUF has no such file, so it's here. Their
    // model cards add a presence penalty of 1.5 for thinking, without which the small ones
    // can circle ("Wait… Actually…") until the thinking budget runs out.
    public static readonly FamilyInfo Qwen = new(ModelFamily.Qwen, "qwen", "Qwen 3.5 – 3.8", "Qwen's recommended settings for thinking",
        new FamilyBehaviour { Temperature = 1.0, TopP = 0.95, TopK = 20, MinP = 0, PresencePenalty = 1.5, Thinking = true, ThinkingBudget = 2000 },
        CanThink: true, HasThinkingBudget: true, ThinkingNote: null);

    public static readonly FamilyInfo Muse = new(ModelFamily.Muse, "muse", "Muse Glimmer", "Meta's recommended settings",
        new FamilyBehaviour { Temperature = 1.0, TopP = 0.95, TopK = 64, MinP = 0, Thinking = true, ThinkingBudget = null },
        CanThink: true, HasThinkingBudget: false,
        ThinkingNote: "Muse Glimmer always reasons. On uses its high strength, off its low one.");

    public static readonly FamilyInfo Other = new(ModelFamily.Other, "other", "Other models", "Vireo's defaults",
        new FamilyBehaviour { Temperature = 0.7, TopP = 0.9, TopK = 0, MinP = 0, Thinking = false, ThinkingBudget = null },
        CanThink: false, HasThinkingBudget: false, ThinkingNote: null);

    public static IReadOnlyList<FamilyInfo> All { get; } = [Gemma4, Qwen, Muse, Other];

    public static FamilyInfo Info(ModelFamily family) => All.First(f => f.Family == family);

    private static readonly (Regex Pattern, ModelFamily Family)[] ByArchitecture =
    [
        (new(@"^gemma4"), ModelFamily.Gemma4),
        (new(@"^qwen3"), ModelFamily.Qwen),
        (new(@"^muse[-_]?glimmer"), ModelFamily.Muse),
    ];

    private static readonly (Regex Pattern, ModelFamily Family)[] ByName =
    [
        (new(@"gemma[-_ ]?4", RegexOptions.IgnoreCase), ModelFamily.Gemma4),
        (new(@"qwen[-_ ]?3", RegexOptions.IgnoreCase), ModelFamily.Qwen),
        (new(@"muse[-_ ]?glimmer", RegexOptions.IgnoreCase), ModelFamily.Muse),
    ];

    /// <summary>The family from the GGUF architecture, or else from the name.</summary>
    public static ModelFamily Detect(string? architecture, string? name)
    {
        foreach (var (pattern, family) in ByArchitecture)
        {
            if (architecture is not null && pattern.IsMatch(architecture)) return family;
        }
        foreach (var (pattern, family) in ByName)
        {
            if (name is not null && pattern.IsMatch(name)) return family;
        }
        return ModelFamily.Other;
    }
}
