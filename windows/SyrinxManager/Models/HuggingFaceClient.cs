// Hugging Face's public API, for finding GGUF models: search (api/models), a repository's
// files with their sizes and SHA-256 (api/models/{repo}/tree/main), and the download address
// of a file. No account is needed; gated repositories are shown but can't be downloaded.
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Syrinx.Manager.Models;

/// <summary>A repository found by a search. <see cref="Architecture"/> is what Hugging Face
/// read from its GGUF files ("gemma4", "qwen35"…), null when it gave none: Syrinx's search
/// keeps only the ones the engine runs (<see cref="EngineSupport.RunsArchitecture"/>).</summary>
public sealed record HubModel(string Repository, long Downloads, long Likes, bool Gated, DateTimeOffset? Updated, string? Architecture = null, long? ContextLength = null)
{
    public string Publisher => Repository.Split('/')[0];
    public string Name => Repository.Split('/').Last();
}

/// <summary>A GGUF file in a repository. Split models are one entry with every part.</summary>
public sealed record HubFile(string Repository, string Path, long Size, string? Sha256, int Parts = 1)
{
    public string FileName => System.IO.Path.GetFileName(Path);
    public string Quantization => HuggingFaceClient.QuantizationOf(FileName);
    public bool IsVisionProjector => FileName.Contains("mmproj", StringComparison.OrdinalIgnoreCase);
    public Uri DownloadUrl => HuggingFaceClient.ResolveUrl(Repository, Path);
}

public sealed class HuggingFaceClient(HttpClient http)
{
    public static readonly Uri Base = new("https://huggingface.co/");

    public static Uri ResolveUrl(string repository, string path) =>
        new(Base, $"{repository}/resolve/main/{string.Join('/', path.Split('/').Select(Uri.EscapeDataString))}?download=true");

    public static Uri PageUrl(string repository) => new(Base, repository);

    /// <summary>GGUF repositories matching the words, most downloaded first, each with the
    /// architecture Hugging Face read from its GGUF files (`expand[]=gguf`; asking for one
    /// field means naming the others wanted too).</summary>
    public async Task<IReadOnlyList<HubModel>> SearchAsync(string query, int limit = 30, CancellationToken cancel = default)
    {
        string url = $"api/models?search={Uri.EscapeDataString(query.Trim())}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
            + "&expand[]=gguf&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=private&expand[]=lastModified";
        var entries = await http.GetFromJsonAsync<List<SearchEntry>>(new Uri(Base, url), cancel) ?? [];
        return entries
            .Where(e => !string.IsNullOrEmpty(e.Id) && !e.Private)
            .Select(e => new HubModel(e.Id!, e.Downloads, e.Likes, e.IsGated, e.LastModified, e.Gguf?.Architecture, e.Gguf?.ContextLength))
            .ToList();
    }

    /// <summary>The repository's GGUF files, split models joined into one entry.</summary>
    public async Task<IReadOnlyList<HubFile>> GgufFilesAsync(string repository, CancellationToken cancel = default)
    {
        var url = new Uri(Base, $"api/models/{repository}/tree/main?recursive=true");
        var entries = await http.GetFromJsonAsync<List<TreeEntry>>(url, cancel) ?? [];
        return GgufFiles(repository, entries);
    }

    internal static IReadOnlyList<HubFile> GgufFiles(string repository, IEnumerable<TreeEntry> entries)
    {
        var files = new List<HubFile>();
        var splits = new Dictionary<string, List<TreeEntry>>();
        foreach (var entry in entries)
        {
            if (entry.Type != "file" || entry.Path is null || !entry.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) continue;
            var split = SplitPart.Match(entry.Path);
            if (split.Success)
            {
                string key = entry.Path[..split.Index];
                if (!splits.TryGetValue(key, out var parts)) splits[key] = parts = [];
                parts.Add(entry);
                continue;
            }
            files.Add(new HubFile(repository, entry.Path, entry.Lfs?.Size ?? entry.Size, entry.Lfs?.Oid));
        }
        foreach (var parts in splits.Values)
        {
            var first = parts.OrderBy(p => p.Path, StringComparer.Ordinal).First();
            files.Add(new HubFile(repository, first.Path!, parts.Sum(p => p.Lfs?.Size ?? p.Size), first.Lfs?.Oid, parts.Count));
        }
        return files.OrderBy(f => f.IsVisionProjector).ThenBy(f => f.Size).ToList();
    }

    private static readonly Regex SplitPart = new(@"-\d{5}-of-\d{5}\.gguf$", RegexOptions.IgnoreCase);

    private static readonly Regex QuantizationName = new(
        @"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d)(?![A-Za-z0-9])", RegexOptions.IgnoreCase);

    /// <summary>"Llama-3.2-1B-Instruct-Q4_K_M.gguf" → "Q4_K_M"; "gemma-4-E2B_q4_0-it.gguf" → "Q4_0".</summary>
    public static string QuantizationOf(string fileName)
    {
        string stem = System.IO.Path.GetFileNameWithoutExtension(fileName).Replace('-', ' ').Replace('.', ' ');
        // Underscores join a quantization's parts (Q4_K_M) but also separate words (E2B_q4_0),
        // so match on the name with words split at underscores that precede a quantization.
        stem = Regex.Replace(stem, @"_(?=(?:I?Q\d|BF16|F16|F32)(?:_|\b))", " ", RegexOptions.IgnoreCase);
        var matches = QuantizationName.Matches(stem);
        return matches.Count > 0 ? matches[^1].Value.ToUpperInvariant() : "GGUF";
    }

    internal sealed class SearchEntry
    {
        [JsonPropertyName("id")] public string? Id { get; set; }
        [JsonPropertyName("downloads")] public long Downloads { get; set; }
        [JsonPropertyName("likes")] public long Likes { get; set; }
        [JsonPropertyName("private")] public bool Private { get; set; }
        // false, or "auto"/"manual" when access must be requested.
        [JsonPropertyName("gated")] public System.Text.Json.JsonElement Gated { get; set; }
        [JsonPropertyName("lastModified")] public DateTimeOffset? LastModified { get; set; }
        [JsonPropertyName("gguf")] public GgufSummary? Gguf { get; set; }

        public bool IsGated => Gated.ValueKind switch
        {
            System.Text.Json.JsonValueKind.String => true,
            System.Text.Json.JsonValueKind.True => true,
            _ => false,
        };
    }

    /// <summary>What Hugging Face read from a repository's GGUF files (it sends the chat
    /// template too, which is left unread).</summary>
    internal sealed class GgufSummary
    {
        [JsonPropertyName("architecture")] public string? Architecture { get; set; }
        [JsonPropertyName("context_length")] public long? ContextLength { get; set; }
    }

    internal sealed class TreeEntry
    {
        [JsonPropertyName("type")] public string? Type { get; set; }
        [JsonPropertyName("path")] public string? Path { get; set; }
        [JsonPropertyName("size")] public long Size { get; set; }
        [JsonPropertyName("lfs")] public LfsInfo? Lfs { get; set; }
    }

    internal sealed class LfsInfo
    {
        /// <summary>The file's SHA-256, in hex.</summary>
        [JsonPropertyName("oid")] public string? Oid { get; set; }
        [JsonPropertyName("size")] public long Size { get; set; }
    }
}
