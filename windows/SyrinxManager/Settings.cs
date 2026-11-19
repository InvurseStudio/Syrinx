// What the manager remembers between launches, in %LOCALAPPDATA%\Syrinx\manager.json: the
// model, where models are looked for, and how the server is started.
using System.Security.Cryptography;
using System.Text.Json;

namespace Syrinx.Manager;

public sealed class Settings
{
    /// <summary>Syrinx's own folder, which the server uses on its own too (chat caches).</summary>
    public static string Folder => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Syrinx");

    public static string ModelsFolder => Path.Combine(Folder, "Models");

    private static string FilePath => Path.Combine(Folder, "manager.json");

    /// <summary>The port syrinx-server --standalone listens on by default.</summary>
    public const int DefaultPort = 8088;

    /// <summary>The context lengths offered; 0 is automatic (as much as the GPU holds).</summary>
    public static readonly int[] ContextChoices = [0, 4096, 8192, 16384, 32768, 65536, 131072];

    public string? Model { get; set; }
    public List<string> ModelFolders { get; set; } = [];
    public int Port { get; set; } = DefaultPort;
    /// <summary>Listen on every network (0.0.0.0), not only this PC. Needs an API key.</summary>
    public bool OnNetwork { get; set; }
    public string ApiKey { get; set; } = "";
    public int Context { get; set; }
    public bool Thinking { get; set; }
    public int Slots { get; set; } = 1;
    public bool StartWithApp { get; set; }

    public static Settings Load()
    {
        try
        {
            if (File.Exists(FilePath) && JsonSerializer.Deserialize<Settings>(File.ReadAllText(FilePath)) is { } loaded)
            {
                return loaded;
            }
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            // Start afresh.
        }
        return new Settings();
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Folder);
            File.WriteAllText(FilePath, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Kept for this session only.
        }
    }

    /// <summary>A new API key: 32 random characters, prefixed so it's recognisable.</summary>
    public static string NewApiKey() => "sx-" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(16));

    /// <summary>Where models are looked for: Syrinx's own folder, the folders added, and the
    /// repository's models folder when the app runs from a Syrinx checkout.</summary>
    public IEnumerable<string> AllModelFolders()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var folders = new List<string> { ModelsFolder };
        folders.AddRange(ModelFolders);
        if (ServerProcess.RepositoryFolder() is { } repository) folders.Add(Path.Combine(repository, "models"));
        foreach (string folder in folders)
        {
            if (seen.Add(Path.GetFullPath(folder))) yield return folder;
        }
    }
}

/// <summary>A GGUF model file found in one of the model folders.</summary>
public sealed record ModelFile(string Path, long Bytes)
{
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    public string Size => Bytes >= 1L << 30
        ? $"{Bytes / (double)(1L << 30):0.0} GB"
        : $"{Bytes / (double)(1L << 20):0} MB";

    /// <summary>The GGUF files in the folders (and one level of subfolders, as Hugging Face
    /// downloads are laid out), largest models last. Files the engine made for its own tests
    /// (F32 copies) are left out.</summary>
    public static List<ModelFile> Find(IEnumerable<string> folders)
    {
        var found = new Dictionary<string, ModelFile>(StringComparer.OrdinalIgnoreCase);
        foreach (string folder in folders)
        {
            if (!Directory.Exists(folder)) continue;
            try
            {
                foreach (string file in Directory.EnumerateFiles(folder, "*.gguf", new EnumerationOptions { RecurseSubdirectories = true, MaxRecursionDepth = 3, IgnoreInaccessible = true }))
                {
                    if (System.IO.Path.GetFileName(file).Contains("-F32", StringComparison.OrdinalIgnoreCase)) continue;
                    found[System.IO.Path.GetFullPath(file)] = new ModelFile(file, new FileInfo(file).Length);
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // Skip a folder that can't be read.
            }
        }
        return [.. found.Values.OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase)];
    }
}
