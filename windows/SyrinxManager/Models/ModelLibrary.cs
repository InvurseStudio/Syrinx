// The GGUF models on this PC: those downloaded into the models folder and those in folders
// the user added. Each file's header is read once (and again only if the file changes), so
// the list can say what each model is and whether the engine runs it.
namespace Syrinx.Manager.Models;

public sealed record InstalledModel(string Path, long Size, DateTime Modified, GgufInfo? Info, string? Problem, bool Downloaded)
{
    public string FileName => System.IO.Path.GetFileName(Path);

    /// <summary>The model's own name from its header, or the file name when the header's
    /// says nothing (Google's Gemma 4 files are named "Hf").</summary>
    public string Title => Info?.Name is { Length: >= 4 } name ? name : System.IO.Path.GetFileNameWithoutExtension(Path);

    public ModelFamily Family => ModelFamilies.Detect(Info?.Architecture, Info?.Name ?? FileName);

    public SupportVerdict Support => Info is null
        ? new SupportVerdict(SupportLevel.Unsupported, Problem ?? "Couldn't read the file")
        : EngineSupport.Check(Info);

    /// <summary>Downloads in progress (.part) and projectors aren't models to choose.</summary>
    public bool IsChat => Info is { IsVisionProjector: false };
}

public sealed class ModelLibrary(string modelsRoot)
{
    public static string DefaultRoot => Settings.ModelsFolder;

    /// <summary>LM Studio keeps models four levels down (models\publisher\repository\file).</summary>
    private const int MaxDepth = 4;

    private readonly Dictionary<string, InstalledModel> _cache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Lock _lock = new();

    public string ModelsRoot => modelsRoot;

    /// <summary>Every GGUF file in the models folder and the added folders, newest first.
    /// Slow on a cold disk: call it off the UI thread.</summary>
    public IReadOnlyList<InstalledModel> Scan(IEnumerable<string> addedFolders)
    {
        var found = new List<InstalledModel>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (folder, downloaded) in addedFolders.Select(f => (f, false)).Prepend((modelsRoot, true)))
        {
            foreach (var file in GgufFiles(folder))
            {
                if (seen.Add(file)) found.Add(Describe(file, downloaded));
            }
        }
        return found.OrderByDescending(m => m.Modified).ToList();
    }

    public InstalledModel Describe(string file, bool downloaded = false)
    {
        var fileInfo = new FileInfo(file);
        lock (_lock)
        {
            if (_cache.TryGetValue(file, out var cached) && cached.Size == fileInfo.Length && cached.Modified == fileInfo.LastWriteTimeUtc)
            {
                return cached;
            }
        }
        GgufInfo? info = null;
        string? problem = null;
        try
        {
            info = GgufReader.Read(file);
        }
        catch (GgufFormatException e)
        {
            problem = e.Message;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            problem = $"Couldn't open the file: {e.Message}";
        }
        var model = new InstalledModel(file, fileInfo.Length, fileInfo.LastWriteTimeUtc, info, problem, downloaded);
        lock (_lock) _cache[file] = model;
        return model;
    }

    /// <summary>Deletes a downloaded model's file, and its folders if that leaves them empty.</summary>
    public void Delete(InstalledModel model)
    {
        File.Delete(model.Path);
        lock (_lock) _cache.Remove(model.Path);
        if (!IsInside(model.Path, modelsRoot)) return;
        var folder = System.IO.Path.GetDirectoryName(model.Path);
        while (folder is not null && IsInside(folder, modelsRoot) && !PathsEqual(folder, modelsRoot)
            && Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
        {
            Directory.Delete(folder);
            folder = System.IO.Path.GetDirectoryName(folder);
        }
    }

    public static bool IsInside(string path, string folder)
    {
        string full = System.IO.Path.GetFullPath(path).TrimEnd('\\', '/');
        string root = System.IO.Path.GetFullPath(folder).TrimEnd('\\', '/');
        return full.Equals(root, StringComparison.OrdinalIgnoreCase)
            || full.StartsWith(root + System.IO.Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
    }

    private static bool PathsEqual(string a, string b) =>
        System.IO.Path.GetFullPath(a).TrimEnd('\\', '/').Equals(System.IO.Path.GetFullPath(b).TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase);

    private static IEnumerable<string> GgufFiles(string folder)
    {
        if (!Directory.Exists(folder)) return [];
        var options = new EnumerationOptions { RecurseSubdirectories = true, MaxRecursionDepth = MaxDepth, IgnoreInaccessible = true };
        try
        {
            return Directory.EnumerateFiles(folder, "*.gguf", options).ToList();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }
}
