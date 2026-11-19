// Downloads GGUF files from Hugging Face into the models folder. Each file downloads to
// name.gguf.part, resumes with an HTTP range request after a pause or a restart of the app,
// and is checked against Hugging Face's SHA-256 before it's renamed into place: a file only
// counts as installed once it matches. Downloads in progress are listed in downloads.json,
// so they come back, paused, the next time the app starts.
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Syrinx.Manager.Models;

[JsonConverter(typeof(JsonStringEnumConverter<DownloadState>))]
public enum DownloadState
{
    Downloading,
    /// <summary>Checking the part already downloaded, before resuming.</summary>
    Preparing,
    Paused,
    /// <summary>Downloaded; comparing its SHA-256 with Hugging Face's.</summary>
    Verifying,
    Failed,
    Completed,
}

public sealed class DownloadItem
{
    public required string Repository { get; init; }
    /// <summary>The file's path in the repository.</summary>
    public required string Path { get; init; }
    public required string Destination { get; init; }
    /// <summary>The size Hugging Face gives; 0 if it gave none.</summary>
    public long Size { get; init; }
    /// <summary>Hugging Face's SHA-256 of the file, in hex; null if it gave none.</summary>
    public string? Sha256 { get; init; }

    public DownloadState State { get; set; } = DownloadState.Paused;
    public string? Error { get; set; }

    [JsonIgnore] public long Received { get; internal set; }
    /// <summary>Bytes a second, averaged over the last few seconds.</summary>
    [JsonIgnore] public double Speed { get; internal set; }
    [JsonIgnore] public string Id => $"{Repository}/{Path}";
    [JsonIgnore] public string FileName => System.IO.Path.GetFileName(Path);
    [JsonIgnore] public string PartFile => Destination + ".part";
    [JsonIgnore] public double Fraction => Size > 0 ? Math.Clamp(Received / (double)Size, 0, 1) : 0;
    [JsonIgnore] public bool IsActive => State is DownloadState.Downloading or DownloadState.Preparing or DownloadState.Verifying;

    [JsonIgnore] internal CancellationTokenSource? Cancel;
    [JsonIgnore] internal bool Discard;

    public TimeSpan? Remaining => Speed > 0 && Size > Received ? TimeSpan.FromSeconds((Size - Received) / Speed) : null;
}

public sealed class ModelDownloads
{
    public static string DefaultStateFile => System.IO.Path.Combine(
        Settings.Folder, "downloads.json");

    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };
    private static readonly TimeSpan ProgressInterval = TimeSpan.FromMilliseconds(250);

    /// <summary>How long a download may receive nothing before it fails (and can be resumed).</summary>
    public TimeSpan IdleTimeout { get; init; } = TimeSpan.FromSeconds(60);

    private readonly HttpClient _http;
    private readonly string _modelsRoot;
    private readonly string _stateFile;
    private readonly List<DownloadItem> _items = [];
    private readonly Lock _lock = new();
    private readonly Lock _saveLock = new();

    /// <summary>A download started, moved on, or changed state. Raised on a background thread.</summary>
    public event EventHandler<DownloadItem>? Changed;

    /// <summary>A file was verified and is in the models folder.</summary>
    public event EventHandler<DownloadItem>? Installed;

    public ModelDownloads(HttpClient http, string modelsRoot, string stateFile)
    {
        _http = http;
        _modelsRoot = modelsRoot;
        _stateFile = stateFile;
        Load();
    }

    public IReadOnlyList<DownloadItem> Items
    {
        get { lock (_lock) return _items.ToList(); }
    }

    /// <summary>Where a repository's file goes: Models\publisher\repository\file.gguf.</summary>
    public string DestinationFor(string repository, string path) =>
        System.IO.Path.Combine([_modelsRoot, .. repository.Split('/'), System.IO.Path.GetFileName(path)]);

    public DownloadItem? Find(string repository, string path)
    {
        lock (_lock) return _items.FirstOrDefault(i => i.Repository == repository && i.Path == path);
    }

    /// <summary>Starts (or resumes) downloading a file. Returns its download.</summary>
    public DownloadItem Start(HubFile file)
    {
        DownloadItem item;
        lock (_lock)
        {
            item = _items.FirstOrDefault(i => i.Repository == file.Repository && i.Path == file.Path)
                ?? new DownloadItem
                {
                    Repository = file.Repository,
                    Path = file.Path,
                    Destination = DestinationFor(file.Repository, file.Path),
                    Size = file.Size,
                    Sha256 = file.Sha256,
                };
            if (!_items.Contains(item)) _items.Add(item);
        }
        Resume(item);
        return item;
    }

    public void Resume(DownloadItem item)
    {
        CancellationToken token;
        lock (_lock)
        {
            if (item.IsActive) return;
            item.State = DownloadState.Preparing;
            item.Error = null;
            item.Discard = false;
            item.Cancel = new CancellationTokenSource();
            token = item.Cancel.Token;
        }
        Save();
        Raise(item);
        _ = Task.Run(() => RunAsync(item, token));
    }

    /// <summary>Stops, keeping what's downloaded so far.</summary>
    public void Pause(DownloadItem item)
    {
        lock (_lock) item.Cancel?.Cancel();
    }

    /// <summary>Stops and throws away what's downloaded.</summary>
    public void Cancel(DownloadItem item)
    {
        bool running;
        lock (_lock)
        {
            item.Discard = true;
            running = item.IsActive;
            item.Cancel?.Cancel();
        }
        if (!running) Discard(item);
    }

    /// <summary>Removes a finished or failed download from the list.</summary>
    public void Dismiss(DownloadItem item)
    {
        if (item.IsActive) return;
        if (item.State != DownloadState.Completed) DeletePart(item);
        lock (_lock) _items.Remove(item);
        Save();
        Raise(item);
    }

    /// <summary>Waits for a download to stop running: for tests.</summary>
    internal async Task WaitAsync(DownloadItem item, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (item.IsActive || item.Cancel is not null)
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException($"{item.FileName} is still {item.State}.");
            await Task.Delay(10);
        }
    }

    private async Task RunAsync(DownloadItem item, CancellationToken cancel)
    {
        try
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(item.Destination)!);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long existing = File.Exists(item.PartFile) ? new FileInfo(item.PartFile).Length : 0;
            if (item.Size > 0 && existing > item.Size)
            {
                File.Delete(item.PartFile);
                existing = 0;
            }
            if (existing > 0) await HashExistingAsync(item, hash, existing, cancel);

            item.Received = existing;
            bool complete = item.Size > 0 && existing == item.Size;
            if (!complete) complete = await DownloadRestAsync(item, hash, existing, cancel);
            if (!complete) throw new IOException("The download ended early. Resume to try again.");

            SetState(item, DownloadState.Verifying);
            string actual = Convert.ToHexStringLower(hash.GetHashAndReset());
            if (item.Sha256 is { Length: > 0 } expected && !string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
            {
                DeletePart(item);
                item.Received = 0;
                Fail(item, "The downloaded file didn't match Hugging Face's checksum, so it was deleted. Try again.");
                return;
            }
            File.Move(item.PartFile, item.Destination, overwrite: true);
            lock (_lock) item.Cancel = null;
            SetState(item, DownloadState.Completed);
            Installed?.Invoke(this, item);
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            if (item.Discard) Discard(item);
            else
            {
                lock (_lock) item.Cancel = null;
                SetState(item, DownloadState.Paused);
            }
        }
        catch (Exception e) when (e is HttpRequestException or IOException or UnauthorizedAccessException or OperationCanceledException)
        {
            Fail(item, e is OperationCanceledException ? "Hugging Face stopped responding. Resume to try again." : Describe(e));
        }
    }

    /// <summary>Feeds the part already on disk to the hash, so the whole file is checked.</summary>
    private async Task HashExistingAsync(DownloadItem item, IncrementalHash hash, long length, CancellationToken cancel)
    {
        await using var part = new FileStream(item.PartFile, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var buffer = new byte[1 << 20];
        long done = 0;
        var last = DateTime.UtcNow;
        while (done < length)
        {
            int read = await part.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, length - done)), cancel);
            if (read == 0) break;
            hash.AppendData(buffer, 0, read);
            done += read;
            if (DateTime.UtcNow - last > ProgressInterval)
            {
                item.Received = done;
                Raise(item);
                last = DateTime.UtcNow;
            }
        }
    }

    /// <returns>Whether the whole file is now in the part file.</returns>
    private async Task<bool> DownloadRestAsync(DownloadItem item, IncrementalHash hash, long existing, CancellationToken cancel)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, HuggingFaceClient.ResolveUrl(item.Repository, item.Path));
        if (existing > 0) request.Headers.Range = new RangeHeaderValue(existing, null);
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel);

        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable && existing > 0 && item.Size == 0) return true;
        if (response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        {
            throw new HttpRequestException("Hugging Face asks for an account to download this model (it's gated).");
        }
        response.EnsureSuccessStatusCode();

        bool append = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent
            && response.Content.Headers.ContentRange?.From == existing;
        if (!append && existing > 0)
        {
            // The server sent the whole file instead of the rest: start over.
            hash.GetHashAndReset();
            existing = 0;
        }
        item.Received = existing;
        SetState(item, DownloadState.Downloading);

        await using var output = new FileStream(item.PartFile, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20, useAsync: true);
        await using var input = await response.Content.ReadAsStreamAsync(cancel);
        var buffer = new byte[1 << 20];
        var window = new Queue<(DateTime Time, long Bytes)>();
        var last = DateTime.UtcNow;
        using var idle = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        while (true)
        {
            idle.CancelAfter(IdleTimeout);
            int read = await input.ReadAsync(buffer, idle.Token);
            if (read == 0) break;
            await output.WriteAsync(buffer.AsMemory(0, read), cancel);
            hash.AppendData(buffer, 0, read);
            item.Received += read;
            var now = DateTime.UtcNow;
            if (now - last > ProgressInterval)
            {
                window.Enqueue((now, item.Received));
                while (window.Count > 1 && now - window.Peek().Time > TimeSpan.FromSeconds(5)) window.Dequeue();
                var (startTime, startBytes) = window.Peek();
                double seconds = (now - startTime).TotalSeconds;
                if (seconds > 0.5) item.Speed = (item.Received - startBytes) / seconds;
                Raise(item);
                last = now;
            }
        }
        await output.FlushAsync(cancel);
        return item.Size == 0 || item.Received == item.Size;
    }

    private static string Describe(Exception e) => e switch
    {
        HttpRequestException { StatusCode: HttpStatusCode.NotFound } => "The file is no longer on Hugging Face.",
        HttpRequestException { StatusCode: { } code } => $"Hugging Face answered {(int)code} {code}. Resume to try again.",
        HttpRequestException http => http.Message.StartsWith("Hugging Face", StringComparison.Ordinal)
            ? http.Message
            : "Couldn't reach Hugging Face. Check the internet connection, then resume.",
        IOException { HResult: unchecked((int)0x80070070) } => "The disk is full. Free some space, then resume.",
        _ => e.Message,
    };

    private void Fail(DownloadItem item, string error)
    {
        lock (_lock)
        {
            item.Error = error;
            item.Cancel = null;
        }
        SetState(item, DownloadState.Failed);
    }

    private void Discard(DownloadItem item)
    {
        DeletePart(item);
        lock (_lock)
        {
            item.Cancel = null;
            item.State = DownloadState.Paused;
            item.Received = 0;
            _items.Remove(item);
        }
        Save();
        Raise(item);
    }

    private static void DeletePart(DownloadItem item)
    {
        try { File.Delete(item.PartFile); } catch (IOException) { /* in use; the next download overwrites it */ }
    }

    private void SetState(DownloadItem item, DownloadState state)
    {
        lock (_lock) item.State = state;
        Save();
        Raise(item);
    }

    private void Raise(DownloadItem item) => Changed?.Invoke(this, item);

    private void Load()
    {
        List<DownloadItem> saved = [];
        try
        {
            if (File.Exists(_stateFile)) saved = JsonSerializer.Deserialize<List<DownloadItem>>(File.ReadAllText(_stateFile), Options) ?? [];
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            // Start with none.
        }
        foreach (var item in saved)
        {
            if (item.State == DownloadState.Completed) continue;
            // Whatever was running when the app closed waits to be resumed.
            if (item.State != DownloadState.Failed) item.State = DownloadState.Paused;
            item.Received = File.Exists(item.PartFile) ? new FileInfo(item.PartFile).Length : 0;
            _items.Add(item);
        }
    }

    private void Save()
    {
        List<DownloadItem> unfinished;
        lock (_lock) unfinished = _items.Where(i => i.State != DownloadState.Completed).ToList();
        lock (_saveLock)
        {
            try
            {
                // Written beside it, then moved over it, so a crash never leaves half a file.
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(_stateFile)!);
                string temporary = _stateFile + ".tmp";
                File.WriteAllText(temporary, JsonSerializer.Serialize(unfinished, Options));
                File.Move(temporary, _stateFile, overwrite: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // Kept in memory; written again at the next change.
            }
        }
    }
}
