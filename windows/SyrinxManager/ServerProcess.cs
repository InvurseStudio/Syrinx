// Starts and stops syrinx-server --standalone with the chosen model, as Vireo's Windows app
// starts it for its chats (Vireo's Windows/Vireo.Core/Engine/LocalEngine.cs). The server
// prints what it runs ("Syrinx: <model> on <GPU>, <n> tokens of context") and, once the
// model is in, where it listens ("Listening on http://127.0.0.1:N"). It accepts the token
// on its command line (--token) for its statistics, and stops when the app does: it watches
// its standard input (--with-app), which the app holds open while it runs. A Windows job
// object ends it if the app crashes, so it never holds the GPU's memory after.
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.RegularExpressions;

namespace Syrinx.Manager;

public enum ServerState
{
    Stopped,
    /// <summary>The process is up and reading the model.</summary>
    Loading,
    Running,
    Stopping,
    /// <summary>It couldn't start, or stopped on its own; <see cref="ServerProcess.Error"/> says why.</summary>
    Failed,
}

public sealed partial class ServerProcess
{
    public const string Program = "syrinx-server.exe";

    private static string PidFile => Path.Combine(Settings.Folder, "server.pid");
    private static readonly TimeSpan LoadTimeout = TimeSpan.FromMinutes(10);
    private static readonly TimeSpan StopGrace = TimeSpan.FromSeconds(5);
    private const int LogLines = 300;

    private readonly Queue<string> _log = new();
    private readonly object _gate = new();
    private readonly SynchronizationContext? _context = SynchronizationContext.Current;
    private Process? _process;

    public ServerState State { get; private set; }
    public string? Error { get; private set; }
    /// <summary>The model file being run.</summary>
    public string? Model { get; private set; }
    /// <summary>What the server says it runs on: "NVIDIA GeForce RTX 5060 Laptop GPU, 8-bit".</summary>
    public string? Device { get; private set; }
    public int? Context { get; private set; }
    /// <summary>The address it listens on, once the model is in: http://127.0.0.1:8088.</summary>
    public Uri? Address { get; private set; }
    /// <summary>The token this app started it with, for its statistics.</summary>
    public string Token { get; private set; } = "";
    /// <summary>The settings it was started with, to tell when a change needs a restart.</summary>
    public string? StartedWith { get; private set; }

    /// <summary>Raised on the UI thread when anything above changes.</summary>
    public event EventHandler? Changed;
    /// <summary>Raised on the UI thread with each line the server prints.</summary>
    public event EventHandler<string>? Output;

    public bool IsActive => State is ServerState.Loading or ServerState.Running or ServerState.Stopping;

    public string RecentOutput
    {
        get { lock (_log) return string.Join(Environment.NewLine, _log); }
    }

    /// <summary>syrinx-server: beside the app (an installed copy), or, run from a Syrinx
    /// checkout, the workspace's release build.</summary>
    public static string? FindExecutable()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, Program);
        if (File.Exists(beside)) return beside;
        if (RepositoryFolder() is { } repository)
        {
            string built = Path.Combine(repository, "target", "release", Program);
            if (File.Exists(built)) return built;
        }
        return null;
    }

    /// <summary>The Syrinx checkout the app runs from, if it does: the folder above it with
    /// the workspace's Cargo.toml.</summary>
    public static string? RepositoryFolder()
    {
        for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
        {
            if (File.Exists(Path.Combine(folder.FullName, "Cargo.toml")) && Directory.Exists(Path.Combine(folder.FullName, "crates", "syrinx-server")))
            {
                return folder.FullName;
            }
        }
        return null;
    }

    /// <summary>The command line for the settings: what <see cref="StartedWith"/> compares.</summary>
    public static List<string> Arguments(Settings settings, string token)
    {
        var args = new List<string>
        {
            "--standalone", settings.Model ?? "",
            "--port", settings.Port.ToString(CultureInfo.InvariantCulture),
            "--context", settings.Context > 0 ? settings.Context.ToString(CultureInfo.InvariantCulture) : "auto",
            "--thinking", settings.Thinking ? "on" : "off",
            "--token", token,
            "--with-app",
        };
        if (settings.Slots > 1) args.AddRange(["--slots", settings.Slots.ToString(CultureInfo.InvariantCulture)]);
        if (settings.OnNetwork) args.AddRange(["--host", "0.0.0.0"]);
        if (!string.IsNullOrWhiteSpace(settings.ApiKey)) args.AddRange(["--api-key", settings.ApiKey.Trim()]);
        return args;
    }

    /// <summary>What a restart would change: the settings' command line without the token.</summary>
    public static string Fingerprint(Settings settings) => string.Join(' ', Arguments(settings, ""));

    /// <summary>Why the settings can't start a server, if they can't.</summary>
    public static string? Problem(Settings settings)
    {
        if (string.IsNullOrEmpty(settings.Model)) return "Choose a model first.";
        if (!File.Exists(settings.Model)) return $"The model file is missing: {settings.Model}";
        if (settings.OnNetwork && string.IsNullOrWhiteSpace(settings.ApiKey))
        {
            return "Other devices can connect only with an API key. Make one under Network.";
        }
        if (FindExecutable() is null)
        {
            return $"{Program} isn't beside the app. Build it in the Syrinx folder with: cargo build --release -p syrinx-server";
        }
        return null;
    }

    public async Task StartAsync(Settings settings)
    {
        await StopAsync();
        KillLeftover();
        lock (_log) _log.Clear();
        Error = null;
        Device = null;
        Context = null;
        Address = null;
        Model = settings.Model;
        if (Problem(settings) is { } problem)
        {
            Fail(problem);
            return;
        }

        Token = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(16));
        StartedWith = Fingerprint(settings);
        string executable = FindExecutable()!;
        var info = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = Path.GetDirectoryName(executable)!,
        };
        foreach (string arg in Arguments(settings, Token)) info.ArgumentList.Add(arg);
        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, e) => OnOutput(process, e.Data);
        process.ErrorDataReceived += (_, e) => OnOutput(process, e.Data);
        process.Exited += (_, _) => Post(() => OnExited(process));
        lock (_gate) _process = process;
        State = ServerState.Loading;
        Raise();
        try
        {
            process.Start();
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            lock (_gate) _process = null;
            Fail($"Couldn't start {Program}: {e.Message}");
            return;
        }
        KillWithApp.Add(process);
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        try
        {
            Directory.CreateDirectory(Settings.Folder);
            File.WriteAllText(PidFile, process.Id.ToString(CultureInfo.InvariantCulture));
        }
        catch (IOException) { }
        _ = WatchLoadAsync(process);
    }

    /// <summary>Closing its standard input asks it to end, once it has saved the chat in
    /// use; if that takes too long, it's ended anyway.</summary>
    public async Task StopAsync()
    {
        Process? process;
        lock (_gate) process = _process;
        if (process is null) return;
        State = ServerState.Stopping;
        Raise();
        lock (_gate) _process = null;
        await Task.Run(() => EndGently(process));
        Cleanup(process);
        Address = null;
        State = ServerState.Stopped;
        Raise();
    }

    /// <summary>Ends the server as the app closes, still giving it a moment to save.</summary>
    public void StopNow()
    {
        Process? process;
        lock (_gate)
        {
            process = _process;
            _process = null;
        }
        if (process is null) return;
        EndGently(process);
        Cleanup(process);
    }

    private static void EndGently(Process process)
    {
        try
        {
            try { process.StandardInput.Close(); } catch (IOException) { }
            if (!process.WaitForExit(StopGrace)) process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            // It had already gone.
        }
    }

    [GeneratedRegex(@"Listening on (http://[^\s]+)")]
    private static partial Regex ListeningLine();

    [GeneratedRegex(@"^Syrinx: (.+) on (.+), (\d+) tokens of context$")]
    private static partial Regex ServingLine();

    private void OnOutput(Process process, string? line)
    {
        if (line is null) return;
        lock (_gate)
        {
            if (process != _process) return;
        }
        lock (_log)
        {
            _log.Enqueue(line);
            while (_log.Count > LogLines) _log.Dequeue();
        }
        Post(() =>
        {
            lock (_gate)
            {
                if (process != _process) return;
            }
            Output?.Invoke(this, line);
            if (ServingLine().Match(line) is { Success: true } serving)
            {
                Device = serving.Groups[2].Value.Trim();
                Context = int.Parse(serving.Groups[3].Value, CultureInfo.InvariantCulture);
                Raise();
            }
            if (ListeningLine().Match(line) is { Success: true } listening)
            {
                Address = new Uri(listening.Groups[1].Value);
                State = ServerState.Running;
                Raise();
            }
        });
    }

    private async Task WatchLoadAsync(Process process)
    {
        await Task.Delay(LoadTimeout);
        Post(() =>
        {
            lock (_gate)
            {
                if (process != _process) return;
            }
            if (State != ServerState.Loading) return;
            StopNow();
            Fail("The model took too long to load.");
        });
    }

    private void OnExited(Process process)
    {
        lock (_gate)
        {
            if (process != _process) return; // stopped on purpose
            _process = null;
        }
        if (State is ServerState.Stopping or ServerState.Stopped or ServerState.Failed) return;
        string last = RecentOutput.Split(Environment.NewLine).LastOrDefault(l => l.Trim().Length > 0) ?? "";
        Fail(State == ServerState.Loading
            ? $"The server stopped while loading the model. {last}".Trim()
            : $"The server stopped unexpectedly (exit code {SafeExitCode(process)}). {last}".Trim());
        Cleanup(process);
    }

    private static string SafeExitCode(Process process)
    {
        try { return process.ExitCode.ToString(CultureInfo.InvariantCulture); } catch (InvalidOperationException) { return "?"; }
    }

    private void Fail(string error)
    {
        Error = error;
        Address = null;
        State = ServerState.Failed;
        Raise();
    }

    private static void Cleanup(Process process)
    {
        process.Dispose();
        try { File.Delete(PidFile); } catch (IOException) { }
    }

    private void Raise() => Changed?.Invoke(this, EventArgs.Empty);

    private void Post(Action action)
    {
        if (_context is { } context) context.Post(_ => action(), null);
        else action();
    }

    /// <summary>A server left running by an app that crashed would hold the GPU's memory.</summary>
    private static void KillLeftover()
    {
        try
        {
            if (!int.TryParse(File.ReadAllText(PidFile), out int pid)) return;
            using var leftover = Process.GetProcessById(pid);
            if (leftover.ProcessName.Equals(Path.GetFileNameWithoutExtension(Program), StringComparison.OrdinalIgnoreCase))
            {
                leftover.Kill(entireProcessTree: true);
            }
        }
        catch (Exception e) when (e is IOException or ArgumentException or InvalidOperationException or System.ComponentModel.Win32Exception or UnauthorizedAccessException)
        {
            // Gone already.
        }
        try { File.Delete(PidFile); } catch (IOException) { }
    }
}

/// <summary>A Windows job object that ends the processes in it when the app's process ends,
/// however it ends.</summary>
internal static partial class KillWithApp
{
    private static readonly Lazy<IntPtr> Job = new(Create);

    public static void Add(Process process)
    {
        if (Job.Value == IntPtr.Zero) return;
        try { AssignProcessToJobObject(Job.Value, process.Handle); } catch (InvalidOperationException) { }
    }

    private static IntPtr Create()
    {
        IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return IntPtr.Zero;
        var info = new ExtendedLimitInformation { BasicLimitInformation = new BasicLimitInformation { LimitFlags = 0x2000 } }; // KILL_ON_JOB_CLOSE
        int size = System.Runtime.InteropServices.Marshal.SizeOf<ExtendedLimitInformation>();
        return SetInformationJobObject(job, 9, ref info, (uint)size) ? job : IntPtr.Zero; // JobObjectExtendedLimitInformation
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct BasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct ExtendedLimitInformation
    {
        public BasicLimitInformation BasicLimitInformation;
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [System.Runtime.InteropServices.LibraryImport("kernel32.dll", StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf16)]
    private static partial IntPtr CreateJobObjectW(IntPtr attributes, string? name);

    [System.Runtime.InteropServices.LibraryImport("kernel32.dll")]
    [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
    private static partial bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimitInformation info, uint length);

    [System.Runtime.InteropServices.LibraryImport("kernel32.dll")]
    [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
    private static partial bool AssignProcessToJobObject(IntPtr job, IntPtr process);
}
