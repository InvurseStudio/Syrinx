using Microsoft.UI.Xaml;

namespace Syrinx.Manager;

public partial class App : Application
{
    private MainWindow? _window;

    /// <summary>The server the app runs, and its settings: one of each for the app.</summary>
    public static Settings Settings { get; } = Settings.Load();
    public static ServerProcess Server { get; private set; } = null!;

    /// <summary>For Hugging Face: searches, file lists and downloads. No timeout of its own:
    /// a download can take hours, and stalls are caught by ModelDownloads.</summary>
    public static HttpClient Http { get; } = MakeHttp();
    public static Models.HuggingFaceClient Hub { get; } = new(Http);
    /// <summary>The GGUF files on this PC, each header read once.</summary>
    public static Models.ModelLibrary Library { get; } = new(Settings.ModelsFolder);
    /// <summary>Downloads go to %LOCALAPPDATA%\Syrinx\Models\publisher\repository\.</summary>
    public static Models.ModelDownloads Downloads { get; } = new(Http, Settings.ModelsFolder, Models.ModelDownloads.DefaultStateFile);

    public static MainWindow? Window { get; private set; }

    private static HttpClient MakeHttp()
    {
        var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("Syrinx/0.1");
        return http;
    }

    /// <summary>The folders models are listed from besides Syrinx's own: those added, and
    /// this checkout's models folder when run from one.</summary>
    public static List<string> AddedModelFolders() =>
        Settings.AllModelFolders().Where(f => !string.Equals(Path.GetFullPath(f), Path.GetFullPath(Settings.ModelsFolder), StringComparison.OrdinalIgnoreCase)).ToList();

    /// <summary>Raised when a page saves the settings, so the others show them.</summary>
    public static event EventHandler? SettingsSaved;

    public static void SettingsChanged() => SettingsSaved?.Invoke(null, EventArgs.Empty);

    /// <summary>For folder pickers, which unpackaged apps must tie to a window.</summary>
    public static IntPtr MainWindowHandle { get; set; }

    public static string CrashLog => Path.Combine(Settings.Folder, "logs", "manager-crash.log");

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, e) => LogCrash("Unhandled", e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => LogCrash("Unhandled (app domain)", e.ExceptionObject as Exception);
        TaskScheduler.UnobservedTaskException += (_, e) => LogCrash("Unobserved task", e.Exception);
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        // The statistics page's WebView2: its data in Syrinx's folder (the app's own may be
        // read-only once installed), and the page's background before it draws, not white.
        Environment.SetEnvironmentVariable("WEBVIEW2_USER_DATA_FOLDER", Path.Combine(Settings.Folder, "WebView2"));
        Environment.SetEnvironmentVariable("WEBVIEW2_DEFAULT_BACKGROUND_COLOR", "FF0B1018");
        // Made on the UI thread, so its events come back there.
        Server = new ServerProcess();
        _window = new MainWindow();
        Window = _window;
        // The server ends with the app; it saves the chat in use first.
        _window.Closed += (_, _) => Server.StopNow();
        _window.Activate();
        if (Settings.StartWithApp && ServerProcess.Problem(Settings) is null) _ = Server.StartAsync(Settings);
    }

    private static void LogCrash(string kind, Exception? exception)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(CrashLog)!);
            File.AppendAllText(CrashLog, $"--- {DateTimeOffset.Now:O} {kind}\n{exception}\n\n");
        }
        catch (IOException)
        {
            // Nothing more to do while crashing.
        }
    }
}
