// The Models page: Vireo's model manager (Vireo's Windows/Vireo.WinUI/Views/
// ModelsSettingsPage.cs) for Syrinx. The model the server runs, downloads in progress, and
// two tabs: Installed (GGUF files in Syrinx's models folder and folders added, each with
// what its header says and whether Syrinx runs it) and Discover (small models that suit
// this PC, and a search of Hugging Face's GGUF repositories). The search keeps only the
// repositories whose architecture Syrinx runs, as Hugging Face read it from their files,
// and says how many it left out.
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Syrinx.Manager.Models;
using Windows.Storage.Pickers;
using Windows.System;
using WinRT.Interop;

namespace Syrinx.Manager.Views;

internal sealed class ModelsPage : Grid
{
    private const string ModelGlyph = "";
    private const string CloseGlyph = "";
    private const string CheckGlyph = "";
    private const string FolderGlyph = "";
    private const string NewFolderGlyph = "";
    private const string DeleteGlyph = "";
    private const string MoreGlyph = "";
    private const string CopyGlyph = "";

    // Hugging Face's answers, kept while the app runs so going back to the page is instant.
    private static string _lastQuery = "";
    private static IReadOnlyList<HubModel>? _lastResults;
    private static readonly Dictionary<string, IReadOnlyList<HubFile>> RepositoryFiles = new();

    /// <summary>Repositories asked for in a search, before those Syrinx can't run are left out.</summary>
    private const int SearchLimit = 60;

    private readonly Settings _settings = App.Settings;
    private readonly ServerProcess _server = App.Server;
    private readonly ModelDownloads _downloads = App.Downloads;
    private readonly StackPanel _page = new();
    private readonly StackPanel _active = new();
    private readonly StackPanel _downloadsSection = new() { Visibility = Visibility.Collapsed };
    private readonly StackPanel _downloadList = SettingsUI.Stack();
    private readonly Dictionary<string, DownloadRow> _downloadRows = [];
    private readonly SelectorBar _tabs = new() { Margin = new Thickness(-8, 20, 0, 4) };
    private readonly ContentControl _tab = new() { HorizontalContentAlignment = HorizontalAlignment.Stretch };
    private readonly StackPanel _installed = new();
    private readonly StackPanel _discover = new();
    private readonly StackPanel _results = SettingsUI.Stack();
    private readonly TextBlock _searchStatus = SettingsUI.Note("");
    // The Download/Installed control for each file shown in Discover, by download id.
    private readonly Dictionary<string, List<StackPanel>> _actionHosts = [];
    private IReadOnlyList<InstalledModel> _models = [];
    private bool _scanned;
    private InstalledModel? _activeModel;
    private bool _searching;

    public ModelsPage()
    {
        _page.Children.Add(UI.PageTitle("Models"));
        _page.Children.Add(SettingsUI.Note("Syrinx runs GGUF models: Gemma 4, Qwen 3.5 and 3.8, Llama 3 and Muse Glimmer. "
            + "Download one below, or add a folder of models you already have."));
        _page.Children.Add(SettingsUI.Header("Model the server runs"));
        _page.Children.Add(_active);

        _downloadsSection.Children.Add(SettingsUI.Header("Downloads"));
        _downloadsSection.Children.Add(_downloadList);
        _page.Children.Add(_downloadsSection);

        _tabs.Items.Add(new SelectorBarItem { Text = "Installed", Icon = new FontIcon { Glyph = "" } });
        _tabs.Items.Add(new SelectorBarItem { Text = "Discover", Icon = new FontIcon { Glyph = "" } });
        _tabs.SelectionChanged += (_, _) => ShowTab();
        _page.Children.Add(_tabs);
        _page.Children.Add(_tab);

        Children.Add(new ScrollViewer
        {
            Content = new Border { Child = _page, MaxWidth = 820, Padding = new Thickness(32, 24, 32, 32) },
        });

        BuildDiscover();
        ShowActive();
        ShowDownloads();
        ShowInstalled();
        _tabs.SelectedItem = _tabs.Items[0];

        _downloads.Changed += (_, item) => DispatcherQueue.TryEnqueue(() =>
        {
            ShowDownloads();
            RenderAction(item.Id);
        });
        _downloads.Installed += (_, item) => DispatcherQueue.TryEnqueue(async () => await OnInstalledAsync(item));
        _server.Changed += (_, _) => ShowActive();
        App.SettingsSaved += (_, _) =>
        {
            ShowActive();
            ShowInstalled();
        };
        _ = ScanAsync();
    }

    /// <summary>Opens Discover (from the Server page when there are no models).</summary>
    public void ShowDiscover() => _tabs.SelectedItem = _tabs.Items[1];

    private void ShowTab()
    {
        bool installed = _tabs.SelectedItem == _tabs.Items[0];
        _tab.Content = installed ? _installed : _discover;
    }

    private async Task OnInstalledAsync(DownloadItem item)
    {
        await ScanAsync();
        // The first model downloaded becomes the one the server runs.
        if (string.IsNullOrEmpty(_settings.Model) && _models.FirstOrDefault(m => PathsEqual(m.Path, item.Destination)) is { } model
            && model.Support.CanRun)
        {
            Choose(model.Path);
        }
        else
        {
            App.SettingsChanged();
        }
    }

    private async Task ScanAsync()
    {
        var folders = App.AddedModelFolders();
        _models = await Task.Run(() => App.Library.Scan(folders));
        _scanned = true;
        ShowActive();
        ShowInstalled();
        foreach (var id in _actionHosts.Keys.ToList()) RenderAction(id);
    }

    private void Choose(string? path)
    {
        _settings.Model = path;
        _settings.Save();
        App.SettingsChanged();
    }

    // Active model ------------------------------------------------------------------------

    private void ShowActive()
    {
        _active.Children.Clear();
        string? path = _settings.Model;
        if (string.IsNullOrEmpty(path))
        {
            _active.Children.Add(SettingsUI.Row(ModelGlyph, "No model chosen",
                "Choose one under Installed, or download one under Discover. The first model you download is chosen for you."));
            return;
        }
        if (!File.Exists(path))
        {
            _active.Children.Add(SettingsUI.Row("", "The chosen model is missing", path,
                SettingsUI.Button("Clear", () => Choose(null))));
            return;
        }
        var model = _models.FirstOrDefault(m => PathsEqual(m.Path, path))
            ?? (_activeModel is { } read && PathsEqual(read.Path, path) ? read : null);
        if (model is null)
        {
            // Its own header first: the whole scan can take seconds on a cold disk.
            _ = Task.Run(() => App.Library.Describe(path)).ContinueWith(
                t => DispatcherQueue.TryEnqueue(() =>
                {
                    _activeModel = t.Result;
                    ShowActive();
                }), TaskContinuationOptions.OnlyOnRanToCompletion);
            _active.Children.Add(SettingsUI.Row(ModelGlyph, Path.GetFileNameWithoutExtension(path), _scanned ? path : "Reading the model…"));
            return;
        }
        var extra = new StackPanel { Spacing = 8 };
        extra.Children.Add(ModelBadges(model, showLocation: false));
        extra.Children.Add(ServerStatus(path));
        var content = SettingsUI.RowContent(ModelGlyph, model.Title, Details(model), ServerButtons(path, model), extra);
        _active.Children.Add(SettingsUI.Card(content, new Thickness(16, 14, 16, 14)));
    }

    /// <summary>Whether the server is running this model.</summary>
    private UIElement ServerStatus(string path)
    {
        bool thisModel = _server.Model is { } running && PathsEqual(running, path);
        var line = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        void Text(string text, Brush brush) =>
            line.Children.Add(new TextBlock { Text = text, FontSize = 12, Foreground = brush, TextWrapping = TextWrapping.Wrap, VerticalAlignment = VerticalAlignment.Center });
        switch (_server.State)
        {
            case ServerState.Loading when thisModel:
                line.Children.Add(new ProgressRing { IsActive = true, Width = 14, Height = 14 });
                Text("Loading…", SettingsUI.TextSecondary);
                break;
            case ServerState.Running when thisModel:
                line.Children.Add(new FontIcon { Glyph = CheckGlyph, FontSize = 12, Foreground = SettingsUI.Success });
                Text($"Serving on {_server.Device ?? "this PC"} · port {_server.Address?.Port}", SettingsUI.Success);
                break;
            case ServerState.Stopping:
                Text("Stopping…", SettingsUI.TextSecondary);
                break;
            case ServerState.Failed when thisModel:
                Text(_server.Error ?? "It couldn't start.", SettingsUI.Critical);
                break;
            case ServerState.Running or ServerState.Loading:
                Text($"The server is running {Path.GetFileNameWithoutExtension(_server.Model)}. Start this one to switch.", SettingsUI.TextSecondary);
                break;
            default:
                Text("Not running. Start it to serve this model to your chat apps.", SettingsUI.TextSecondary);
                break;
        }
        return line;
    }

    private StackPanel ServerButtons(string path, InstalledModel model)
    {
        bool thisModel = _server.Model is { } running && PathsEqual(running, path);
        var buttons = SettingsUI.Buttons();
        if (thisModel && _server.State is ServerState.Loading or ServerState.Running)
        {
            if (_server.State == ServerState.Running)
            {
                buttons.Children.Add(SettingsUI.Button("Restart", () => _ = _server.StartAsync(_settings)));
            }
            buttons.Children.Add(SettingsUI.Button("Stop", () => _ = _server.StopAsync()));
        }
        else
        {
            var start = SettingsUI.Button(_server.State == ServerState.Failed && thisModel ? "Try Again" : "Start",
                () => _ = _server.StartAsync(_settings), accent: true);
            start.IsEnabled = model.Support.CanRun && _server.State != ServerState.Stopping;
            buttons.Children.Add(start);
        }
        return buttons;
    }

    // Downloads ---------------------------------------------------------------------------

    private void ShowDownloads()
    {
        var items = _downloads.Items;
        foreach (var id in _downloadRows.Keys.Except(items.Select(i => i.Id)).ToList())
        {
            _downloadList.Children.Remove(_downloadRows[id].Card);
            _downloadRows.Remove(id);
        }
        foreach (var item in items)
        {
            if (!_downloadRows.TryGetValue(item.Id, out var row))
            {
                row = new DownloadRow(item, _downloads);
                _downloadRows[item.Id] = row;
                _downloadList.Children.Add(row.Card);
            }
            row.Update();
        }
        _downloadsSection.Visibility = items.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    /// <summary>A download's card. Kept between updates, so its buttons don't flicker while
    /// the progress moves.</summary>
    private sealed class DownloadRow
    {
        private readonly DownloadItem _item;
        private readonly ModelDownloads _downloads;
        private readonly ProgressBar _progress = new() { Maximum = 1, Margin = new Thickness(0, 8, 0, 2) };
        private readonly TextBlock _status = new() { FontSize = 12, TextWrapping = TextWrapping.Wrap };
        private readonly StackPanel _buttons = SettingsUI.Buttons();
        private DownloadState? _shownState;

        public Border Card { get; }

        public DownloadRow(DownloadItem item, ModelDownloads downloads)
        {
            _item = item;
            _downloads = downloads;
            var details = new StackPanel { Children = { _progress, _status } };
            Card = SettingsUI.Card(SettingsUI.RowContent("", item.FileName, item.Repository, _buttons, details));
        }

        public void Update()
        {
            var item = _item;
            _progress.Value = item.Fraction;
            _progress.IsIndeterminate = item.State is DownloadState.Preparing or DownloadState.Verifying;
            _progress.ShowPaused = item.State == DownloadState.Paused;
            _progress.ShowError = item.State == DownloadState.Failed;
            string amount = item.Size > 0
                ? $"{ModelFit.FormatBytes(item.Received)} of {ModelFit.FormatBytes(item.Size)}"
                : ModelFit.FormatBytes(item.Received);
            _status.Foreground = item.State == DownloadState.Failed ? SettingsUI.Critical : SettingsUI.TextSecondary;
            _status.Text = item.State switch
            {
                DownloadState.Downloading => item.Speed > 0
                    ? $"{amount} · {ModelFit.FormatBytes((long)item.Speed)}/s" + (item.Remaining is { } left ? $" · {FormatDuration(left)} left" : "")
                    : $"{amount} · starting…",
                DownloadState.Preparing => item.Received > 0 ? $"Checking the {ModelFit.FormatBytes(item.Received)} already downloaded…" : "Connecting to Hugging Face…",
                DownloadState.Verifying => "Checking the file against Hugging Face's checksum…",
                DownloadState.Paused => $"Paused · {amount}",
                DownloadState.Failed => item.Error ?? "Failed",
                _ => "Done",
            };
            if (_shownState == item.State) return;
            _shownState = item.State;
            _buttons.Children.Clear();
            switch (item.State)
            {
                case DownloadState.Downloading or DownloadState.Preparing:
                    _buttons.Children.Add(SettingsUI.IconButton("", "Pause", () => _downloads.Pause(item)));
                    _buttons.Children.Add(SettingsUI.IconButton(CloseGlyph, "Cancel download", () => _downloads.Cancel(item)));
                    break;
                case DownloadState.Paused:
                    _buttons.Children.Add(SettingsUI.IconButton("", "Resume", () => _downloads.Resume(item)));
                    _buttons.Children.Add(SettingsUI.IconButton(CloseGlyph, "Cancel download", () => _downloads.Cancel(item)));
                    break;
                case DownloadState.Failed:
                    _buttons.Children.Add(SettingsUI.Button("Retry", () => _downloads.Resume(item)));
                    _buttons.Children.Add(SettingsUI.IconButton(CloseGlyph, "Remove", () => _downloads.Dismiss(item)));
                    break;
                case DownloadState.Completed:
                    _buttons.Children.Add(SettingsUI.Badge("Installed", Tone.Success, CheckGlyph));
                    _buttons.Children.Add(SettingsUI.IconButton(CloseGlyph, "Remove from the list", () => _downloads.Dismiss(item)));
                    break;
            }
        }
    }

    private static string FormatDuration(TimeSpan time) =>
        time.TotalHours >= 1 ? $"{(int)time.TotalHours} h {time.Minutes} min"
        : time.TotalMinutes >= 1 ? $"{(int)time.TotalMinutes} min"
        : $"{Math.Max(1, time.Seconds)} s";

    // Installed ---------------------------------------------------------------------------

    private void ShowInstalled()
    {
        _installed.Children.Clear();
        var chatModels = _models.Where(m => m.Info is null || m.IsChat).ToList();
        if (!_scanned)
        {
            var reading = SettingsUI.Buttons(new ProgressRing { IsActive = true, Width = 20, Height = 20 },
                new TextBlock { Text = "Reading the models on this PC…", FontSize = 13, Foreground = SettingsUI.TextSecondary, VerticalAlignment = VerticalAlignment.Center });
            reading.Spacing = 12;
            _installed.Children.Add(SettingsUI.Card(reading));
        }
        else if (chatModels.Count == 0)
        {
            var discover = SettingsUI.Button("Discover models", ShowDiscover, accent: true);
            _installed.Children.Add(SettingsUI.Row(ModelGlyph, "No models yet",
                "Download a recommended model under Discover, or add a folder that already has GGUF files.", discover));
        }
        else
        {
            var list = SettingsUI.Stack();
            foreach (var model in chatModels) list.Children.Add(InstalledCard(model));
            _installed.Children.Add(list);
        }

        _installed.Children.Add(SettingsUI.Header("Model folders"));
        var folders = SettingsUI.Stack();
        string root = App.Library.ModelsRoot;
        folders.Children.Add(SettingsUI.Row(FolderGlyph, "Syrinx's models folder", root + "\nDownloads go here.",
            SettingsUI.Button("Open", () => OpenFolder(root))));
        foreach (var folder in App.AddedModelFolders())
        {
            bool added = _settings.ModelFolders.Any(f => PathsEqual(f, folder));
            var buttons = SettingsUI.Buttons(SettingsUI.Button("Open", () => OpenFolder(folder)));
            if (added)
            {
                buttons.Children.Add(SettingsUI.IconButton(DeleteGlyph, $"Stop listing {folder}", () =>
                {
                    _settings.ModelFolders.RemoveAll(f => PathsEqual(f, folder));
                    if (_settings.Model is { } chosen && ModelLibrary.IsInside(chosen, folder)) _settings.Model = null;
                    _settings.Save();
                    App.SettingsChanged();
                    _ = ScanAsync();
                }));
            }
            string name = Path.GetFileName(folder.TrimEnd('\\')) is { Length: > 0 } n ? n : folder;
            folders.Children.Add(SettingsUI.Row(FolderGlyph, added ? name : $"{name} (this Syrinx checkout)", folder, buttons));
        }
        folders.Children.Add(SettingsUI.Row(NewFolderGlyph, "Add a folder of models",
            "List GGUF files kept elsewhere, such as LM Studio's models folder, without copying them.",
            SettingsUI.Button("Add Folder…", () => _ = AddFolderAsync())));
        _installed.Children.Add(folders);
    }

    private Border InstalledCard(InstalledModel model)
    {
        bool active = PathsEqual(model.Path, _settings.Model);
        UIElement action;
        if (active)
        {
            action = SettingsUI.Badge("In use", Tone.Accent, CheckGlyph);
        }
        else
        {
            var use = SettingsUI.Button("Use", () => Choose(model.Path));
            use.IsEnabled = model.Support.CanRun;
            if (!model.Support.CanRun) ToolTipService.SetToolTip(use, model.Support.Summary);
            action = use;
        }
        var menu = new MenuFlyout();
        var show = new MenuFlyoutItem { Text = "Show in File Explorer", Icon = new FontIcon { Glyph = FolderGlyph } };
        show.Click += (_, _) => System.Diagnostics.Process.Start("explorer.exe", $"/select,\"{model.Path}\"");
        menu.Items.Add(show);
        var copy = new MenuFlyoutItem { Text = "Copy path", Icon = new FontIcon { Glyph = CopyGlyph } };
        copy.Click += (_, _) =>
        {
            var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
            package.SetText(model.Path);
            Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
        };
        menu.Items.Add(copy);
        if (model.Downloaded)
        {
            menu.Items.Add(new MenuFlyoutSeparator());
            var delete = new MenuFlyoutItem { Text = "Delete…", Icon = new FontIcon { Glyph = DeleteGlyph } };
            delete.Click += async (_, _) => await DeleteAsync(model);
            menu.Items.Add(delete);
        }
        var more = SettingsUI.IconButton(MoreGlyph, $"More for {model.Title}", () => { });
        more.Flyout = menu;
        var content = SettingsUI.RowContent(ModelGlyph, model.Title, Details(model), SettingsUI.Buttons(action, more), ModelBadges(model, showLocation: true));
        return SettingsUI.Card(content, new Thickness(16, 14, 12, 14));
    }

    private static string Details(InstalledModel model)
    {
        if (model.Info is not { } info) return $"{model.FileName} · {ModelFit.FormatBytes(model.Size)}";
        var parts = new List<string> { EngineSupport.FamilyName(info.Architecture) };
        if (info.ParameterCount > 0) parts.Add($"{ModelFit.FormatParameters(info.ParameterCount)} parameters");
        parts.Add(info.Quantization);
        parts.Add(ModelFit.FormatBytes(model.Size));
        if (info.ContextLength is { } context) parts.Add($"{context / 1024:N0}K context");
        return string.Join(" · ", parts);
    }

    private static StackPanel ModelBadges(InstalledModel model, bool showLocation)
    {
        var support = model.Support;
        var badges = SettingsUI.Badges(SupportBadge(support.Level, support.Summary));
        if (support.CanRun) badges.Children.Add(FitBadge(model.Size, support.Level));
        if (showLocation && !model.Downloaded) badges.Children.Add(SettingsUI.Badge("Added folder"));
        return badges;
    }

    private static Border SupportBadge(SupportLevel level, string summary)
    {
        var badge = level switch
        {
            SupportLevel.Gpu => SettingsUI.Badge("Runs on GPU", Tone.Success, ""),
            SupportLevel.CpuOnly => SettingsUI.Badge("CPU only for now", Tone.Neutral, ""),
            _ => SettingsUI.Badge("Not supported yet", Tone.Critical, ""),
        };
        ToolTipService.SetToolTip(badge, summary);
        return badge;
    }

    /// <summary>Whether a model of this size fits, on the GPU if it runs there, else in memory.</summary>
    private static Border FitBadge(long size, SupportLevel level)
    {
        var budget = HardwareInfo.Budget;
        if (level == SupportLevel.CpuOnly) budget = budget with { GpuBytes = 0 };
        var fit = ModelFit.Assess(size, budget);
        var badge = fit.Level switch
        {
            FitLevel.Gpu => SettingsUI.Badge("Fits in GPU memory", Tone.Success),
            FitLevel.SystemMemory => SettingsUI.Badge(level == SupportLevel.CpuOnly ? "Fits in memory" : "Slower: exceeds GPU memory",
                level == SupportLevel.CpuOnly ? Tone.Success : Tone.Caution),
            FitLevel.TooLarge => SettingsUI.Badge("Too large for this PC", Tone.Critical),
            _ => SettingsUI.Badge(fit.Summary),
        };
        ToolTipService.SetToolTip(badge, fit.Summary);
        return badge;
    }

    private async Task DeleteAsync(InstalledModel model)
    {
        if (_server.IsActive && PathsEqual(_server.Model, model.Path))
        {
            await Error("The server is running this model. Stop it first.");
            return;
        }
        bool confirmed = await Confirm($"Delete {model.Title}?",
            $"This deletes {model.FileName} ({ModelFit.FormatBytes(model.Size)}) from this PC. You can download it again later.", "Delete", destructive: true);
        if (!confirmed) return;
        try
        {
            App.Library.Delete(model);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            await Error($"{model.FileName} couldn't be deleted: {e.Message}");
            return;
        }
        if (PathsEqual(model.Path, _settings.Model)) Choose(null);
        await ScanAsync();
    }

    private async Task AddFolderAsync()
    {
        var picker = new FolderPicker { SuggestedStartLocation = PickerLocationId.Downloads };
        picker.FileTypeFilter.Add("*");
        InitializeWithWindow.Initialize(picker, App.MainWindowHandle);
        if (await picker.PickSingleFolderAsync() is not { } picked) return;
        string folder = picked.Path;
        if (_settings.ModelFolders.Any(f => PathsEqual(f, folder)) || ModelLibrary.IsInside(folder, App.Library.ModelsRoot)) return;
        _settings.ModelFolders.Add(folder);
        _settings.Save();
        App.SettingsChanged();
        await ScanAsync();
    }

    private static void OpenFolder(string folder)
    {
        Directory.CreateDirectory(folder);
        _ = Launcher.LaunchFolderPathAsync(folder);
    }

    // Discover ----------------------------------------------------------------------------

    private void BuildDiscover()
    {
        var gpu = HardwareInfo.MainGpu;
        string machine = gpu is not null && HardwareInfo.HasCudaGpu
            ? $"{gpu.Name} ({ModelFit.FormatBytes(gpu.MemoryBytes)}) and {ModelFit.FormatBytes(HardwareInfo.SystemMemoryBytes)} of memory"
            : $"{ModelFit.FormatBytes(HardwareInfo.SystemMemoryBytes)} of memory";
        _discover.Children.Add(SettingsUI.Note($"Small models of each family Syrinx runs, chosen for this PC: {machine}."));
        var list = SettingsUI.Stack();
        list.Margin = new Thickness(0, 8, 0, 0);
        foreach (var model in RecommendedModels.All) list.Children.Add(RecommendedCard(model));
        _discover.Children.Add(list);

        _discover.Children.Add(SettingsUI.Header("Search Hugging Face"));
        _discover.Children.Add(SettingsUI.Note("Only models Syrinx runs are shown: Hugging Face reads each repository's architecture from its files."));
        var search = new TextBox
        {
            PlaceholderText = "Search GGUF models, for example \"qwen3.5\" or \"gemma 4\"",
            Text = _lastQuery,
        };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(search, "Search Hugging Face");
        var go = new Button { Content = new SymbolIcon(Symbol.Find), Padding = new Thickness(10, 7, 10, 7) };
        ToolTipService.SetToolTip(go, "Search");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(go, "Search");
        go.Click += (_, _) => _ = SearchAsync(search.Text);
        search.KeyDown += (_, e) =>
        {
            if (e.Key != VirtualKey.Enter) return;
            e.Handled = true;
            _ = SearchAsync(search.Text);
        };
        // Also searches once typing pauses.
        var typing = DispatcherQueue.CreateTimer();
        typing.Interval = TimeSpan.FromMilliseconds(700);
        typing.IsRepeating = false;
        typing.Tick += (_, _) => _ = SearchAsync(search.Text);
        search.TextChanged += (_, _) =>
        {
            typing.Stop();
            if (search.Text.Trim().Length >= 3 && search.Text.Trim() != _lastQuery) typing.Start();
        };
        var bar = new Grid { ColumnSpacing = 8, Margin = new Thickness(0, 8, 0, 4) };
        bar.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        bar.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        bar.Children.Add(search);
        Grid.SetColumn(go, 1);
        bar.Children.Add(go);
        _discover.Children.Add(bar);
        _discover.Children.Add(_searchStatus);
        _results.Margin = new Thickness(0, 8, 0, 0);
        _discover.Children.Add(_results);
        if (_lastResults is not null) ShowResults(_lastResults);
    }

    private Border RecommendedCard(RecommendedModel model)
    {
        var level = RecommendedModels.Support(model);
        var badges = SettingsUI.Badges(
            SettingsUI.Badge(ModelFit.FormatBytes(model.ApproximateSize)),
            SupportBadge(level, level == SupportLevel.Gpu ? "Runs on the GPU" : "Runs on the CPU; GPU support for this architecture is coming"),
            FitBadge(model.ApproximateSize, level));
        var action = ActionHost(model.Repository, model.FileName, () => DownloadRecommendedAsync(model));
        var content = SettingsUI.RowContent(ModelGlyph, model.Title, model.Summary, action, badges);
        return SettingsUI.Card(content, new Thickness(16, 14, 16, 14));
    }

    private async Task DownloadRecommendedAsync(RecommendedModel model)
    {
        try
        {
            var files = await FilesAsync(model.Repository);
            var file = files.FirstOrDefault(f => f.Path == model.FileName)
                ?? throw new HttpRequestException($"{model.FileName} is no longer in {model.Repository}.");
            await DownloadAsync(file);
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
        {
            await Error($"Couldn't reach Hugging Face: {e.Message}");
        }
    }

    private static async Task<IReadOnlyList<HubFile>> FilesAsync(string repository)
    {
        if (RepositoryFiles.TryGetValue(repository, out var cached)) return cached;
        var files = await App.Hub.GgufFilesAsync(repository);
        RepositoryFiles[repository] = files;
        return files;
    }

    private async Task DownloadAsync(HubFile file)
    {
        var level = ModelFit.Assess(file.Size, HardwareInfo.Budget).Level;
        if (level == FitLevel.TooLarge && !await Confirm("Download anyway?",
                $"{file.FileName} is {ModelFit.FormatBytes(file.Size)}, more than this PC can run.", "Download"))
        {
            return;
        }
        Directory.CreateDirectory(App.Library.ModelsRoot);
        var root = Path.GetPathRoot(Path.GetFullPath(App.Library.ModelsRoot))!;
        long free = new DriveInfo(root).AvailableFreeSpace;
        if (free < file.Size + (1L << 30))
        {
            await Error($"{file.FileName} needs {ModelFit.FormatBytes(file.Size)}, and {root} has "
                + $"{ModelFit.FormatBytes(free)} free. Free some space, then try again.");
            return;
        }
        _downloads.Start(file);
    }

    /// <summary>The control for one file in Discover: Download, its progress, or Installed.
    /// Redrawn when the file's download changes.</summary>
    private StackPanel ActionHost(string repository, string path, Func<Task> download)
    {
        var host = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, VerticalAlignment = VerticalAlignment.Center };
        host.Tag = (repository, path, download);
        string id = $"{repository}/{path}";
        if (!_actionHosts.TryGetValue(id, out var hosts)) _actionHosts[id] = hosts = [];
        hosts.Add(host);
        Render(host);
        return host;
    }

    private void RenderAction(string id)
    {
        if (_actionHosts.TryGetValue(id, out var hosts)) foreach (var host in hosts) Render(host);
    }

    private void Render(StackPanel host)
    {
        var (repository, path, download) = ((string, string, Func<Task>))host.Tag;
        host.Children.Clear();
        string fileName = Path.GetFileName(path);
        var installed = _models.FirstOrDefault(m => m.FileName.Equals(fileName, StringComparison.OrdinalIgnoreCase));
        var item = _downloads.Find(repository, path);
        if (installed is not null && item is not { IsActive: true })
        {
            host.Children.Add(SettingsUI.Badge("Installed", Tone.Success, CheckGlyph));
            if (!PathsEqual(installed.Path, _settings.Model) && installed.Support.CanRun)
            {
                host.Children.Add(SettingsUI.Button("Use", () => Choose(installed.Path)));
            }
            return;
        }
        if (item is { State: not DownloadState.Completed })
        {
            string state = item.State switch
            {
                DownloadState.Paused => "Paused",
                DownloadState.Failed => "Failed",
                DownloadState.Verifying => "Verifying…",
                _ => $"{item.Fraction:P0}",
            };
            host.Children.Add(new TextBlock { Text = state, FontSize = 12, Foreground = SettingsUI.TextSecondary, VerticalAlignment = VerticalAlignment.Center });
            return;
        }
        Button button = null!;
        button = SettingsUI.Button("Download", async () =>
        {
            button.IsEnabled = false;
            await download();
            button.IsEnabled = true;
        });
        host.Children.Add(button);
    }

    private async Task SearchAsync(string query)
    {
        query = query.Trim();
        if (query.Length < 2 || (query == _lastQuery && _searching)) return;
        _searching = true;
        _lastQuery = query;
        _searchStatus.Text = "Searching…";
        _results.Children.Clear();
        try
        {
            _lastResults = await App.Hub.SearchAsync(query, SearchLimit);
            ShowResults(_lastResults);
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
        {
            _searchStatus.Text = $"Couldn't reach Hugging Face: {e.Message}";
        }
        finally
        {
            _searching = false;
        }
    }

    /// <summary>The repositories Syrinx runs, and a line on those left out and why.</summary>
    private void ShowResults(IReadOnlyList<HubModel> results)
    {
        _results.Children.Clear();
        var runnable = results.Where(r => EngineSupport.RunsArchitecture(r.Architecture)).ToList();
        var left = results.Where(r => !EngineSupport.RunsArchitecture(r.Architecture)).ToList();
        string leftOut = "";
        if (left.Count > 0)
        {
            var kinds = left.Select(r => r.Architecture ?? "unknown").GroupBy(a => a).OrderByDescending(g => g.Count()).Select(g => g.Key).ToList();
            leftOut = $" {left.Count} more {(left.Count == 1 ? "isn't" : "aren't")} shown: {string.Join(", ", kinds.Take(4))}{(kinds.Count > 4 ? " and others" : "")}, which Syrinx doesn't run yet.";
        }
        _searchStatus.Text = runnable.Count == 0
            ? (results.Count == 0 ? "No GGUF models match." : "None of the matching models are ones Syrinx runs." + leftOut)
            : $"{runnable.Count} {(runnable.Count == 1 ? "repository" : "repositories")} Syrinx runs, most downloaded first. Open one to see its files." + leftOut;
        foreach (var model in runnable) _results.Children.Add(ResultExpander(model));
    }

    private Expander ResultExpander(HubModel model)
    {
        var header = new Grid { ColumnSpacing = 12, Padding = new Thickness(0, 10, 0, 10) };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var names = new StackPanel { Spacing = 2 };
        names.Children.Add(new TextBlock { Text = model.Name, FontSize = 14, Foreground = SettingsUI.TextPrimary, TextTrimming = TextTrimming.CharacterEllipsis });
        names.Children.Add(new TextBlock { Text = model.Publisher, FontSize = 12, Foreground = SettingsUI.TextSecondary });
        header.Children.Add(names);
        var stats = SettingsUI.Badges(SettingsUI.Badge(EngineSupport.FamilyName(model.Architecture), Tone.Accent),
            SettingsUI.Badge($"{FormatCount(model.Downloads)} downloads", Tone.Neutral, ""));
        stats.Margin = new Thickness(0);
        if (model.Gated) stats.Children.Add(SettingsUI.Badge("Needs a Hugging Face account", Tone.Caution, ""));
        Grid.SetColumn(stats, 1);
        header.Children.Add(stats);

        var files = new StackPanel { Spacing = 8 };
        var expander = new Expander
        {
            Header = header,
            Content = files,
            HorizontalAlignment = HorizontalAlignment.Stretch,
            HorizontalContentAlignment = HorizontalAlignment.Stretch,
        };
        bool loaded = false;
        expander.Expanding += async (_, _) =>
        {
            if (loaded) return;
            loaded = true;
            files.Children.Add(new ProgressRing { IsActive = true, Width = 20, Height = 20, HorizontalAlignment = HorizontalAlignment.Left });
            try
            {
                ShowFiles(files, model, await FilesAsync(model.Repository));
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException)
            {
                loaded = false;
                files.Children.Clear();
                files.Children.Add(SettingsUI.Note($"Couldn't load the files: {e.Message}"));
            }
        };
        return expander;
    }

    private void ShowFiles(StackPanel panel, HubModel model, IReadOnlyList<HubFile> files)
    {
        panel.Children.Clear();
        var models = files.Where(f => !f.IsVisionProjector).ToList();
        if (models.Count == 0)
        {
            panel.Children.Add(SettingsUI.Note("No GGUF model files in this repository."));
            return;
        }
        string note = EngineSupport.FamilyName(model.Architecture)
            + (model.ContextLength is { } context ? $", {context / 1024:N0}K context" : "")
            + ". Whether Syrinx runs a file's quantization is a guess from its name until it's downloaded and its header is read.";
        if (EngineSupport.SearchCaveat(model.Architecture) is { } caveat) note += $" {caveat}.";
        panel.Children.Add(SettingsUI.Note(note));
        var budget = HardwareInfo.Budget;
        // The largest quantized file that fits the GPU: past 8 bits, a bigger file is slower
        // without being noticeably better. Unquantized files only when there's nothing else.
        var candidates = models.Where(f => f.Parts == 1 && EngineSupport.QuantizationLikelySupported(f.Quantization)
            && ModelFit.Assess(f.Size, budget).Level == FitLevel.Gpu).ToList();
        var best = candidates.Where(f => f.Quantization is not ("F16" or "BF16" or "F32")).MaxBy(f => f.Size)
            ?? candidates.MinBy(f => f.Size);
        foreach (var file in models)
        {
            var row = new Grid { ColumnSpacing = 12 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var text = new StackPanel { Spacing = 2 };
            var title = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            title.Children.Add(new TextBlock { Text = file.Quantization, FontSize = 14, FontWeight = FontWeights.SemiBold, Foreground = SettingsUI.TextPrimary });
            if (file == best) title.Children.Add(SettingsUI.Badge("Suggested", Tone.Accent));
            text.Children.Add(title);
            text.Children.Add(new TextBlock { Text = file.Path, FontSize = 12, Foreground = SettingsUI.TextSecondary, TextTrimming = TextTrimming.CharacterEllipsis });
            var badges = SettingsUI.Badges(SettingsUI.Badge(ModelFit.FormatBytes(file.Size)));
            bool likely = EngineSupport.QuantizationLikelySupported(file.Quantization);
            if (file.Parts > 1) badges.Children.Add(SettingsUI.Badge($"Split in {file.Parts} files: not supported yet", Tone.Critical));
            else if (!likely) badges.Children.Add(SettingsUI.Badge($"{file.Quantization} isn't supported yet", Tone.Critical));
            else badges.Children.Add(FitBadge(file.Size, SupportLevel.Gpu));
            text.Children.Add(badges);
            row.Children.Add(text);
            // Only files Syrinx can run are offered.
            if (file.Parts == 1 && likely && !model.Gated)
            {
                var action = ActionHost(file.Repository, file.Path, () => DownloadAsync(file));
                Grid.SetColumn(action, 1);
                row.Children.Add(action);
            }
            panel.Children.Add(row);
        }
        panel.Children.Add(SettingsUI.Link("Open on Hugging Face", HuggingFaceClient.PageUrl(model.Repository)));
    }

    private static string FormatCount(long count) => count switch
    {
        >= 1_000_000 => $"{count / 1e6:0.#}M",
        >= 1_000 => $"{count / 1e3:0.#}K",
        _ => $"{count}",
    };

    private static bool PathsEqual(string? a, string? b) =>
        a is not null && b is not null && string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase);

    // Dialogs -----------------------------------------------------------------------------

    private async Task<bool> Confirm(string title, string message, string confirm, bool destructive = false)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = XamlRoot,
            Title = title,
            Content = new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap },
            PrimaryButtonText = confirm,
            CloseButtonText = "Cancel",
            DefaultButton = destructive ? ContentDialogButton.Close : ContentDialogButton.Primary,
        };
        return await dialog.ShowAsync() == ContentDialogResult.Primary;
    }

    private async Task Error(string message) => await new ContentDialog
    {
        XamlRoot = XamlRoot,
        Title = "Something went wrong",
        Content = new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap },
        CloseButtonText = "OK",
    }.ShowAsync();
}
