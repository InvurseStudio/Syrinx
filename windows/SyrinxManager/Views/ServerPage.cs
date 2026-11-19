// The Server page: the server's state with Start and Stop, the model it runs, how it's
// reached (port, the network, the API key), the model's options, and its output.
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace Syrinx.Manager.Views;

internal sealed class ServerPage : Grid
{
    private readonly Settings _settings = App.Settings;
    private readonly ServerProcess _server = App.Server;

    private readonly Microsoft.UI.Xaml.Shapes.Ellipse _dot = UI.Dot(Theme.TextSecondary, 12);
    private readonly TextBlock _state = UI.Text("Stopped", 20, bold: true);
    private readonly TextBlock _detail = UI.Caption("");
    private readonly TextBlock _error = UI.Text("", 12, Theme.Danger);
    private readonly Button _start = UI.Button("Start", Glyphs.Play, accent: true);
    private readonly Button _stop = UI.Button("Stop", Glyphs.Stop);
    private readonly InfoBar _restart = new() { IsClosable = false, Severity = InfoBarSeverity.Informational, Title = "Restart to use the new settings", Message = "The server is still running with the ones it started with." };

    private readonly ComboBox _model = new() { MinWidth = 320, MaxWidth = 420, PlaceholderText = "No models found" };
    private readonly TextBlock _folders = UI.Caption("");
    private readonly NumberBox _port = new() { Minimum = 1, Maximum = 65535, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Compact, Width = 130 };
    private readonly ToggleSwitch _network = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    private readonly TextBox _key = new() { Width = 300, PlaceholderText = "No key: anyone on this PC can use it", FontFamily = Theme.Monospace };
    private readonly ComboBox _context = new() { Width = 170 };
    private readonly ToggleSwitch _thinking = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    private readonly NumberBox _slots = new() { Minimum = 1, Maximum = 16, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Compact, Width = 130 };
    private readonly ToggleSwitch _startWithApp = new() { OnContent = "", OffContent = "", MinWidth = 0 };
    private readonly TextBox _log = new()
    {
        IsReadOnly = true,
        AcceptsReturn = true,
        TextWrapping = TextWrapping.NoWrap,
        FontFamily = Theme.Monospace,
        FontSize = 12,
        Height = 220,
    };
    private List<ModelFile> _models = [];
    private bool _loading;

    public ServerPage()
    {
        _start.Click += async (_, _) => await StartAsync();
        _stop.Click += async (_, _) => await _server.StopAsync();
        var restartButton = UI.Button("Restart", Glyphs.Restart);
        restartButton.Click += async (_, _) => await StartAsync();
        _restart.ActionButton = restartButton;

        Children.Add(UI.Page(
            UI.PageTitle("Server"),
            StatusCard(),
            _restart,
            ModelSection(),
            NetworkSection(),
            OptionsSection(),
            UI.Section("App", UI.Row("Start the server when Syrinx opens", "With the model chosen above.", _startWithApp)),
            LogSection()));

        Load();
        _server.Changed += (_, _) => ShowState();
        _server.Output += (_, line) => AppendLog(line);
        _log.Text = _server.RecentOutput;
        ShowState();
    }

    // ---- Status ----

    private Border StatusCard()
    {
        var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 12 };
        head.Children.Add(_dot);
        head.Children.Add(_state);
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, VerticalAlignment = VerticalAlignment.Center };
        buttons.Children.Add(_start);
        buttons.Children.Add(_stop);
        var top = new Grid();
        top.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        top.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        top.Children.Add(head);
        Grid.SetColumn(buttons, 1);
        top.Children.Add(buttons);
        _error.IsTextSelectionEnabled = true;
        return new Border
        {
            Background = Theme.BgSecondary,
            BorderBrush = Theme.Border,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(10),
            Padding = new Thickness(20, 18, 20, 18),
            Child = new StackPanel { Spacing = 8, Children = { top, _detail, _error } },
        };
    }

    private void ShowState()
    {
        var (color, text) = UI.Describe(_server.State);
        _dot.Fill = color;
        _state.Text = text;
        string model = Path.GetFileNameWithoutExtension(_server.Model) ?? "";
        _detail.Text = _server.State switch
        {
            ServerState.Running => $"{model} on {_server.Device ?? "this PC"}, {_server.Context:N0} tokens of context. Listening on {_server.Address?.Host}:{_server.Address?.Port}{(_settings.OnNetwork ? ", and to other devices on the network" : "")}.",
            ServerState.Loading => _server.Device is null ? $"Reading {model}…" : $"{model} on {_server.Device}…",
            ServerState.Stopped => ServerProcess.Problem(_settings) ?? "Start it to serve the model to your chat apps.",
            _ => "",
        };
        _detail.Visibility = _detail.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        _error.Text = _server.State == ServerState.Failed ? _server.Error ?? "" : "";
        _error.Visibility = _error.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        bool active = _server.IsActive;
        _start.Visibility = active ? Visibility.Collapsed : Visibility.Visible;
        _start.IsEnabled = ServerProcess.Problem(_settings) is null;
        _stop.Visibility = active ? Visibility.Visible : Visibility.Collapsed;
        _stop.IsEnabled = _server.State != ServerState.Stopping;
        _restart.IsOpen = _server.State is ServerState.Running or ServerState.Loading
            && _server.StartedWith != ServerProcess.Fingerprint(_settings);
    }

    private async Task StartAsync()
    {
        Save();
        await _server.StartAsync(_settings);
    }

    // ---- Model ----

    private StackPanel ModelSection()
    {
        var manage = UI.Button("Download and manage models…", Glyphs.Models);
        manage.Click += (_, _) => App.Window?.Show("Models");
        var refresh = UI.Button("Refresh", Glyphs.Refresh);
        refresh.Click += (_, _) => FillModels();
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        buttons.Children.Add(manage);
        buttons.Children.Add(refresh);
        App.SettingsSaved += (_, _) => { if (!_loading) FillModels(); };
        _model.SelectionChanged += (_, _) =>
        {
            if (_loading || _model.SelectedIndex < 0) return;
            _settings.Model = _models[_model.SelectedIndex].Path;
            Save();
        };
        var folders = new StackPanel { Spacing = 8, Children = { buttons, _folders } };
        return UI.Section("Model",
            UI.Row("Model", "A GGUF file: Gemma 4, Qwen 3.5 or 3.8, Llama 3, or Muse.", _model),
            folders);
    }

    private void FillModels()
    {
        _loading = true;
        _models = ModelFile.Find(_settings.AllModelFolders());
        // A chosen model outside the folders stays on the list.
        if (_settings.Model is { } chosen && File.Exists(chosen) && !_models.Any(m => string.Equals(m.Path, chosen, StringComparison.OrdinalIgnoreCase)))
        {
            _models.Insert(0, new ModelFile(chosen, new FileInfo(chosen).Length));
        }
        _model.Items.Clear();
        foreach (var model in _models) _model.Items.Add($"{model.Name}  ·  {model.Size}");
        _model.SelectedIndex = _models.FindIndex(m => string.Equals(m.Path, _settings.Model, StringComparison.OrdinalIgnoreCase));
        if (_model.SelectedIndex < 0 && _models.Count > 0 && string.IsNullOrEmpty(_settings.Model))
        {
            _model.SelectedIndex = 0;
            _settings.Model = _models[0].Path;
            _settings.Save();
        }
        _folders.Text = "Looked for in: " + string.Join("; ", _settings.AllModelFolders());
        _loading = false;
        ShowState();
    }

    // ---- Network ----

    private StackPanel NetworkSection()
    {
        var generate = UI.Button("New key", Glyphs.Key);
        generate.Click += (_, _) => { _key.Text = Settings.NewApiKey(); Save(); };
        var copy = UI.CopyButton(() => _key.Text, "Copy the key");
        var keyRow = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        keyRow.Children.Add(_key);
        keyRow.Children.Add(generate);
        keyRow.Children.Add(copy);
        _port.ValueChanged += (_, _) => Save();
        _network.Toggled += (_, _) =>
        {
            // Other devices need a key: make one if there isn't.
            if (_network.IsOn && string.IsNullOrWhiteSpace(_key.Text)) _key.Text = Settings.NewApiKey();
            Save();
        };
        _key.LostFocus += (_, _) => Save();
        return UI.Section("Network",
            UI.Row("Port", $"Chat apps connect to this port. {Settings.DefaultPort} unless you change it.", _port),
            UI.Row("Let other devices on this network connect", "Phones, laptops and other PCs on your Wi-Fi or network. They need the API key.", _network),
            UI.Row("API key", "Apps give it as their API key. Without one, any app on this PC can use the server.", keyRow));
    }

    // ---- Options ----

    private StackPanel OptionsSection()
    {
        foreach (int choice in Settings.ContextChoices)
        {
            _context.Items.Add(choice == 0 ? "Automatic" : $"{choice / 1024}K tokens");
        }
        _context.SelectionChanged += (_, _) => Save();
        _thinking.Toggled += (_, _) => Save();
        _slots.ValueChanged += (_, _) => Save();
        _startWithApp.Toggled += (_, _) => Save();
        return UI.Section("Model options",
            UI.Row("Context length", "How much of a chat the model keeps. Automatic is as much as the GPU's memory holds, up to the model's limit.", _context),
            UI.Row("Think before answering", "For requests that don't say. Gemma 4, Qwen and Muse think first; apps get the thought apart from the answer.", _thinking),
            UI.Row("Chats at once", "Replies written together, each in a slot of its own on the GPU. Gemma 4 models only.", _slots));
    }

    // ---- Log ----

    private StackPanel LogSection()
    {
        var copy = UI.Button("Copy", Glyphs.Copy);
        copy.Click += (_, _) =>
        {
            var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
            package.SetText(_log.Text);
            Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
        };
        var head = new Grid();
        head.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        head.Children.Add(UI.Caption("What the server prints: the model loading, and anything that goes wrong."));
        Grid.SetColumn(copy, 1);
        head.Children.Add(copy);
        return UI.Section("Server output", new StackPanel { Spacing = 10, Children = { head, _log } });
    }

    private void AppendLog(string line)
    {
        _log.Text = _server.RecentOutput;
        _log.SelectionStart = _log.Text.Length;
    }

    // ---- Settings ----

    private void Load()
    {
        _loading = true;
        _port.Value = _settings.Port;
        _network.IsOn = _settings.OnNetwork;
        _key.Text = _settings.ApiKey;
        _context.SelectedIndex = Math.Max(0, Array.IndexOf(Settings.ContextChoices, _settings.Context));
        _thinking.IsOn = _settings.Thinking;
        _slots.Value = _settings.Slots;
        _startWithApp.IsOn = _settings.StartWithApp;
        _loading = false;
        FillModels();
    }

    private void Save()
    {
        if (_loading) return;
        if (!double.IsNaN(_port.Value)) _settings.Port = (int)_port.Value;
        _settings.OnNetwork = _network.IsOn;
        _settings.ApiKey = _key.Text.Trim();
        if (_context.SelectedIndex >= 0) _settings.Context = Settings.ContextChoices[_context.SelectedIndex];
        _settings.Thinking = _thinking.IsOn;
        if (!double.IsNaN(_slots.Value)) _settings.Slots = (int)_slots.Value;
        _settings.StartWithApp = _startWithApp.IsOn;
        _settings.Save();
        App.SettingsChanged();
        ShowState();
    }
}
