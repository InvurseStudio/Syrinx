// The window: a sidebar of pages (Server, Connect, Statistics) with the server's state at
// its foot, as Vireo's Windows app lays out its settings (a sidebar, the title bar hidden so
// it runs to the top), and the page beside it.
using System.Runtime.InteropServices;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Syrinx.Manager.Views;
using WinRT.Interop;

namespace Syrinx.Manager;

public sealed partial class MainWindow : Window
{
    private const int DefaultWidth = 1100, DefaultHeight = 760;
    private const int MinimumWidth = 820, MinimumHeight = 560;

    private readonly ContentPresenter _page = new();
    private readonly List<(Button Button, Func<UIElement> Make, string Name)> _nav = [];
    private readonly Dictionary<string, UIElement> _pages = [];
    private readonly Microsoft.UI.Xaml.Shapes.Ellipse _stateDot = UI.Dot(Theme.TextSecondary);
    private readonly TextBlock _stateText = UI.Text("Stopped", 12, Theme.TextSecondary, wrap: false);
    private readonly TextBlock _stateDetail = UI.Text("", 11, Theme.TextSecondary, wrap: false);

    public MainWindow()
    {
        Title = "Syrinx";
        App.MainWindowHandle = WindowNative.GetWindowHandle(this);
        ExtendsContentIntoTitleBar = true;
        // The taskbar and Alt+Tab icon; the .exe gets the same one from ApplicationIcon.
        var icon = Path.Combine(AppContext.BaseDirectory, "assets", "Syrinx.ico");
        if (File.Exists(icon)) AppWindow.SetIcon(icon);
        AppWindow.TitleBar.ButtonBackgroundColor = Colors.Transparent;
        AppWindow.TitleBar.ButtonInactiveBackgroundColor = Colors.Transparent;
        double scale = GetDpiForWindow(App.MainWindowHandle) / 96.0;
        AppWindow.Resize(new((int)(DefaultWidth * scale), (int)(DefaultHeight * scale)));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.PreferredMinimumWidth = (int)(MinimumWidth * scale);
            presenter.PreferredMinimumHeight = (int)(MinimumHeight * scale);
        }

        var root = new Grid { Background = Theme.BgPrimary };
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(240) });
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(32) });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });

        var sidebarBack = new Border { Background = Theme.BgSecondary, BorderBrush = Theme.Border, BorderThickness = new Thickness(0, 0, 1, 0) };
        Grid.SetRowSpan(sidebarBack, 2);
        root.Children.Add(sidebarBack);

        var sidebar = Sidebar();
        Grid.SetRow(sidebar, 1);
        root.Children.Add(sidebar);

        Grid.SetRow(_page, 1);
        Grid.SetColumn(_page, 1);
        root.Children.Add(_page);

        // Drawn last so it's on top: the draggable title bar across both columns.
        var titleBar = new Grid { Background = new SolidColorBrush(Colors.Transparent) };
        Grid.SetColumnSpan(titleBar, 2);
        var titleText = UI.Text("Syrinx", 12, Theme.TextSecondary, wrap: false);
        titleText.Margin = new Thickness(16, 0, 0, 0);
        titleText.VerticalAlignment = VerticalAlignment.Center;
        titleBar.Children.Add(titleText);
        root.Children.Add(titleBar);

        Content = root;
        SetTitleBar(titleBar);

        App.Server.Changed += (_, _) => ShowState();
        ShowState();
        Show("Server");
    }

    private Grid Sidebar()
    {
        var grid = new Grid();
        grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        var brand = new StackPanel { Margin = new Thickness(20, 6, 16, 18), Spacing = 2 };
        brand.Children.Add(UI.Text("Syrinx", 20, bold: true));
        brand.Children.Add(UI.Caption("Local models for any chat app"));
        grid.Children.Add(brand);

        var nav = new StackPanel { Spacing = 2, Margin = new Thickness(10, 0, 10, 0) };
        AddNav(nav, "Server", Glyphs.Server, () => new ServerPage());
        AddNav(nav, "Models", Glyphs.Models, () => new ModelsPage());
        AddNav(nav, "Connect", Glyphs.Connect, () => new ConnectPage());
        AddNav(nav, "Statistics", Glyphs.Chart, () => new StatsPage());
        Grid.SetRow(nav, 1);
        grid.Children.Add(nav);

        var state = new StackPanel { Spacing = 3, Padding = new Thickness(20, 12, 16, 14) };
        var line = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        line.Children.Add(_stateDot);
        line.Children.Add(_stateText);
        state.Children.Add(line);
        state.Children.Add(_stateDetail);
        var foot = new Border { Child = state, BorderBrush = Theme.Border, BorderThickness = new Thickness(0, 1, 0, 0) };
        Grid.SetRow(foot, 2);
        grid.Children.Add(foot);
        return grid;
    }

    private void AddNav(StackPanel nav, string name, string glyph, Func<UIElement> make)
    {
        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 12 };
        content.Children.Add(UI.Icon(glyph, 14));
        content.Children.Add(new TextBlock { Text = name, FontSize = 13.5 });
        var button = new Button
        {
            Content = content,
            HorizontalAlignment = HorizontalAlignment.Stretch,
            HorizontalContentAlignment = HorizontalAlignment.Left,
            Padding = new Thickness(12, 9, 12, 9),
            Background = new SolidColorBrush(Colors.Transparent),
            BorderThickness = new Thickness(0),
            CornerRadius = new CornerRadius(9),
        };
        button.Click += (_, _) => Show(name);
        _nav.Add((button, make, name));
        nav.Children.Add(button);
    }

    /// <summary>Opens a page; each is made once and kept, so the statistics keep their place.</summary>
    public void Show(string name)
    {
        foreach (var (button, make, itemName) in _nav)
        {
            bool on = itemName == name;
            button.Background = on ? Theme.AccentSoft : new SolidColorBrush(Colors.Transparent);
            button.Foreground = on ? Theme.TextPrimary : Theme.TextSecondary;
            if (on)
            {
                if (!_pages.TryGetValue(name, out var page)) _pages[name] = page = make();
                _page.Content = page;
            }
        }
    }

    private void ShowState()
    {
        var server = App.Server;
        var (color, text) = UI.Describe(server.State);
        _stateDot.Fill = color;
        _stateText.Text = text;
        _stateText.Foreground = server.State == ServerState.Running ? Theme.TextPrimary : Theme.TextSecondary;
        _stateDetail.Text = server.State switch
        {
            ServerState.Running when server.Address is { } address => $"{Path.GetFileNameWithoutExtension(server.Model)} · port {address.Port}",
            ServerState.Loading => Path.GetFileNameWithoutExtension(server.Model) ?? "",
            _ => "",
        };
        _stateDetail.Visibility = _stateDetail.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    [LibraryImport("user32.dll")]
    private static partial uint GetDpiForWindow(IntPtr hwnd);
}
