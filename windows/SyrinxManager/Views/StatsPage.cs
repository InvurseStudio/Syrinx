// The Statistics page: the server's own dashboard (/stats, stats.html in syrinx-server: the
// Vireo Mac app's Statistics window as a page), shown here with the token this app started
// the server with, so it opens even when the server has an API key.
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Syrinx.Manager.Views;

internal sealed class StatsPage : Grid
{
    private readonly ServerProcess _server = App.Server;
    private readonly WebView2 _web = new() { Visibility = Visibility.Collapsed };
    private readonly StackPanel _empty;
    private string? _shown;

    public StatsPage()
    {
        var open = UI.Button("Open in browser", Glyphs.Open);
        open.HorizontalAlignment = HorizontalAlignment.Center;
        _empty = new StackPanel
        {
            Spacing = 10,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Children =
            {
                UI.Icon(Glyphs.Chart, 40, Theme.TextSecondary),
                Centered(UI.Text("The server isn't running", 17, bold: true)),
                Centered(UI.Caption("Start it to see how the model is being used: requests, speed, the prompt cache, and which apps are connected.")),
            },
        };
        Children.Add(_empty);
        Children.Add(_web);
        _server.Changed += (_, _) => Show();
        Show();
    }

    private static TextBlock Centered(TextBlock text)
    {
        text.TextAlignment = TextAlignment.Center;
        text.MaxWidth = 420;
        return text;
    }

    private void Show()
    {
        bool running = _server.State == ServerState.Running && _server.Address is not null;
        _web.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        _empty.Visibility = running ? Visibility.Collapsed : Visibility.Visible;
        if (!running)
        {
            _shown = null;
            return;
        }
        string url = $"http://127.0.0.1:{_server.Address!.Port}/stats#token={_server.Token}";
        if (url == _shown) return;
        _shown = url;
        _web.Source = new Uri(url);
    }
}
