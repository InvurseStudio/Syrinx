// The Connect page: what another chat app needs to use the model. Syrinx speaks the OpenAI
// chat API, so any app that lets you add an OpenAI-compatible server works: the base URL
// (on this PC, and on the network when that's on), the API key, and the model's name.
using System.Net.NetworkInformation;
using System.Net.Sockets;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Syrinx.Manager.Views;

internal sealed class ConnectPage : Grid
{
    private readonly Settings _settings = App.Settings;
    private readonly ServerProcess _server = App.Server;
    private readonly InfoBar _notRunning = new()
    {
        IsClosable = false,
        Severity = InfoBarSeverity.Warning,
        Title = "The server isn't running",
        Message = "Start it on the Server page. These are the details it will use.",
    };
    private readonly StackPanel _content = new() { Spacing = 24 };

    public ConnectPage()
    {
        var page = UI.Page(UI.PageTitle("Connect"),
            UI.Caption("Syrinx speaks the OpenAI chat API. In any app that lets you add an OpenAI-compatible server or provider, use the details below."),
            _notRunning, _content);
        Children.Add(page);
        Build();
        _server.Changed += (_, _) => Build();
        App.SettingsSaved += (_, _) => Build();
    }

    private string Port => (_server.Address?.Port ?? _settings.Port).ToString(System.Globalization.CultureInfo.InvariantCulture);
    private string LocalBase => $"http://127.0.0.1:{Port}/v1";
    private string Key => _settings.ApiKey;
    private string ModelName => Path.GetFileNameWithoutExtension(_server.Model ?? _settings.Model) ?? "local";

    private void Build()
    {
        _notRunning.IsOpen = _server.State != ServerState.Running;
        _content.Children.Clear();

        var addresses = new List<UIElement>
        {
            UI.Row("On this PC", "For apps running on this computer.", Field(LocalBase, 360)),
        };
        if (_settings.OnNetwork)
        {
            foreach (string ip in NetworkAddresses())
            {
                addresses.Add(UI.Row($"On the network ({ip})", "For phones and other computers on the same network.", Field($"http://{ip}:{Port}/v1", 360)));
            }
        }
        else
        {
            addresses.Add(UI.Caption("Only apps on this PC can connect. To let other devices in, turn on \"Let other devices on this network connect\" on the Server page."));
        }
        _content.Children.Add(UI.Section("Base URL", [.. addresses]));

        _content.Children.Add(UI.Section("Sign-in",
            UI.Row("API key", string.IsNullOrEmpty(Key)
                    ? "No key is set: apps on this PC can give anything (some apps need a value; any will do)."
                    : "Give this as the app's API key.",
                Field(string.IsNullOrEmpty(Key) ? "none" : Key, 360)),
            UI.Row("Model", "Syrinx serves one model; apps that ask for a name can use this one, or any.", Field(ModelName, 360))));

        var chat = UI.Button("Open the chat page", Glyphs.Chat);
        chat.Click += (_, _) => Open($"http://127.0.0.1:{Port}/");
        chat.IsEnabled = _server.State == ServerState.Running;
        _content.Children.Add(UI.Section("In a browser",
            UI.Row("Syrinx's own chat page", "A Vireo-style chat in any browser on this PC" + (_settings.OnNetwork ? $", or on the network at http://<address>:{Port}/" : "") + ".", chat)));

        _content.Children.Add(UI.Section("From code",
            Example("Python (the openai package)", Python()),
            Example("curl", Curl())));
    }

    private static Grid Field(string value, double width)
    {
        var field = UI.CopyField(() => value, out _);
        field.Width = width;
        return field;
    }

    private static StackPanel Example(string title, string code)
    {
        var text = UI.Text(code, 12, wrap: false);
        text.FontFamily = Theme.Monospace;
        text.IsTextSelectionEnabled = true;
        var box = new Border
        {
            Child = new ScrollViewer { Content = text, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled },
            Background = Theme.BgTertiary,
            CornerRadius = new CornerRadius(6),
            Padding = new Thickness(12, 10, 12, 10),
        };
        var head = new Grid();
        head.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        head.Children.Add(UI.Text(title, 13));
        var copy = UI.CopyButton(() => code, "Copy the example");
        Grid.SetColumn(copy, 1);
        head.Children.Add(copy);
        return new StackPanel { Spacing = 8, Children = { head, box } };
    }

    private string Python() =>
        $$"""
        from openai import OpenAI

        client = OpenAI(base_url="{{LocalBase}}", api_key="{{(string.IsNullOrEmpty(Key) ? "none" : Key)}}")
        reply = client.chat.completions.create(
            model="{{ModelName}}",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(reply.choices[0].message.content)
        """;

    private string Curl() =>
        $"curl {LocalBase}/chat/completions -H \"Content-Type: application/json\"" +
        (string.IsNullOrEmpty(Key) ? "" : $" -H \"Authorization: Bearer {Key}\"") +
        $" -d \"{{\\\"model\\\": \\\"{ModelName}\\\", \\\"messages\\\": [{{\\\"role\\\": \\\"user\\\", \\\"content\\\": \\\"Hello!\\\"}}]}}\"";

    /// <summary>This PC's IPv4 addresses on networks that are up (not loopback or tunnels).</summary>
    private static List<string> NetworkAddresses()
    {
        var found = new List<string>();
        try
        {
            foreach (var adapter in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (adapter.OperationalStatus != OperationalStatus.Up
                    || adapter.NetworkInterfaceType is NetworkInterfaceType.Loopback or NetworkInterfaceType.Tunnel)
                {
                    continue;
                }
                foreach (var unicast in adapter.GetIPProperties().UnicastAddresses)
                {
                    if (unicast.Address.AddressFamily == AddressFamily.InterNetwork && !unicast.Address.ToString().StartsWith("169.254.", StringComparison.Ordinal))
                    {
                        found.Add(unicast.Address.ToString());
                    }
                }
            }
        }
        catch (NetworkInformationException)
        {
            // None to show.
        }
        return found.Distinct().ToList();
    }

    private static void Open(string url) =>
        System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(url) { UseShellExecute = true });
}
