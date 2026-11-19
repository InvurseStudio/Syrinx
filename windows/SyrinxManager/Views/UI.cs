// The colours in Themes/AppTheme.xaml and small building blocks for views made in code,
// in the shapes Vireo's Windows app uses (Vireo's Windows/Vireo.WinUI/Views/Theme.cs and
// SettingsUI.cs).
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.ApplicationModel.DataTransfer;

namespace Syrinx.Manager.Views;

internal static class Theme
{
    public static Brush BgPrimary => Brush("BgPrimaryBrush");
    public static Brush BgSecondary => Brush("BgSecondaryBrush");
    public static Brush BgTertiary => Brush("BgTertiaryBrush");
    public static Brush TextPrimary => Brush("TextPrimaryBrush");
    public static Brush TextSecondary => Brush("TextSecondaryBrush");
    public static Brush Accent => Brush("AccentBrush");
    public static Brush Border => Brush("BorderBrush");
    public static Brush Danger => Brush("DangerBrush");
    public static Brush Warning => Brush("WarningBrush");
    public static Brush Raised => Brush("RaisedBrush");
    public static Brush AccentSoft => Brush("AccentSoftBrush");

    public static readonly FontFamily Monospace = new("Cascadia Mono, Consolas");

    private static Brush Brush(string key) => (Brush)Application.Current.Resources[key];
}

/// <summary>Segoe Fluent Icons.</summary>
internal static class Glyphs
{
    public const string Server = "";
    public const string Models = "";
    public const string Connect = "";
    public const string Chart = "";
    public const string Play = "";
    public const string Stop = "";
    public const string Restart = "";
    public const string Copy = "";
    public const string Check = "";
    public const string Folder = "";
    public const string Add = "";
    public const string Delete = "";
    public const string Refresh = "";
    public const string Key = "";
    public const string Open = "";
    public const string Chat = "";
}

internal static class UI
{
    public static TextBlock Text(string text, double size = 13, Brush? color = null, bool bold = false, bool wrap = true)
    {
        var block = new TextBlock
        {
            Text = text,
            FontSize = size,
            Foreground = color ?? Theme.TextPrimary,
            TextWrapping = wrap ? TextWrapping.Wrap : TextWrapping.NoWrap,
            TextTrimming = wrap ? TextTrimming.None : TextTrimming.CharacterEllipsis,
        };
        if (bold) block.FontWeight = FontWeights.SemiBold;
        return block;
    }

    public static TextBlock Caption(string text) => Text(text, 12, Theme.TextSecondary);

    /// <summary>A page's title, as Vireo's settings pages have them.</summary>
    public static TextBlock PageTitle(string text)
    {
        var block = Text(text, 24, bold: true);
        block.Margin = new Thickness(0, 0, 0, 4);
        return block;
    }

    /// <summary>A section: its title in small capitals, then its rows in a card.</summary>
    public static StackPanel Section(string title, params UIElement[] rows)
    {
        var card = new StackPanel { Spacing = 0 };
        for (int i = 0; i < rows.Length; i++)
        {
            var row = new Border
            {
                Child = rows[i],
                Padding = new Thickness(16, 12, 16, 12),
                BorderBrush = Theme.Border,
                BorderThickness = new Thickness(0, i == 0 ? 0 : 1, 0, 0),
            };
            card.Children.Add(row);
        }
        var heading = Text(title.ToUpperInvariant(), 10.5, Theme.TextSecondary, bold: true);
        heading.CharacterSpacing = 60;
        return new StackPanel
        {
            Spacing = 8,
            Children =
            {
                heading,
                new Border
                {
                    Child = card,
                    Background = Theme.BgSecondary,
                    BorderBrush = Theme.Border,
                    BorderThickness = new Thickness(1),
                    CornerRadius = new CornerRadius(10),
                },
            },
        };
    }

    /// <summary>A row: a label and caption on the left, a control on the right.</summary>
    public static Grid Row(string label, string? caption, FrameworkElement control)
    {
        var grid = new Grid { ColumnSpacing = 16 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var left = new StackPanel { Spacing = 2, VerticalAlignment = VerticalAlignment.Center };
        left.Children.Add(Text(label));
        if (caption is not null) left.Children.Add(Caption(caption));
        grid.Children.Add(left);
        control.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(control, 1);
        grid.Children.Add(control);
        return grid;
    }

    public static FontIcon Icon(string glyph, double size = 13, Brush? color = null)
    {
        var icon = new FontIcon { Glyph = glyph, FontSize = size };
        if (color is not null) icon.Foreground = color;
        return icon;
    }

    /// <summary>A button with an icon and text.</summary>
    public static Button Button(string text, string? glyph = null, bool accent = false)
    {
        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        if (glyph is not null) content.Children.Add(Icon(glyph, 12));
        content.Children.Add(new TextBlock { Text = text });
        var button = new Button { Content = content };
        if (accent) button.Style = (Style)Application.Current.Resources["AccentButtonStyle"];
        return button;
    }

    /// <summary>A small copy button: copies what <paramref name="value"/> gives, and says so.</summary>
    public static Button CopyButton(Func<string> value, string help = "Copy")
    {
        var button = new Button
        {
            Content = Icon(Glyphs.Copy, 12),
            Padding = new Thickness(8, 6, 8, 6),
        };
        ToolTipService.SetToolTip(button, help);
        button.Click += async (_, _) =>
        {
            var package = new DataPackage();
            package.SetText(value());
            Clipboard.SetContent(package);
            button.Content = Icon(Glyphs.Check, 12, Theme.Accent);
            await Task.Delay(1500);
            button.Content = Icon(Glyphs.Copy, 12);
        };
        return button;
    }

    /// <summary>A value shown in monospace in a raised box, with a copy button beside it.</summary>
    public static Grid CopyField(Func<string> value, out TextBlock shown, string help = "Copy")
    {
        shown = Text(value(), 13, wrap: false);
        shown.FontFamily = Theme.Monospace;
        shown.IsTextSelectionEnabled = true;
        shown.VerticalAlignment = VerticalAlignment.Center;
        var box = new Border
        {
            Child = shown,
            Background = Theme.BgTertiary,
            CornerRadius = new CornerRadius(6),
            Padding = new Thickness(10, 7, 10, 7),
        };
        var grid = new Grid { ColumnSpacing = 8 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.Children.Add(box);
        var copy = CopyButton(value, help);
        Grid.SetColumn(copy, 1);
        grid.Children.Add(copy);
        return grid;
    }

    /// <summary>A page's content: scrolls, at a readable width.</summary>
    public static ScrollViewer Page(params UIElement[] children)
    {
        var stack = new StackPanel { Spacing = 24, MaxWidth = 760, Padding = new Thickness(32, 24, 32, 32) };
        foreach (var child in children) stack.Children.Add(child);
        return new ScrollViewer { Content = stack, HorizontalAlignment = HorizontalAlignment.Stretch };
    }

    public static Microsoft.UI.Xaml.Shapes.Ellipse Dot(Brush color, double size = 8) =>
        new() { Width = size, Height = size, Fill = color, VerticalAlignment = VerticalAlignment.Center };

    /// <summary>A server state's dot colour and words.</summary>
    public static (Brush Color, string Text) Describe(ServerState state) => state switch
    {
        ServerState.Running => (Theme.Accent, "Running"),
        ServerState.Loading => (Theme.Warning, "Loading the model…"),
        ServerState.Stopping => (Theme.Warning, "Stopping…"),
        ServerState.Failed => (Theme.Danger, "Couldn't start"),
        _ => (Theme.TextSecondary, "Stopped"),
    };
}
