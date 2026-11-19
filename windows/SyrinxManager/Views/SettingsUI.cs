// What the Settings pages are made of, after Windows 11's own Settings: a page title, small
// section headers, and cards with an icon, a title, a line of description and a control on
// the right. The Settings window can be light or dark while the chat window stays dark, so
// these colours are chosen here for the window's theme rather than read from AppTheme.xaml;
// the window rebuilds its page when the theme changes.
using Microsoft.UI;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.UI;

namespace Syrinx.Manager.Views;

internal enum Tone { Neutral, Accent, Success, Caution, Critical }

internal static class SettingsUI
{
    /// <summary>Set by the Settings window before it builds a page.</summary>
    public static bool IsLight { get; set; }

    private static SolidColorBrush Pick(uint dark, uint light) => new(FromArgb(IsLight ? light : dark));

    private static Color FromArgb(uint argb) =>
        Color.FromArgb((byte)(argb >> 24), (byte)(argb >> 16), (byte)(argb >> 8), (byte)argb);

    // Windows 11's own fills and strokes (CardBackgroundFillColorDefault and friends), with
    // Vireo's green as the accent.
    public static Brush TextPrimary => Pick(0xFFFFFFFF, 0xE4000000);
    public static Brush TextSecondary => Pick(0xC5FFFFFF, 0x9E000000);
    public static Brush TextTertiary => Pick(0x87FFFFFF, 0x72000000);
    public static Brush CardBackground => Pick(0x0DFFFFFF, 0xB3FFFFFF);
    public static Brush CardBorder => Pick(0x19000000, 0x0F000000);
    public static Brush Layer => Pick(0x4C3A3A3A, 0x80FFFFFF);
    public static Brush LayerBorder => Pick(0x19000000, 0x0F000000);
    public static Brush Subtle => Pick(0x0FFFFFFF, 0x0A000000);
    public static Brush Accent => Pick(0xFF4CF0AC, 0xFF0B7A4F);
    public static Brush Success => Pick(0xFF6CCB5F, 0xFF0F7B0F);
    public static Brush Caution => Pick(0xFFFCE100, 0xFF9D5D00);
    public static Brush Critical => Pick(0xFFFF99A4, 0xFFC42B1C);

    public static Brush ToneBrush(Tone tone) => tone switch
    {
        Tone.Accent => Accent,
        Tone.Success => Success,
        Tone.Caution => Caution,
        Tone.Critical => Critical,
        _ => TextSecondary,
    };

    private static Brush ToneBackground(Tone tone) => tone switch
    {
        Tone.Accent => Pick(0x264CF0AC, 0x1F0B7A4F),
        Tone.Success => Pick(0x266CCB5F, 0x1A0F7B0F),
        Tone.Caution => Pick(0x26FCE100, 0x1F9D5D00),
        Tone.Critical => Pick(0x26FF99A4, 0x1AC42B1C),
        _ => Subtle,
    };

    public static TextBlock PageTitle(string text) =>
        new() { Text = text, FontSize = 28, FontWeight = FontWeights.SemiBold, Foreground = TextPrimary, Margin = new Thickness(0, 0, 0, 8) };

    /// <summary>A section's name, above its cards.</summary>
    public static TextBlock Header(string text) =>
        new() { Text = text, FontSize = 14, FontWeight = FontWeights.SemiBold, Foreground = TextPrimary, Margin = new Thickness(1, 20, 0, 4) };

    public static TextBlock Note(string text) =>
        new() { Text = text, FontSize = 12, Foreground = TextSecondary, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(1, 0, 0, 0) };

    public static TextBlock Status() => new() { FontSize = 13, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };

    public static TextBlock Value(string text) => new()
    {
        Text = text, FontSize = 13, Foreground = TextSecondary, TextWrapping = TextWrapping.Wrap,
        IsTextSelectionEnabled = true, VerticalAlignment = VerticalAlignment.Center, TextAlignment = TextAlignment.Right,
    };

    /// <summary>Cards one under the other, a few pixels apart, as Windows' Settings groups them.</summary>
    public static StackPanel Stack(params UIElement[] children)
    {
        var stack = new StackPanel { Spacing = 4 };
        foreach (var child in children) stack.Children.Add(child);
        return stack;
    }

    public static Border Card(UIElement content, Thickness? padding = null) => new()
    {
        Background = CardBackground,
        BorderBrush = CardBorder,
        BorderThickness = new Thickness(1),
        CornerRadius = new CornerRadius(8),
        Padding = padding ?? new Thickness(16, 12, 16, 12),
        Child = content,
    };

    /// <summary>A setting: icon, title and description on the left, its control on the right.</summary>
    public static Border Row(string? glyph, string title, string? description, UIElement? control = null) =>
        Card(RowContent(glyph, title, description, control), new Thickness(16, 12, 16, 12));

    public static Grid RowContent(string? glyph, string title, string? description, UIElement? control, UIElement? extra = null)
    {
        var grid = new Grid { ColumnSpacing = 16, MinHeight = 44 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        if (glyph is not null)
        {
            // Beside the title when the card holds more below it, else centred.
            grid.Children.Add(new FontIcon
            {
                Glyph = glyph, FontSize = 20, Foreground = TextPrimary, Width = 24,
                VerticalAlignment = extra is null ? VerticalAlignment.Center : VerticalAlignment.Top,
                Margin = extra is null ? new Thickness(0) : new Thickness(0, 2, 0, 0),
            });
        }
        var text = new StackPanel { VerticalAlignment = VerticalAlignment.Center, Spacing = 2 };
        text.Children.Add(new TextBlock { Text = title, FontSize = 14, Foreground = TextPrimary, TextWrapping = TextWrapping.Wrap });
        if (description is { Length: > 0 })
        {
            text.Children.Add(new TextBlock { Text = description, FontSize = 12, Foreground = TextSecondary, TextWrapping = TextWrapping.Wrap });
        }
        if (extra is not null) text.Children.Add(extra);
        Grid.SetColumn(text, 1);
        grid.Children.Add(text);
        if (control is FrameworkElement element)
        {
            element.VerticalAlignment = VerticalAlignment.Center;
            Grid.SetColumn(element, 2);
            grid.Children.Add(element);
        }
        return grid;
    }

    /// <summary>A small rounded label: "Runs on the GPU", "Q4_K_M".</summary>
    public static Border Badge(string text, Tone tone = Tone.Neutral, string? glyph = null)
    {
        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4 };
        if (glyph is not null) content.Children.Add(new FontIcon { Glyph = glyph, FontSize = 10, Foreground = ToneBrush(tone) });
        content.Children.Add(new TextBlock { Text = text, FontSize = 11, Foreground = tone == Tone.Neutral ? TextSecondary : ToneBrush(tone) });
        return new Border
        {
            Background = ToneBackground(tone),
            CornerRadius = new CornerRadius(4),
            Padding = new Thickness(6, 2, 6, 3),
            Child = content,
            VerticalAlignment = VerticalAlignment.Center,
        };
    }

    public static StackPanel Badges(params UIElement[] badges)
    {
        var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6, Margin = new Thickness(0, 4, 0, 0) };
        foreach (var badge in badges) row.Children.Add(badge);
        return row;
    }

    public static StackPanel Buttons(params UIElement[] buttons)
    {
        var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        foreach (var button in buttons) row.Children.Add(button);
        return row;
    }

    public static Button Button(string text, Action click, bool accent = false)
    {
        var button = new Button { Content = text };
        if (accent) button.Style = (Style)Application.Current.Resources["AccentButtonStyle"];
        button.Click += (_, _) => click();
        return button;
    }

    public static Button IconButton(string glyph, string name, Action click)
    {
        var button = new Button
        {
            Content = new FontIcon { Glyph = glyph, FontSize = 14 },
            Padding = new Thickness(8),
            Background = new SolidColorBrush(Colors.Transparent),
            BorderThickness = new Thickness(0),
        };
        ToolTipService.SetToolTip(button, name);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(button, name);
        button.Click += (_, _) => click();
        return button;
    }

    public static HyperlinkButton Link(string text, Uri uri) => new() { Content = text, NavigateUri = uri, Padding = new Thickness(0), FontSize = 13 };

    /// <summary>A label and its value on one line, inside a card.</summary>
    public static Grid Pair(string label, string value)
    {
        var grid = new Grid { ColumnSpacing = 16, Margin = new Thickness(0, 2, 0, 2) };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.Children.Add(new TextBlock { Text = label, FontSize = 13, Foreground = TextPrimary });
        var text = Value(value);
        Grid.SetColumn(text, 1);
        grid.Children.Add(text);
        return grid;
    }
}
