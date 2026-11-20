// Vireo's look (the Mac app's AppTheme.swift, the Windows apps' AppTheme.xaml): dark navy
// with a green accent, dark only. And the pieces every page is made of: a page with its
// title, sections, rows of a label, a caption and a control, cards and badges.
import AppKit
import SwiftUI

enum Theme {
    private static func rgb(_ hex: UInt32, _ alpha: Double = 1) -> Color {
        Color(nsColor: NSColor(srgbRed: CGFloat((hex >> 16) & 0xff) / 255, green: CGFloat((hex >> 8) & 0xff) / 255,
                               blue: CGFloat(hex & 0xff) / 255, alpha: alpha))
    }

    static let bgPrimary = rgb(0x0B1018)
    static let bgSecondary = rgb(0x0E141E)
    static let bgTertiary = rgb(0x111A26)
    static let textPrimary = rgb(0xE8F0F1)
    static let textSecondary = rgb(0x8D9BA8)
    static let accent = rgb(0x4CF0AC)
    static let border = rgb(0xE8F0F1, 0.10)
    static let danger = rgb(0xF87171)
    static let warning = rgb(0xFACC15)
    static let raised = rgb(0xE8F0F1, 0.06)
    static let accentSoft = rgb(0x4CF0AC, 0.10)
    static let backgroundNS = NSColor(srgbRed: 0x0B / 255, green: 0x10 / 255, blue: 0x18 / 255, alpha: 1)

    static func describe(_ state: ServerState) -> (color: Color, text: String) {
        switch state {
        case .running: return (accent, "Running")
        case .loading: return (warning, "Loading the model…")
        case .stopping: return (warning, "Stopping…")
        case .failed: return (danger, "Couldn't start")
        case .stopped: return (textSecondary, "Stopped")
        }
    }
}

enum Tone {
    case neutral, accent, success, caution, critical

    var foreground: Color {
        switch self {
        case .neutral: return Theme.textSecondary
        case .accent, .success: return Theme.accent
        case .caution: return Theme.warning
        case .critical: return Theme.danger
        }
    }

    var background: Color {
        switch self {
        case .neutral: return Theme.raised
        case .accent, .success: return Theme.accentSoft
        case .caution: return Theme.warning.opacity(0.12)
        case .critical: return Theme.danger.opacity(0.12)
        }
    }
}

/// A page: its title and sections, scrolling, at most 820 points wide.
struct Page<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text(title).font(.system(size: 26, weight: .semibold)).foregroundStyle(Theme.textPrimary)
                content
            }
            .frame(maxWidth: 820, alignment: .leading)
            .padding(.horizontal, 32)
            .padding(.top, 8)
            .padding(.bottom, 32)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// A section: a heading over a card of rows, divided by lines.
struct SettingsSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title)
            VStack(alignment: .leading, spacing: 0) {
                _VariadicView.Tree(DividedLayout()) { content }
            }
            .background(Theme.bgSecondary, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.border))
        }
    }
}

private struct DividedLayout: _VariadicView_MultiViewRoot {
    func body(children: _VariadicView.Children) -> some View {
        let last = children.last?.id
        ForEach(children) { child in
            child.padding(.horizontal, 16).padding(.vertical, 12)
            if child.id != last { Divider().overlay(Theme.border) }
        }
    }
}

struct SectionHeader: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title).font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.textPrimary)
    }
}

/// A label with a caption under it, and a control beside them.
struct SettingRow<Control: View>: View {
    let title: String
    let caption: String?
    @ViewBuilder let control: Control

    init(_ title: String, _ caption: String? = nil, @ViewBuilder control: () -> Control) {
        self.title = title
        self.caption = caption
        self.control = control()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 13)).foregroundStyle(Theme.textPrimary)
                if let caption { Caption(caption) }
            }
            Spacer(minLength: 12)
            control
        }
    }
}

struct Caption: View {
    let text: String

    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text).font(.system(size: 12)).foregroundStyle(Theme.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }
}

/// A raised box around content.
struct Card<Content: View>: View {
    var padding = EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16)
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.bgSecondary, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.border))
    }
}

/// An icon, a title and details, actions on the right, and more below.
struct ItemRow<Actions: View, Extra: View>: View {
    let symbol: String
    let title: String
    let details: String
    @ViewBuilder let actions: Actions
    @ViewBuilder let extra: Extra

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: symbol).font(.system(size: 18)).foregroundStyle(Theme.textSecondary).frame(width: 24).padding(.top, 2)
            VStack(alignment: .leading, spacing: 8) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.textPrimary).lineLimit(1).truncationMode(.middle)
                    Caption(details)
                }
                extra
            }
            Spacer(minLength: 8)
            actions
        }
    }
}

extension ItemRow where Extra == EmptyView {
    init(symbol: String, title: String, details: String, @ViewBuilder actions: () -> Actions) {
        self.init(symbol: symbol, title: title, details: details, actions: actions, extra: { EmptyView() })
    }
}

struct Badge: View {
    let text: String
    var tone: Tone = .neutral
    var symbol: String?

    var body: some View {
        HStack(spacing: 4) {
            if let symbol { Image(systemName: symbol).font(.system(size: 10, weight: .semibold)) }
            Text(text).font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(tone.foreground)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(tone.background, in: Capsule())
    }
}

/// The green button for a page's main action.
struct AccentButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(Color.black.opacity(isEnabled ? 0.9 : 0.5))
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Theme.accent.opacity(isEnabled ? (configuration.isPressed ? 0.8 : 1) : 0.35), in: RoundedRectangle(cornerRadius: 7))
    }
}

/// A read-only value with a button that copies it.
struct CopyField: View {
    let value: String
    var width: CGFloat = 360
    @State private var copied = false

    var body: some View {
        HStack(spacing: 6) {
            Text(value)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            CopyButton(value: { value }, help: "Copy")
        }
        .padding(.leading, 10)
        .padding(.trailing, 4)
        .padding(.vertical, 4)
        .frame(width: width)
        .background(Theme.bgTertiary, in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Theme.border))
    }
}

struct CopyButton: View {
    let value: () -> String
    var help = "Copy"
    @State private var copied = false

    var body: some View {
        Button {
            copy(value())
            copied = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc").frame(width: 16)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(copied ? Theme.accent : Theme.textSecondary)
        .help(help)
    }
}

/// A notice across the page: a title, a message and an optional action.
struct Banner<Action: View>: View {
    let tone: Tone
    let title: String
    let message: String
    @ViewBuilder let action: Action

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: tone == .caution ? "exclamationmark.triangle.fill" : "info.circle.fill").foregroundStyle(tone.foreground)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.textPrimary)
                Caption(message)
            }
            Spacer()
            action
        }
        .padding(12)
        .background(tone.background, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(tone.foreground.opacity(0.3)))
    }
}

extension Banner where Action == EmptyView {
    init(tone: Tone, title: String, message: String) {
        self.init(tone: tone, title: title, message: message, action: { EmptyView() })
    }
}

func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}

/// Asks before something that can't be undone; true to go ahead.
@MainActor
func confirm(_ title: String, _ message: String, button: String, destructive: Bool = false) -> Bool {
    let alert = NSAlert()
    alert.messageText = title
    alert.informativeText = message
    alert.alertStyle = destructive ? .critical : .warning
    let go = alert.addButton(withTitle: button)
    go.hasDestructiveAction = destructive
    alert.addButton(withTitle: "Cancel")
    if destructive {
        // Return cancels, so a destructive action takes a deliberate click.
        go.keyEquivalent = ""
        alert.buttons[1].keyEquivalent = "\r"
    }
    return alert.runModal() == .alertFirstButtonReturn
}

@MainActor
func showError(_ message: String) {
    let alert = NSAlert()
    alert.messageText = "Something went wrong"
    alert.informativeText = message
    alert.addButton(withTitle: "OK")
    alert.runModal()
}

/// "2 h 5 min", "12 min", "40 s".
func formatDuration(_ seconds: TimeInterval) -> String {
    if seconds >= 3600 { return "\(Int(seconds / 3600)) h \(Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)) min" }
    if seconds >= 60 { return "\(Int(seconds / 60)) min" }
    return "\(max(1, Int(seconds))) s"
}
