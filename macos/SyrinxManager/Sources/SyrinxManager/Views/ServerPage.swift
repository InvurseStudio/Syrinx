// The Server page: the server's state with Start and Stop, the model it runs, how it's
// reached (port, the network, the API key), the model's options, and its output.
import SwiftUI

struct ServerPage: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess

    var body: some View {
        Page(title: "Server") {
            statusCard
            if server.state == .running || server.state == .loading, server.startedWith != ServerProcess.fingerprint(settings) {
                Banner(tone: .accent, title: "Restart to use the new settings",
                       message: "The server is still running with the ones it started with.") {
                    Button { app.start() } label: { Label("Restart", systemImage: "arrow.clockwise") }
                }
            }
            modelSection
            networkSection
            optionsSection
            SettingsSection("App") {
                SettingRow("Start the server when Syrinx opens", "With the model chosen above.") {
                    Toggle("", isOn: $settings.startWithApp).labelsHidden().toggleStyle(.switch)
                }
            }
            logSection
        }
    }

    // MARK: Status

    private var statusCard: some View {
        Card(padding: EdgeInsets(top: 18, leading: 20, bottom: 18, trailing: 20)) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    let (color, text) = Theme.describe(server.state)
                    Circle().fill(color).frame(width: 12, height: 12)
                    Text(text).font(.system(size: 20, weight: .semibold))
                    Spacer()
                    if server.isActive {
                        Button { app.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                            .disabled(server.state == .stopping)
                            .controlSize(.large)
                    } else {
                        Button { app.start() } label: { Label("Start", systemImage: "play.fill") }
                            .buttonStyle(AccentButtonStyle())
                            .disabled(ServerProcess.problem(settings) != nil)
                    }
                }
                if let detail { Caption(detail) }
                if server.state == .failed, let error = server.error {
                    Text(error).font(.system(size: 12)).foregroundStyle(Theme.danger).textSelection(.enabled)
                }
            }
        }
    }

    private var detail: String? {
        let model = server.model.map { (($0 as NSString).lastPathComponent as NSString).deletingPathExtension } ?? ""
        switch server.state {
        case .running:
            let context = server.context.map { $0.formatted() } ?? "?"
            let host = server.address?.host ?? "127.0.0.1"
            let port = server.address?.port ?? settings.port
            return "\(model) on \(server.device ?? "this Mac"), \(context) tokens of context. Listening on \(host):\(String(port))"
                + ((server.startedWith ?? "").contains("--host") ? ", and to other devices on the network." : ".")
        case .loading:
            return server.device.map { "\(model) on \($0)…" } ?? "Reading \(model)…"
        case .stopped:
            return ServerProcess.problem(settings) ?? "Start it to serve the model to your chat apps."
        default:
            return nil
        }
    }

    // MARK: Model

    /// The models to choose from: those that chat, and the chosen one even if it's elsewhere.
    private var choices: [InstalledModel] {
        var list = app.models.filter(\.isChat)
        if let chosen = settings.model, FileManager.default.fileExists(atPath: chosen), !list.contains(where: { samePath($0.path, chosen) }) {
            list.insert(app.library.describe(chosen), at: 0)
        }
        return list.sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    private var modelSection: some View {
        SettingsSection("Model") {
            SettingRow("Model", "A GGUF file: Gemma 4, Qwen 3.5 or 3.8, Llama 3, or Muse Glimmer.") {
                let list = choices
                Picker("", selection: Binding(
                    get: { list.first { samePath($0.path, settings.model) }?.path ?? "" },
                    set: { if !$0.isEmpty { settings.model = $0 } })) {
                    if list.isEmpty { Text(app.scanned ? "No models found" : "Reading the models…").tag("") }
                    if !list.isEmpty, !list.contains(where: { samePath($0.path, settings.model) }) { Text("Choose a model").tag("") }
                    ForEach(list) { model in
                        Text("\(model.title)  ·  \(ModelFit.formatBytes(model.size))").tag(model.path)
                    }
                }
                .labelsHidden()
                .frame(minWidth: 280, maxWidth: 420)
            }
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Button {
                        app.page = .models
                    } label: { Label("Download and manage models…", systemImage: "cube") }
                    Button {
                        Task { await app.scan() }
                    } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                }
                Caption("Looked for in: " + settings.allModelFolders().map { ($0.path as NSString).abbreviatingWithTildeInPath }.joined(separator: "; "))
            }
        }
    }

    // MARK: Network

    private var networkSection: some View {
        SettingsSection("Network") {
            SettingRow("Port", "Chat apps connect to this port. \(Settings.defaultPort) unless you change it.") {
                HStack(spacing: 4) {
                    TextField("", value: Binding(get: { settings.port }, set: { settings.port = min(max($0, 1), 65535) }),
                              format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                    Stepper("", value: $settings.port, in: 1...65535).labelsHidden()
                }
            }
            SettingRow("Let other devices on this network connect",
                       "Phones, laptops and other computers on your Wi-Fi or network. They need the API key.") {
                Toggle("", isOn: Binding(get: { settings.onNetwork }, set: { on in
                    // Other devices need a key: make one if there isn't.
                    if on, settings.apiKey.trimmingCharacters(in: .whitespaces).isEmpty { settings.apiKey = Settings.newApiKey() }
                    settings.onNetwork = on
                })).labelsHidden().toggleStyle(.switch)
            }
            SettingRow("API key", "Apps give it as their API key. Without one, any app on this Mac can use the server.") {
                HStack(spacing: 8) {
                    TextField("No key: anyone on this Mac can use it", text: $settings.apiKey)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                        .frame(width: 260)
                    Button { settings.apiKey = Settings.newApiKey() } label: { Label("New key", systemImage: "key") }
                        .fixedSize()
                    CopyButton(value: { settings.apiKey }, help: "Copy the key")
                }
            }
        }
    }

    // MARK: Options

    private var optionsSection: some View {
        SettingsSection("Model options") {
            SettingRow("Context length",
                       "How much of a chat the model keeps. Automatic is as much as the GPU's memory holds, up to the model's limit.") {
                Picker("", selection: $settings.context) {
                    ForEach(Settings.contextChoices, id: \.self) { choice in
                        Text(choice == 0 ? "Automatic" : "\(choice / 1024)K tokens").tag(choice)
                    }
                }
                .labelsHidden()
                .frame(width: 160)
            }
            SettingRow("Think before answering",
                       "For requests that don't say. Gemma 4, Qwen and Muse think first; apps get the thought apart from the answer.") {
                Toggle("", isOn: $settings.thinking).labelsHidden().toggleStyle(.switch)
            }
            SettingRow("Chats at once", "Replies written together, each in a slot of its own on the GPU. Gemma 4 models only.") {
                Stepper(value: $settings.slots, in: 1...16) {
                    Text("\(settings.slots)").monospacedDigit().frame(minWidth: 24, alignment: .trailing)
                }
            }
        }
    }

    // MARK: Log

    private var logSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader("Server output")
            HStack {
                Caption("What the server prints: the model loading, and anything that goes wrong.")
                Spacer()
                Button { copy(server.recentOutput) } label: { Label("Copy", systemImage: "doc.on.doc") }
            }
            ScrollViewReader { proxy in
                ScrollView([.vertical, .horizontal]) {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(server.log.isEmpty ? " " : server.recentOutput)
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(Theme.textPrimary)
                            .textSelection(.enabled)
                            .fixedSize()
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Color.clear.frame(height: 1).id("end")
                    }
                    .padding(10)
                }
                .frame(height: 220)
                .background(Theme.bgTertiary, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.border))
                .onChange(of: server.log.count) { proxy.scrollTo("end", anchor: .bottom) }
            }
        }
    }
}
