// The Models page, after the Windows app's (windows/SyrinxManager/Views/ModelsPage.cs): the
// model the server runs, downloads in progress, and two tabs: Installed (GGUF files in
// Syrinx's models folder and folders added, each with what its header says and whether
// Syrinx runs it) and Discover (small models of each family, and a search of Hugging Face's
// GGUF repositories). The search keeps only the repositories whose architecture Syrinx runs,
// as Hugging Face read it from their files, and says how many it left out.
import AppKit
import SwiftUI

struct ModelsPage: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess
    @EnvironmentObject private var downloads: ModelDownloads

    var body: some View {
        Page(title: "Models") {
            Caption("Syrinx runs GGUF models: Gemma 4, Qwen 3.5 and 3.8, Llama 3 and Muse Glimmer. "
                + "Download one below, or add a folder of models you already have.")
            VStack(alignment: .leading, spacing: 10) {
                SectionHeader("Model the server runs")
                ActiveModel()
            }
            if !downloads.items.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    SectionHeader("Downloads")
                    ForEach(downloads.items) { item in DownloadRow(item: item) }
                }
            }
            Picker("", selection: $app.discover) {
                Label("Installed", systemImage: "internaldrive").tag(false)
                Label("Discover", systemImage: "magnifyingglass").tag(true)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 260)
            if app.discover { DiscoverTab() } else { InstalledTab() }
        }
    }
}

// MARK: - Shared pieces

@MainActor
private func chooseModel(_ path: String?, _ settings: Settings) {
    settings.model = path
}

private struct SupportBadge: View {
    let level: SupportLevel
    let summary: String

    var body: some View {
        switch level {
        case .gpu: Badge(text: "Runs on GPU", tone: .success, symbol: "bolt.fill").help(summary)
        case .cpuOnly: Badge(text: "CPU only for now", tone: .neutral, symbol: "cpu").help(summary)
        case .unsupported: Badge(text: "Not supported yet", tone: .critical, symbol: "xmark.circle").help(summary)
        }
    }
}

/// Whether a model of this size fits, on the GPU if it runs there, else in memory.
private struct FitBadge: View {
    let size: Int64
    let level: SupportLevel

    var body: some View {
        var budget = HardwareInfo.budget
        if level == .cpuOnly { budget.gpuBytes = 0 }
        let fit = ModelFit.assess(size, budget)
        return Group {
            switch fit.level {
            case .gpu: Badge(text: "Fits in GPU memory", tone: .success)
            case .systemMemory:
                Badge(text: level == .cpuOnly ? "Fits in memory" : "Slower: exceeds GPU memory", tone: level == .cpuOnly ? .success : .caution)
            case .tooLarge: Badge(text: "Too large for this Mac", tone: .critical)
            case .unknown: Badge(text: fit.summary)
            }
        }
        .help(fit.summary)
    }
}

private struct ModelBadges: View {
    let model: InstalledModel
    let showLocation: Bool

    var body: some View {
        HStack(spacing: 6) {
            let support = model.support
            SupportBadge(level: support.level, summary: support.summary)
            if support.canRun { FitBadge(size: model.size, level: support.level) }
            if showLocation, !model.downloaded { Badge(text: "Added folder") }
        }
    }
}

// MARK: - The model the server runs

private struct ActiveModel: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess

    var body: some View {
        Card {
            if let path = settings.model, !path.isEmpty {
                if !FileManager.default.fileExists(atPath: path) {
                    ItemRow(symbol: "exclamationmark.triangle", title: "The chosen model is missing", details: path) {
                        Button("Clear") { chooseModel(nil, settings) }
                    }
                } else {
                    let model = app.models.first { samePath($0.path, path) } ?? app.library.describe(path)
                    ItemRow(symbol: "cube", title: model.title, details: model.details) {
                        buttons(path, model)
                    } extra: {
                        VStack(alignment: .leading, spacing: 8) {
                            ModelBadges(model: model, showLocation: false)
                            status(path)
                        }
                    }
                }
            } else {
                ItemRow(symbol: "cube", title: "No model chosen",
                        details: "Choose one under Installed, or download one under Discover. The first model you download is chosen for you.") {
                    EmptyView()
                }
            }
        }
    }

    private func isThis(_ path: String) -> Bool { samePath(server.model, path) }

    /// Whether the server is running this model.
    @ViewBuilder
    private func status(_ path: String) -> some View {
        HStack(spacing: 8) {
            switch server.state {
            case .loading where isThis(path):
                ProgressView().controlSize(.small)
                Caption("Loading…")
            case .running where isThis(path):
                Image(systemName: "checkmark").foregroundStyle(Theme.accent)
                Text("Serving on \(server.device ?? "this Mac") · port \(String(server.address?.port ?? settings.port))")
                    .font(.system(size: 12)).foregroundStyle(Theme.accent)
            case .stopping:
                Caption("Stopping…")
            case .failed where isThis(path):
                Text(server.error ?? "It couldn't start.").font(.system(size: 12)).foregroundStyle(Theme.danger).textSelection(.enabled)
            case .running, .loading:
                let running = server.model.map { (($0 as NSString).lastPathComponent as NSString).deletingPathExtension } ?? ""
                Caption("The server is running \(running). Start this one to switch.")
            default:
                Caption("Not running. Start it to serve this model to your chat apps.")
            }
        }
    }

    @ViewBuilder
    private func buttons(_ path: String, _ model: InstalledModel) -> some View {
        HStack(spacing: 8) {
            if isThis(path), server.state == .loading || server.state == .running {
                if server.state == .running { Button("Restart") { app.start() } }
                Button("Stop") { app.stop() }
            } else {
                Button(server.state == .failed && isThis(path) ? "Try Again" : "Start") { app.start() }
                    .buttonStyle(AccentButtonStyle())
                    .disabled(!model.support.canRun || server.state == .stopping)
            }
        }
    }
}

// MARK: - Downloads

private struct DownloadRow: View {
    @EnvironmentObject private var downloads: ModelDownloads
    let item: DownloadItem

    var body: some View {
        Card {
            ItemRow(symbol: "arrow.down.circle", title: item.fileName, details: item.repository) {
                HStack(spacing: 6) { buttons }
            } extra: {
                VStack(alignment: .leading, spacing: 4) {
                    if item.state == .preparing || item.state == .verifying {
                        ProgressView().progressViewStyle(.linear)
                    } else {
                        ProgressView(value: item.fraction).progressViewStyle(.linear)
                            .tint(item.state == .failed ? Theme.danger : item.state == .paused ? Theme.textSecondary : Theme.accent)
                    }
                    Text(status).font(.system(size: 12))
                        .foregroundStyle(item.state == .failed ? Theme.danger : Theme.textSecondary)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var status: String {
        let amount = item.size > 0
            ? "\(ModelFit.formatBytes(item.received)) of \(ModelFit.formatBytes(item.size))"
            : ModelFit.formatBytes(item.received)
        switch item.state {
        case .downloading:
            guard item.speed > 0 else { return "\(amount) · starting…" }
            return "\(amount) · \(ModelFit.formatBytes(Int64(item.speed)))/s" + (item.remaining.map { " · \(formatDuration($0)) left" } ?? "")
        case .preparing:
            return item.received > 0 ? "Checking the \(ModelFit.formatBytes(item.received)) already downloaded…" : "Connecting to Hugging Face…"
        case .verifying: return "Checking the file against Hugging Face's checksum…"
        case .paused: return "Paused · \(amount)"
        case .failed: return item.error ?? "Failed"
        case .completed: return "Done"
        }
    }

    @ViewBuilder
    private var buttons: some View {
        switch item.state {
        case .downloading, .preparing:
            iconButton("pause.fill", "Pause") { downloads.pause(item.id) }
            iconButton("xmark", "Cancel download") { downloads.cancel(item.id) }
        case .paused:
            iconButton("play.fill", "Resume") { downloads.resume(item.id) }
            iconButton("xmark", "Cancel download") { downloads.cancel(item.id) }
        case .failed:
            Button("Retry") { downloads.resume(item.id) }
            iconButton("xmark", "Remove") { downloads.dismiss(item.id) }
        case .verifying:
            EmptyView()
        case .completed:
            Badge(text: "Installed", tone: .success, symbol: "checkmark")
            iconButton("xmark", "Remove from the list") { downloads.dismiss(item.id) }
        }
    }

    private func iconButton(_ symbol: String, _ help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Image(systemName: symbol).frame(width: 16) }.help(help)
    }
}

// MARK: - Installed

private struct InstalledTab: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            let chatModels = app.models.filter(\.isChat)
            if !app.scanned {
                Card {
                    HStack(spacing: 12) {
                        ProgressView().controlSize(.small)
                        Caption("Reading the models on this Mac…")
                    }
                }
            } else if chatModels.isEmpty {
                Card {
                    ItemRow(symbol: "cube", title: "No models yet",
                            details: "Download a recommended model under Discover, or add a folder that already has GGUF files.") {
                        Button("Discover models") { app.discover = true }.buttonStyle(AccentButtonStyle())
                    }
                }
            } else {
                ForEach(chatModels) { model in InstalledCard(model: model) }
            }

            SectionHeader("Model folders").padding(.top, 14)
            Card {
                ItemRow(symbol: "folder", title: "Syrinx's models folder",
                        details: (app.library.modelsRoot.path as NSString).abbreviatingWithTildeInPath + "\nDownloads go here.") {
                    Button("Open") { openFolder(app.library.modelsRoot) }
                }
            }
            ForEach(settings.addedModelFolders(), id: \.self) { folder in
                let added = settings.modelFolders.contains { samePath($0, folder.path) }
                let name = folder.lastPathComponent.isEmpty ? folder.path : folder.lastPathComponent
                Card {
                    ItemRow(symbol: "folder", title: added ? name : "\(name) (this Syrinx checkout)",
                            details: (folder.path as NSString).abbreviatingWithTildeInPath) {
                        HStack(spacing: 6) {
                            Button("Open") { openFolder(folder) }
                            if added {
                                Button { removeFolder(folder) } label: { Image(systemName: "trash") }
                                    .help("Stop listing \(folder.path)")
                            }
                        }
                    }
                }
            }
            Card {
                ItemRow(symbol: "folder.badge.plus", title: "Add a folder of models",
                        details: "List GGUF files kept elsewhere, such as LM Studio's models folder, without copying them.") {
                    Button("Add Folder…") { addFolder() }
                }
            }
        }
    }

    private func removeFolder(_ folder: URL) {
        settings.modelFolders.removeAll { samePath($0, folder.path) }
        if let chosen = settings.model, isInside(chosen, folder.path) { settings.model = nil }
        Task { await app.scan() }
    }

    private func addFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Add"
        panel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        guard panel.runModal() == .OK, let folder = panel.url?.standardizedFileURL else { return }
        if settings.modelFolders.contains(where: { samePath($0, folder.path) }) || isInside(folder.path, app.library.modelsRoot.path) { return }
        settings.modelFolders.append(folder.path)
        Task { await app.scan() }
    }

    private func openFolder(_ folder: URL) {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(folder)
    }
}

private struct InstalledCard: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess
    let model: InstalledModel

    var body: some View {
        Card(padding: EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 12)) {
            ItemRow(symbol: "cube", title: model.title, details: model.details) {
                HStack(spacing: 6) {
                    if samePath(model.path, settings.model) {
                        Badge(text: "In use", tone: .accent, symbol: "checkmark")
                    } else {
                        Button("Use") { chooseModel(model.path, settings) }
                            .disabled(!model.support.canRun)
                            .help(model.support.canRun ? "Serve this model" : model.support.summary)
                    }
                    Menu {
                        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)]) }
                        Button("Copy Path") { copy(model.path) }
                        if model.downloaded {
                            Divider()
                            Button("Delete…", role: .destructive) { delete() }
                        }
                    } label: { Image(systemName: "ellipsis") }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("More for \(model.title)")
                }
            } extra: {
                ModelBadges(model: model, showLocation: true)
            }
        }
    }

    private func delete() {
        if server.isActive, samePath(server.model, model.path) {
            showError("The server is running this model. Stop it first.")
            return
        }
        guard confirm("Delete \(model.title)?",
                      "This deletes \(model.fileName) (\(ModelFit.formatBytes(model.size))) from this Mac. You can download it again later.",
                      button: "Delete", destructive: true) else { return }
        do {
            try app.library.delete(model)
        } catch {
            showError("\(model.fileName) couldn't be deleted: \(error.localizedDescription)")
            return
        }
        if samePath(model.path, settings.model) { settings.model = nil }
        Task { await app.scan() }
    }
}

// MARK: - Discover

/// Hugging Face's answers, kept while the app runs so going back to the page is instant.
@MainActor
final class DiscoverState: ObservableObject {
    static let shared = DiscoverState()

    /// Repositories asked for in a search, before those Syrinx can't run are left out.
    static let searchLimit = 60

    @Published var query = ""
    @Published private(set) var results: [HubModel]?
    @Published private(set) var status = ""
    @Published private(set) var searching = false
    @Published private(set) var files: [String: [HubFile]] = [:]
    private var searched = ""
    private var typing: Task<Void, Never>?

    func typed() {
        typing?.cancel()
        let query = self.query.trimmingCharacters(in: .whitespaces)
        guard query.count >= 3, query != searched else { return }
        // Searches once typing pauses.
        typing = Task {
            try? await Task.sleep(for: .milliseconds(700))
            guard !Task.isCancelled else { return }
            await search()
        }
    }

    func search() async {
        let query = self.query.trimmingCharacters(in: .whitespaces)
        guard query.count >= 2, !(query == searched && searching) else { return }
        typing?.cancel()
        searching = true
        searched = query
        status = "Searching…"
        results = nil
        defer { searching = false }
        do {
            let found = try await HuggingFace.search(query, limit: Self.searchLimit)
            guard query == searched else { return }
            results = found
            status = Self.describe(found)
        } catch {
            status = "Couldn't reach Hugging Face: \(error.localizedDescription)"
        }
    }

    var runnable: [HubModel] { (results ?? []).filter { EngineSupport.runsArchitecture($0.architecture) } }

    /// How many Syrinx runs, and a line on those left out and why.
    static func describe(_ results: [HubModel]) -> String {
        let runnable = results.filter { EngineSupport.runsArchitecture($0.architecture) }
        let left = results.filter { !EngineSupport.runsArchitecture($0.architecture) }
        var leftOut = ""
        if !left.isEmpty {
            var counts: [String: Int] = [:]
            for model in left { counts[model.architecture ?? "unknown", default: 0] += 1 }
            let kinds = counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.map(\.key)
            leftOut = " \(left.count) more \(left.count == 1 ? "isn't" : "aren't") shown: \(kinds.prefix(4).joined(separator: ", "))"
                + (kinds.count > 4 ? " and others" : "") + ", which Syrinx doesn't run yet."
        }
        if runnable.isEmpty {
            return results.isEmpty ? "No GGUF models match." : "None of the matching models are ones Syrinx runs." + leftOut
        }
        return "\(runnable.count) \(runnable.count == 1 ? "repository" : "repositories") Syrinx runs, most downloaded first. Open one to see its files." + leftOut
    }

    func files(for repository: String) async throws -> [HubFile] {
        if let cached = files[repository] { return cached }
        let loaded = try await HuggingFace.ggufFiles(repository)
        files[repository] = loaded
        return loaded
    }
}

private struct DiscoverTab: View {
    @EnvironmentObject private var app: AppModel
    @StateObject private var state = DiscoverState.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Caption("Small models of each family Syrinx runs, chosen for this Mac: \(HardwareInfo.gpuName) with "
                + "\(ModelFit.formatBytes(HardwareInfo.systemMemoryBytes)) of memory, "
                + "\(ModelFit.formatBytes(HardwareInfo.gpuMemoryBytes)) of it for the GPU.")
            ForEach(RecommendedModels.all) { model in RecommendedCard(model: model) }

            SectionHeader("Search Hugging Face").padding(.top, 14)
            Caption("Only models Syrinx runs are shown: Hugging Face reads each repository's architecture from its files.")
            HStack(spacing: 8) {
                TextField("Search GGUF models, for example \"qwen3.5\" or \"gemma 4\"", text: $state.query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await state.search() } }
                    .onChange(of: state.query) { state.typed() }
                Button { Task { await state.search() } } label: { Image(systemName: "magnifyingglass") }
                    .help("Search")
            }
            if !state.status.isEmpty { Caption(state.status) }
            ForEach(state.runnable) { model in ResultCard(model: model, state: state) }
        }
    }
}

/// The control for one file in Discover: Download, its progress, or Installed.
private struct DownloadAction: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var downloads: ModelDownloads
    let repository: String
    let path: String
    let download: () async -> Void
    @State private var working = false

    var body: some View {
        let fileName = (path as NSString).lastPathComponent
        let installed = app.models.first { $0.fileName.caseInsensitiveCompare(fileName) == .orderedSame }
        let item = downloads.find(repository, path)
        HStack(spacing: 8) {
            if let installed, !(item?.isActive ?? false) {
                Badge(text: "Installed", tone: .success, symbol: "checkmark")
                if !samePath(installed.path, settings.model), installed.support.canRun {
                    Button("Use") { chooseModel(installed.path, settings) }
                }
            } else if let item, item.state != .completed {
                Caption(item.state == .paused ? "Paused"
                        : item.state == .failed ? "Failed"
                        : item.state == .verifying ? "Verifying…"
                        : "\(Int(item.fraction * 100))%")
            } else {
                Button("Download") {
                    working = true
                    Task {
                        await download()
                        working = false
                    }
                }
                .disabled(working)
            }
        }
    }
}

@MainActor
private func startDownload(_ file: HubFile, _ app: AppModel) {
    if ModelFit.assess(file.size, HardwareInfo.budget).level == .tooLarge,
       !confirm("Download anyway?", "\(file.fileName) is \(ModelFit.formatBytes(file.size)), more than this Mac can run.", button: "Download") {
        return
    }
    let root = app.library.modelsRoot
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let free = (try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage ?? Int64.max
    if free < file.size + (1 << 30) {
        showError("\(file.fileName) needs \(ModelFit.formatBytes(file.size)), and the disk has \(ModelFit.formatBytes(free)) free. Free some space, then try again.")
        return
    }
    app.downloads.start(file)
}

private struct RecommendedCard: View {
    @EnvironmentObject private var app: AppModel
    let model: RecommendedModel

    var body: some View {
        Card {
            ItemRow(symbol: "cube", title: model.title, details: model.summary) {
                DownloadAction(repository: model.repository, path: model.fileName) { await download() }
            } extra: {
                HStack(spacing: 6) {
                    Badge(text: ModelFit.formatBytes(model.approximateSize))
                    SupportBadge(level: model.support,
                                 summary: model.support == .gpu ? "Runs on the GPU" : "Runs on the CPU; GPU support for this architecture is coming")
                    FitBadge(size: model.approximateSize, level: model.support)
                }
            }
        }
    }

    private func download() async {
        do {
            let files = try await DiscoverState.shared.files(for: model.repository)
            guard let file = files.first(where: { $0.path == model.fileName }) else {
                showError("\(model.fileName) is no longer in \(model.repository).")
                return
            }
            startDownload(file, app)
        } catch {
            showError("Couldn't reach Hugging Face: \(error.localizedDescription)")
        }
    }
}

private struct ResultCard: View {
    @EnvironmentObject private var app: AppModel
    let model: HubModel
    @ObservedObject var state: DiscoverState
    @State private var expanded = false
    @State private var loadError: String?

    var body: some View {
        Card(padding: EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14)) {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    expanded.toggle()
                    if expanded { Task { await load() } }
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold))
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                            .foregroundStyle(Theme.textSecondary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.name).font(.system(size: 14)).lineLimit(1).truncationMode(.tail)
                            Caption(model.publisher)
                        }
                        Spacer()
                        Badge(text: EngineSupport.familyName(model.architecture), tone: .accent)
                        Badge(text: "\(formatCount(model.downloads)) downloads", symbol: "arrow.down")
                        if model.gated { Badge(text: "Needs a Hugging Face account", tone: .caution, symbol: "lock") }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                if expanded { content.padding(.leading, 24) }
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if let loadError {
            Caption("Couldn't load the files: \(loadError)")
        } else if let files = state.files[model.repository] {
            FileList(model: model, files: files)
        } else {
            ProgressView().controlSize(.small)
        }
    }

    private func load() async {
        loadError = nil
        do {
            _ = try await state.files(for: model.repository)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

private struct FileList: View {
    @EnvironmentObject private var app: AppModel
    let model: HubModel
    let files: [HubFile]

    var body: some View {
        let models = files.filter { !$0.isVisionProjector }
        VStack(alignment: .leading, spacing: 10) {
            if models.isEmpty {
                Caption("No GGUF model files in this repository.")
            } else {
                Caption(note)
                let best = suggested(models)
                ForEach(models) { file in
                    HStack(alignment: .center, spacing: 12) {
                        VStack(alignment: .leading, spacing: 3) {
                            HStack(spacing: 8) {
                                Text(file.quantization).font(.system(size: 14, weight: .semibold))
                                if file == best { Badge(text: "Suggested", tone: .accent) }
                            }
                            Caption(file.path)
                            HStack(spacing: 6) {
                                Badge(text: ModelFit.formatBytes(file.size))
                                let likely = EngineSupport.quantizationLikelySupported(file.quantization)
                                if file.parts > 1 {
                                    Badge(text: "Split in \(file.parts) files: not supported yet", tone: .critical)
                                } else if !likely {
                                    Badge(text: "\(file.quantization) isn't supported yet", tone: .critical)
                                } else {
                                    FitBadge(size: file.size, level: .gpu)
                                }
                            }
                        }
                        Spacer()
                        // Only files Syrinx can run are offered.
                        if file.parts == 1, EngineSupport.quantizationLikelySupported(file.quantization), !model.gated {
                            DownloadAction(repository: file.repository, path: file.path) { startDownload(file, app) }
                        }
                    }
                }
            }
            Link("Open on Hugging Face", destination: HuggingFace.pageURL(model.repository)).font(.system(size: 12))
        }
    }

    private var note: String {
        var note = EngineSupport.familyName(model.architecture)
            + (model.contextLength.map { ", \(($0 / 1024).formatted())K context" } ?? "")
            + ". Whether Syrinx runs a file's quantization is a guess from its name until it's downloaded and its header is read."
        if let caveat = EngineSupport.searchCaveat(model.architecture) { note += " \(caveat)." }
        return note
    }

    /// The largest quantized file that fits the GPU: past 8 bits, a bigger file is slower
    /// without being noticeably better. Unquantized files only when there's nothing else.
    private func suggested(_ files: [HubFile]) -> HubFile? {
        let budget = HardwareInfo.budget
        let candidates = files.filter {
            $0.parts == 1 && EngineSupport.quantizationLikelySupported($0.quantization) && ModelFit.assess($0.size, budget).level == .gpu
        }
        return candidates.filter { !["F16", "BF16", "F32"].contains($0.quantization) }.max { $0.size < $1.size }
            ?? candidates.min { $0.size < $1.size }
    }
}

private func formatCount(_ count: Int64) -> String {
    switch count {
    case 1_000_000...: return String(format: "%.1fM", Double(count) / 1e6).replacingOccurrences(of: ".0M", with: "M")
    case 1_000...: return String(format: "%.1fK", Double(count) / 1e3).replacingOccurrences(of: ".0K", with: "K")
    default: return "\(count)"
    }
}
