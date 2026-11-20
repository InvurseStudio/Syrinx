// The Syrinx app for the Mac: one window with a sidebar of pages (Server, Models, Connect,
// Statistics) and the server's state at its foot, after the Windows app
// (windows/SyrinxManager). The server ends with the app: quitting, or closing the window,
// stops it once it has saved the chat in use.
import AppKit
import SwiftUI

@main
struct SyrinxManagerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var app = AppModel.shared

    var body: some Scene {
        Window("Syrinx", id: "main") {
            MainView()
                .environmentObject(app)
                .environmentObject(app.settings)
                .environmentObject(app.server)
                .environmentObject(app.downloads)
                .frame(minWidth: 820, minHeight: 560)
                .preferredColorScheme(.dark)
                .tint(Theme.accent)
        }
        .defaultSize(width: 1100, height: 760)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Run from a terminal (swift run), it's still an app with a Dock icon and a menu.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        Task { @MainActor in AppModel.shared.launched() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { AppModel.shared.server.stopNow() }
    }
}

/// The server the app runs, its settings, the models on this Mac and the downloads: one of
/// each for the app, shared by the pages.
@MainActor
final class AppModel: ObservableObject {
    static let shared = AppModel()

    enum PageName: String, CaseIterable, Identifiable {
        case server = "Server", models = "Models", connect = "Connect", statistics = "Statistics"

        var id: String { rawValue }

        var symbol: String {
            switch self {
            case .server: return "server.rack"
            case .models: return "cube"
            case .connect: return "link"
            case .statistics: return "chart.bar"
            }
        }
    }

    let settings = Settings()
    let server = ServerProcess()
    let library = ModelLibrary(modelsRoot: Settings.modelsFolder)
    let downloads = ModelDownloads(modelsRoot: Settings.modelsFolder, stateFile: ModelDownloads.defaultStateFile)

    @Published var page: PageName = .server
    /// The Models page's tab: Installed, or Discover.
    @Published var discover = false
    /// The GGUF files on this Mac, newest first.
    @Published private(set) var models: [InstalledModel] = []
    @Published private(set) var scanned = false

    private init() {
        downloads.onInstalled = { [weak self] item in
            Task { await self?.installed(item) }
        }
    }

    func launched() {
        Task { await scan() }
        if settings.startWithApp, ServerProcess.problem(settings) == nil {
            Task { await server.start(settings) }
        }
    }

    func scan() async {
        let folders = settings.addedModelFolders()
        let library = self.library
        models = await Task.detached { library.scan(folders) }.value
        scanned = true
        // The first model found becomes the one the server runs.
        if settings.model == nil, let first = models.first(where: { $0.isChat && $0.support.canRun }) {
            settings.model = first.path
        }
    }

    /// The first model downloaded becomes the one the server runs.
    private func installed(_ item: DownloadItem) async {
        let hadModel = settings.model.map { FileManager.default.fileExists(atPath: $0) } ?? false
        await scan()
        if !hadModel, let model = models.first(where: { samePath($0.path, item.destination) }), model.support.canRun {
            settings.model = model.path
        }
    }

    func start() {
        Task { await server.start(settings) }
    }

    func stop() {
        Task { await server.stop() }
    }
}
