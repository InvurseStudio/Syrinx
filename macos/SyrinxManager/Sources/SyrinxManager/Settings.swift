// What the app remembers between launches, in ~/Library/Application Support/Syrinx/
// manager.json (the folder the server uses on its own too, for chat caches): the model,
// where models are looked for, and how the server is started. The Windows app keeps the
// same file in %LOCALAPPDATA%\Syrinx.
import Foundation
import Security

@MainActor
final class Settings: ObservableObject {
    /// Syrinx's own folder, which the server uses on its own too (chat caches).
    nonisolated static var folder: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Syrinx", isDirectory: true)
    }

    nonisolated static var modelsFolder: URL { folder.appendingPathComponent("Models", isDirectory: true) }

    nonisolated static var defaultFile: URL { folder.appendingPathComponent("manager.json") }

    /// The port syrinx-server --standalone listens on by default.
    static let defaultPort = 8088

    /// The context lengths offered; 0 is automatic (as much as the GPU's memory holds).
    static let contextChoices = [0, 4096, 8192, 16384, 32768, 65536, 131072]

    @Published var model: String? { didSet { save() } }
    @Published var modelFolders: [String] = [] { didSet { save() } }
    @Published var port = defaultPort { didSet { save() } }
    /// Listen on every network (0.0.0.0), not only this Mac. Needs an API key.
    @Published var onNetwork = false { didSet { save() } }
    @Published var apiKey = "" { didSet { save() } }
    @Published var context = 0 { didSet { save() } }
    @Published var thinking = false { didSet { save() } }
    @Published var slots = 1 { didSet { save() } }
    @Published var startWithApp = false { didSet { save() } }

    private var loading = false
    /// Where they're saved: manager.json, or a file of a test's own.
    private let file: URL

    /// What's saved: the same names as the Windows app's manager.json.
    private struct Stored: Codable {
        var Model: String?
        var ModelFolders: [String]?
        var Port: Int?
        var OnNetwork: Bool?
        var ApiKey: String?
        var Context: Int?
        var Thinking: Bool?
        var Slots: Int?
        var StartWithApp: Bool?
    }

    init(file: URL = Settings.defaultFile) {
        self.file = file
        loading = true
        defer { loading = false }
        guard let data = try? Data(contentsOf: file),
              let stored = try? JSONDecoder().decode(Stored.self, from: data) else { return }
        model = stored.Model
        modelFolders = stored.ModelFolders ?? []
        port = stored.Port ?? Self.defaultPort
        onNetwork = stored.OnNetwork ?? false
        apiKey = stored.ApiKey ?? ""
        context = stored.Context ?? 0
        thinking = stored.Thinking ?? false
        slots = stored.Slots ?? 1
        startWithApp = stored.StartWithApp ?? false
    }

    private func save() {
        guard !loading else { return }
        let stored = Stored(Model: model, ModelFolders: modelFolders, Port: port, OnNetwork: onNetwork, ApiKey: apiKey,
                            Context: context, Thinking: thinking, Slots: slots, StartWithApp: startWithApp)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(stored).write(to: file, options: .atomic)
        } catch {
            // Kept for this session only.
        }
    }

    /// A new API key: 32 random characters, prefixed so it's recognisable.
    static func newApiKey() -> String { "sx-" + randomHex(16) }

    static func randomHex(_ count: Int) -> String {
        var bytes = [UInt8](repeating: 0, count: count)
        _ = SecRandomCopyBytes(kSecRandomDefault, count, &bytes)
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// Where models are looked for: Syrinx's own folder, the folders added, and the
    /// repository's models folder when the app runs from a Syrinx checkout.
    func allModelFolders() -> [URL] {
        var folders = [Self.modelsFolder] + modelFolders.map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let repository = ServerProcess.repositoryFolder() {
            folders.append(repository.appendingPathComponent("models", isDirectory: true))
        }
        var seen = Set<String>()
        return folders.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    /// The folders models are listed from besides Syrinx's own.
    func addedModelFolders() -> [URL] {
        allModelFolders().filter { !samePath($0.path, Self.modelsFolder.path) }
    }
}

/// Whether two paths name the same file (the Mac's disks are usually case-insensitive).
func samePath(_ a: String?, _ b: String?) -> Bool {
    guard let a, let b else { return false }
    return URL(fileURLWithPath: a).standardizedFileURL.path.lowercased()
        == URL(fileURLWithPath: b).standardizedFileURL.path.lowercased()
}

/// Whether `path` is `folder` or inside it.
func isInside(_ path: String, _ folder: String) -> Bool {
    let full = URL(fileURLWithPath: path).standardizedFileURL.path.lowercased()
    var root = URL(fileURLWithPath: folder).standardizedFileURL.path.lowercased()
    while root.count > 1, root.hasSuffix("/") { root.removeLast() }
    return full == root || full.hasPrefix(root + "/")
}
