// Starts and stops syrinx-server --standalone with the chosen model, as the Vireo Mac app
// starts it for its chats (Vireo's Mac/Sources/VireoMac/Networking/LocalServerManager.swift).
// The server prints what it runs ("Syrinx: <model> on <GPU>, <n> tokens of context") and,
// once the model is in, where it listens ("Listening on http://127.0.0.1:N"). It accepts the
// token on its command line (--token) for its statistics, and stops when the app does: it
// watches its standard input (--with-app), which the app holds open while it runs, so it
// ends even if the app crashes and never holds the GPU's memory after. The Windows app's
// copy is windows/SyrinxManager/ServerProcess.cs.
import Foundation

enum ServerState: Equatable {
    case stopped
    /// The process is up and reading the model.
    case loading
    case running
    case stopping
    /// It couldn't start, or stopped on its own; `ServerProcess.error` says why.
    case failed
}

@MainActor
final class ServerProcess: ObservableObject {
    nonisolated static let program = "syrinx-server"

    private static var pidFile: URL { Settings.folder.appendingPathComponent("server.pid") }
    private static let loadTimeout: TimeInterval = 10 * 60
    private nonisolated static let stopGrace: TimeInterval = 5
    private static let logLines = 300

    @Published private(set) var state: ServerState = .stopped
    @Published private(set) var error: String?
    /// The model file being run.
    @Published private(set) var model: String?
    /// What the server says it runs on: "Apple M5 Max, 8-bit".
    @Published private(set) var device: String?
    @Published private(set) var context: Int?
    /// The address it listens on, once the model is in: http://127.0.0.1:8088.
    @Published private(set) var address: URL?
    /// The token this app started it with, for its statistics.
    private(set) var token = ""
    /// The settings it was started with, to tell when a change needs a restart.
    @Published private(set) var startedWith: String?
    /// The last lines it printed.
    @Published private(set) var log: [String] = []

    private var process: Process?
    private var input: Pipe?
    private var pending = Data()

    var isActive: Bool { state == .loading || state == .running || state == .stopping }

    var recentOutput: String { log.joined(separator: "\n") }

    /// syrinx-server: beside the app (Syrinx.app/Contents/MacOS), or, run from a Syrinx
    /// checkout, the workspace's release build.
    nonisolated static func findExecutable() -> URL? {
        let manager = FileManager.default
        if let beside = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent()
            .appendingPathComponent(program), manager.isExecutableFile(atPath: beside.path) {
            return beside
        }
        if let repository = repositoryFolder() {
            let built = repository.appendingPathComponent("target/release/\(program)")
            if manager.isExecutableFile(atPath: built.path) { return built }
        }
        return nil
    }

    /// The Syrinx checkout the app runs from, if it does: the folder above it with the
    /// workspace's Cargo.toml.
    nonisolated static func repositoryFolder() -> URL? {
        guard var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return nil }
        let manager = FileManager.default
        while folder.path != "/" {
            if manager.fileExists(atPath: folder.appendingPathComponent("Cargo.toml").path),
               manager.fileExists(atPath: folder.appendingPathComponent("crates/syrinx-server").path) {
                return folder
            }
            folder = folder.deletingLastPathComponent()
        }
        return nil
    }

    /// The command line for the settings: what `startedWith` compares.
    static func arguments(_ settings: Settings, token: String) -> [String] {
        var args = [
            "--standalone", settings.model ?? "",
            "--port", String(settings.port),
            "--context", settings.context > 0 ? String(settings.context) : "auto",
            "--thinking", settings.thinking ? "on" : "off",
            "--token", token,
            "--with-app",
        ]
        if settings.slots > 1 { args += ["--slots", String(settings.slots)] }
        if settings.onNetwork { args += ["--host", "0.0.0.0"] }
        let key = settings.apiKey.trimmingCharacters(in: .whitespaces)
        if !key.isEmpty { args += ["--api-key", key] }
        return args
    }

    /// What a restart would change: the settings' command line without the token.
    static func fingerprint(_ settings: Settings) -> String { arguments(settings, token: "").joined(separator: " ") }

    /// Why the settings can't start a server, if they can't.
    static func problem(_ settings: Settings) -> String? {
        guard let model = settings.model, !model.isEmpty else { return "Choose a model first." }
        if !FileManager.default.fileExists(atPath: model) { return "The model file is missing: \(model)" }
        if settings.onNetwork, settings.apiKey.trimmingCharacters(in: .whitespaces).isEmpty {
            return "Other devices can connect only with an API key. Make one under Network."
        }
        if findExecutable() == nil {
            return "\(program) isn't beside the app. Build it in the Syrinx folder with: cargo build --release -p syrinx-server"
        }
        return nil
    }

    func start(_ settings: Settings) async {
        await stop()
        Self.killLeftover()
        log = []
        pending = Data()
        error = nil
        device = nil
        context = nil
        address = nil
        model = settings.model
        if let problem = Self.problem(settings) {
            fail(problem)
            return
        }

        token = Settings.randomHex(16)
        startedWith = Self.fingerprint(settings)
        let executable = Self.findExecutable()!
        let process = Process()
        process.executableURL = executable
        process.arguments = Self.arguments(settings, token: token)
        process.currentDirectoryURL = executable.deletingLastPathComponent()
        let input = Pipe()
        let output = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = output
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                // End of output: stop the handler, or it keeps firing with empty reads.
                handle.readabilityHandler = nil
                return
            }
            Task { @MainActor in self?.received(data, from: process) }
        }
        process.terminationHandler = { [weak self] ended in
            Task { @MainActor in self?.exited(ended) }
        }
        self.process = process
        self.input = input
        state = .loading
        do {
            try process.run()
        } catch {
            self.process = nil
            self.input = nil
            fail("Couldn't start \(Self.program): \(error.localizedDescription)")
            return
        }
        try? FileManager.default.createDirectory(at: Settings.folder, withIntermediateDirectories: true)
        try? String(process.processIdentifier).write(to: Self.pidFile, atomically: true, encoding: .utf8)
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.loadTimeout))
            guard let self, self.process === process, self.state == .loading else { return }
            self.stopNow()
            self.fail("The model took too long to load.")
        }
    }

    /// Closing its standard input asks it to end, once it has saved the chat in use; if
    /// that takes too long, it's ended anyway.
    func stop() async {
        guard let process else { return }
        state = .stopping
        let input = self.input
        self.process = nil
        self.input = nil
        await Task.detached { Self.endGently(process, input: input) }.value
        Self.removePidFile()
        address = nil
        state = .stopped
    }

    /// Ends the server as the app quits, still giving it a moment to save.
    func stopNow() {
        guard let process else { return }
        let input = self.input
        self.process = nil
        self.input = nil
        Self.endGently(process, input: input)
        Self.removePidFile()
    }

    private nonisolated static func endGently(_ process: Process, input: Pipe?) {
        try? input?.fileHandleForWriting.close()
        let deadline = Date().addingTimeInterval(stopGrace)
        while process.isRunning, Date() < deadline { usleep(50_000) }
        if process.isRunning {
            process.terminate()
            let killAt = Date().addingTimeInterval(2)
            while process.isRunning, Date() < killAt { usleep(50_000) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
    }

    private func received(_ data: Data, from process: Process) {
        guard process === self.process else { return }
        pending.append(data)
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = String(decoding: pending[pending.startIndex..<newline], as: UTF8.self)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            pending.removeSubrange(pending.startIndex...newline)
            handle(line)
        }
    }

    private func handle(_ line: String) {
        log.append(line)
        if log.count > Self.logLines { log.removeFirst(log.count - Self.logLines) }
        if let match = line.wholeMatch(of: #/Syrinx: (.+) on (.+), (\d+) tokens of context/#) {
            device = String(match.2).trimmingCharacters(in: .whitespaces)
            context = Int(match.3)
        }
        if let match = line.firstMatch(of: #/Listening on (http:\/\/\S+)/#), let url = URL(string: String(match.1)) {
            address = url
            state = .running
        }
    }

    private func exited(_ ended: Process) {
        guard ended === process else { return } // stopped on purpose
        process = nil
        input = nil
        Self.removePidFile()
        guard state == .loading || state == .running else { return }
        if !pending.isEmpty {
            handle(String(decoding: pending, as: UTF8.self))
            pending = Data()
        }
        let last = log.last { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? ""
        fail(state == .loading
            ? "The server stopped while loading the model. \(last)".trimmingCharacters(in: .whitespaces)
            : "The server stopped unexpectedly (exit code \(ended.terminationStatus)). \(last)".trimmingCharacters(in: .whitespaces))
    }

    private func fail(_ message: String) {
        error = message
        address = nil
        state = .failed
    }

    private static func removePidFile() {
        try? FileManager.default.removeItem(at: pidFile)
    }

    /// A server left running by an app that was force-quit before it noticed would hold the
    /// GPU's memory: end it if it's still syrinx-server.
    private static func killLeftover() {
        defer { removePidFile() }
        guard let text = try? String(contentsOf: pidFile, encoding: .utf8),
              let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 else { return }
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return }
        let path = String(cString: buffer)
        guard (path as NSString).lastPathComponent == program else { return }
        kill(pid, SIGTERM)
        for _ in 0..<40 where kill(pid, 0) == 0 { usleep(50_000) }
        if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
    }
}
