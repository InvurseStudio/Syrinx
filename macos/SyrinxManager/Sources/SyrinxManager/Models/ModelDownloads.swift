// Downloads GGUF files from Hugging Face into the models folder. Each file downloads to
// name.gguf.part, resumes with an HTTP range request after a pause or a restart of the app,
// and is checked against Hugging Face's SHA-256 before it's moved into place: a file only
// counts as installed once it matches. Downloads in progress are listed in downloads.json,
// so they come back, paused, the next time the app starts. The Windows app's copy is
// windows/SyrinxManager/Models/ModelDownloads.cs.
import CryptoKit
import Foundation

enum DownloadState: String, Codable {
    case downloading = "Downloading"
    /// Checking the part already downloaded, before resuming.
    case preparing = "Preparing"
    case paused = "Paused"
    /// Downloaded; comparing its SHA-256 with Hugging Face's.
    case verifying = "Verifying"
    case failed = "Failed"
    case completed = "Completed"
}

struct DownloadItem: Identifiable, Codable, Equatable {
    let repository: String
    /// The file's path in the repository.
    let path: String
    let destination: String
    /// The size Hugging Face gives; 0 if it gave none.
    let size: Int64
    /// Hugging Face's SHA-256 of the file, in hex; nil if it gave none.
    let sha256: String?
    var state: DownloadState = .paused
    var error: String?

    var received: Int64 = 0
    /// Bytes a second, averaged over the last few seconds.
    var speed: Double = 0

    var id: String { "\(repository)/\(path)" }
    var fileName: String { (path as NSString).lastPathComponent }
    var partFile: String { destination + ".part" }
    var fraction: Double { size > 0 ? min(max(Double(received) / Double(size), 0), 1) : 0 }
    var isActive: Bool { state == .downloading || state == .preparing || state == .verifying }
    var remaining: TimeInterval? { speed > 0 && size > received ? Double(size - received) / speed : nil }

    // Saved with the Windows app's names; received and speed aren't saved.
    enum CodingKeys: String, CodingKey {
        case repository = "Repository", path = "Path", destination = "Destination", size = "Size"
        case sha256 = "Sha256", state = "State", error = "Error"
    }
}

@MainActor
final class ModelDownloads: ObservableObject {
    static var defaultStateFile: URL { Settings.folder.appendingPathComponent("downloads.json") }

    @Published private(set) var items: [DownloadItem] = []

    /// A file was verified and is in the models folder.
    var onInstalled: ((DownloadItem) -> Void)?

    private let modelsRoot: URL
    private let stateFile: URL
    private var workers: [String: DownloadWorker] = [:]
    /// Downloads cancelled while running: their part is thrown away once they stop.
    private var discarding = Set<String>()

    init(modelsRoot: URL, stateFile: URL) {
        self.modelsRoot = modelsRoot
        self.stateFile = stateFile
        load()
    }

    /// Where a repository's file goes: Models/publisher/repository/file.gguf.
    func destination(for repository: String, _ path: String) -> String {
        repository.split(separator: "/").reduce(modelsRoot) { $0.appendingPathComponent(String($1), isDirectory: true) }
            .appendingPathComponent((path as NSString).lastPathComponent).path
    }

    func find(_ repository: String, _ path: String) -> DownloadItem? {
        items.first { $0.repository == repository && $0.path == path }
    }

    /// Starts (or resumes) downloading a file.
    func start(_ file: HubFile) {
        if find(file.repository, file.path) == nil {
            items.append(DownloadItem(repository: file.repository, path: file.path,
                                      destination: destination(for: file.repository, file.path),
                                      size: file.size, sha256: file.sha256))
        }
        resume(file.id)
    }

    func resume(_ id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }), !items[index].isActive else { return }
        items[index].state = .preparing
        items[index].error = nil
        discarding.remove(id)
        save()
        let worker = DownloadWorker(item: items[index]) { [weak self] update in
            Task { @MainActor in self?.apply(update, for: id) }
        }
        workers[id] = worker
        worker.start()
    }

    /// Stops, keeping what's downloaded so far.
    func pause(_ id: String) {
        workers[id]?.cancel()
    }

    /// Stops and throws away what's downloaded.
    func cancel(_ id: String) {
        if let worker = workers[id] {
            discarding.insert(id)
            worker.cancel()
        } else if let item = items.first(where: { $0.id == id }) {
            discard(item)
        }
    }

    /// Removes a finished or failed download from the list.
    func dismiss(_ id: String) {
        guard let item = items.first(where: { $0.id == id }), !item.isActive else { return }
        if item.state != .completed { try? FileManager.default.removeItem(atPath: item.partFile) }
        items.removeAll { $0.id == id }
        save()
    }

    private func apply(_ update: DownloadWorker.Update, for id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        switch update {
        case .progress(let state, let received, let speed):
            items[index].state = state
            items[index].received = received
            items[index].speed = speed
        case .stopped:
            workers[id] = nil
            if discarding.remove(id) != nil {
                discard(items[index])
            } else {
                items[index].state = .paused
                items[index].speed = 0
                save()
            }
        case .failed(let message, let received):
            workers[id] = nil
            discarding.remove(id)
            items[index].state = .failed
            items[index].error = message
            items[index].received = received
            items[index].speed = 0
            save()
        case .completed:
            workers[id] = nil
            items[index].state = .completed
            items[index].received = items[index].size
            items[index].speed = 0
            save()
            onInstalled?(items[index])
        }
    }

    private func discard(_ item: DownloadItem) {
        try? FileManager.default.removeItem(atPath: item.partFile)
        items.removeAll { $0.id == item.id }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: stateFile),
              let saved = try? JSONDecoder().decode([DownloadItem].self, from: data) else { return }
        for var item in saved where item.state != .completed {
            // Whatever was running when the app closed waits to be resumed.
            if item.state != .failed { item.state = .paused }
            let attributes = try? FileManager.default.attributesOfItem(atPath: item.partFile)
            item.received = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
            items.append(item)
        }
    }

    private func save() {
        let unfinished = items.filter { $0.state != .completed }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        do {
            try FileManager.default.createDirectory(at: stateFile.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(unfinished).write(to: stateFile, options: .atomic)
        } catch {
            // Kept in memory; written again at the next change.
        }
    }
}

/// One file's download, off the main thread: hashes what's already on disk, asks for the
/// rest with a range request, writes and hashes it as it arrives, then checks the hash.
final class DownloadWorker: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    enum Update {
        case progress(DownloadState, received: Int64, speed: Double)
        case stopped
        case failed(String, received: Int64)
        case completed
    }

    private static let progressInterval: TimeInterval = 0.25

    private let item: DownloadItem
    private let report: (Update) -> Void
    private let queue = OperationQueue()
    private var session: URLSession?
    private var digest = SHA256()
    private var output: FileHandle?
    private var received: Int64 = 0
    private var cancelled = false
    private var lastReport = Date.distantPast
    private var window: [(time: Date, bytes: Int64)] = []
    private var speed: Double = 0
    private var failure: String?
    /// The server said the part already holds the whole file (a 416 to the range request).
    private var alreadyComplete = false
    private let lock = NSLock()

    init(item: DownloadItem, report: @escaping (Update) -> Void) {
        self.item = item
        self.report = report
        queue.maxConcurrentOperationCount = 1
    }

    func start() {
        queue.addOperation { self.prepare() }
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let session = self.session
        lock.unlock()
        session?.invalidateAndCancel()
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    /// Feeds the part already on disk to the hash, so the whole file is checked, then asks
    /// for the rest.
    private func prepare() {
        let manager = FileManager.default
        do {
            try manager.createDirectory(atPath: (item.destination as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            var existing = ((try? manager.attributesOfItem(atPath: item.partFile))?[.size] as? NSNumber)?.int64Value ?? 0
            if item.size > 0, existing > item.size {
                try? manager.removeItem(atPath: item.partFile)
                existing = 0
            }
            if existing > 0 {
                let part = try FileHandle(forReadingFrom: URL(fileURLWithPath: item.partFile))
                defer { try? part.close() }
                var done: Int64 = 0
                while done < existing {
                    if isCancelled { report(.stopped); return }
                    guard let chunk = try part.read(upToCount: Int(min(1 << 20, existing - done))), !chunk.isEmpty else { break }
                    digest.update(data: chunk)
                    done += Int64(chunk.count)
                    progress(.preparing, received: done)
                }
                existing = done
            }
            received = existing
            if item.size > 0, existing == item.size {
                finish()
                return
            }
            if isCancelled { report(.stopped); return }
            var request = URLRequest(url: HuggingFace.resolveURL(item.repository, item.path))
            request.setValue("Syrinx/0.1", forHTTPHeaderField: "User-Agent")
            if existing > 0 { request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range") }
            let configuration = URLSessionConfiguration.default
            // How long a download may receive nothing before it fails (and can be resumed).
            configuration.timeoutIntervalForRequest = 60
            configuration.timeoutIntervalForResource = 7 * 24 * 3600
            let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
            lock.lock()
            self.session = session
            let stop = cancelled
            lock.unlock()
            if stop { session.invalidateAndCancel(); report(.stopped); return }
            session.dataTask(with: request).resume()
        } catch {
            report(.failed(Self.describe(error), received: received))
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        let status = http.statusCode
        if status == 416, received > 0, item.size == 0 {
            alreadyComplete = true
            completionHandler(.cancel)
            return // didComplete finishes it
        }
        if status == 401 || status == 403 {
            failure = "Hugging Face asks for an account to download this model (it's gated)."
            completionHandler(.cancel)
            return
        }
        if status == 404 {
            failure = "The file is no longer on Hugging Face."
            completionHandler(.cancel)
            return
        }
        guard (200..<300).contains(status) else {
            failure = "Hugging Face answered \(status). Resume to try again."
            completionHandler(.cancel)
            return
        }
        let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
        let append = received > 0 && status == 206 && range.hasPrefix("bytes \(received)-")
        do {
            if !append {
                // The server sent the whole file instead of the rest: start over.
                digest = SHA256()
                received = 0
                FileManager.default.createFile(atPath: item.partFile, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: item.partFile))
            if append { try handle.seekToEnd() } else { try handle.truncate(atOffset: 0) }
            output = handle
            progress(.downloading, received: received, force: true)
            completionHandler(.allow)
        } catch {
            failure = Self.describe(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let output else { return }
        do {
            try output.write(contentsOf: data)
        } catch {
            failure = Self.describe(error)
            dataTask.cancel()
            return
        }
        digest.update(data: data)
        received += Int64(data.count)
        let now = Date()
        if now.timeIntervalSince(lastReport) > Self.progressInterval {
            window.append((now, received))
            while window.count > 1, now.timeIntervalSince(window[0].time) > 5 { window.removeFirst() }
            let seconds = now.timeIntervalSince(window[0].time)
            if seconds > 0.5 { speed = Double(received - window[0].bytes) / seconds }
            progress(.downloading, received: received)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? output?.synchronize()
        try? output?.close()
        output = nil
        session.finishTasksAndInvalidate()
        if let failure {
            report(.failed(failure, received: received))
            return
        }
        if isCancelled {
            report(.stopped)
            return
        }
        if alreadyComplete {
            finish()
            return
        }
        if let error {
            report(.failed(Self.describe(error), received: received))
            return
        }
        guard item.size == 0 || received == item.size else {
            report(.failed("The download ended early. Resume to try again.", received: received))
            return
        }
        finish()
    }

    private func finish() {
        progress(.verifying, received: received, force: true)
        let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
        if let expected = item.sha256, !expected.isEmpty, actual.lowercased() != expected.lowercased() {
            try? FileManager.default.removeItem(atPath: item.partFile)
            report(.failed("The downloaded file didn't match Hugging Face's checksum, so it was deleted. Try again.", received: 0))
            return
        }
        do {
            let manager = FileManager.default
            if manager.fileExists(atPath: item.destination) { try manager.removeItem(atPath: item.destination) }
            try manager.moveItem(atPath: item.partFile, toPath: item.destination)
            report(.completed)
        } catch {
            report(.failed(Self.describe(error), received: received))
        }
    }

    private func progress(_ state: DownloadState, received: Int64, force: Bool = false) {
        let now = Date()
        guard force || now.timeIntervalSince(lastReport) > Self.progressInterval else { return }
        lastReport = now
        report(.progress(state, received: received, speed: state == .downloading ? speed : 0))
    }

    private static func describe(_ error: Error) -> String {
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut: return "Hugging Face stopped responding. Resume to try again."
            case NSURLErrorNotConnectedToInternet, NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost, NSURLErrorNetworkConnectionLost:
                return "Couldn't reach Hugging Face. Check the internet connection, then resume."
            default: return "\(ns.localizedDescription) Resume to try again."
            }
        }
        if ns.domain == NSCocoaErrorDomain, ns.code == NSFileWriteOutOfSpaceError
            || (ns.underlyingErrors.first as NSError?)?.code == Int(ENOSPC) {
            return "The disk is full. Free some space, then resume."
        }
        if ns.domain == NSPOSIXErrorDomain, ns.code == Int(ENOSPC) { return "The disk is full. Free some space, then resume." }
        return error.localizedDescription
    }
}
