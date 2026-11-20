// The GGUF models on this Mac: those downloaded into the models folder and those in folders
// the user added. Each file's header is read once (and again only if the file changes), so
// the list can say what each model is and whether the engine runs it.
import Foundation

struct InstalledModel: Identifiable, Hashable {
    let path: String
    let size: Int64
    let modified: Date
    let info: GGUFInfo?
    let problem: String?
    let downloaded: Bool

    var id: String { path }
    var fileName: String { (path as NSString).lastPathComponent }

    /// The model's own name from its header, or the file name when the header's says
    /// nothing (Google's Gemma 4 files are named "Hf").
    var title: String {
        if let name = info?.name, name.count >= 4 { return name }
        return (fileName as NSString).deletingPathExtension
    }

    var support: SupportVerdict {
        guard let info else { return SupportVerdict(level: .unsupported, summary: problem ?? "Couldn't read the file") }
        return EngineSupport.check(info)
    }

    /// Projectors aren't models to choose.
    var isChat: Bool { info.map { !$0.isVisionProjector } ?? true }

    /// "Gemma 4 · 4.6B parameters · Q6_K · 3.6 GB · 128K context"
    var details: String {
        guard let info else { return "\(fileName) · \(ModelFit.formatBytes(size))" }
        var parts = [EngineSupport.familyName(info.architecture)]
        if info.parameterCount > 0 { parts.append("\(ModelFit.formatParameters(info.parameterCount)) parameters") }
        parts.append(info.quantization)
        parts.append(ModelFit.formatBytes(size))
        if let context = info.contextLength { parts.append("\((context / 1024).formatted())K context") }
        return parts.joined(separator: " · ")
    }
}

final class ModelLibrary: @unchecked Sendable {
    /// LM Studio keeps models four levels down (models/publisher/repository/file).
    private static let maxDepth = 4

    let modelsRoot: URL
    private var cache: [String: InstalledModel] = [:]
    private let lock = NSLock()

    init(modelsRoot: URL) {
        self.modelsRoot = modelsRoot
    }

    /// Every GGUF file in the models folder and the added folders, newest first. Slow on a
    /// cold disk: call it off the main thread.
    func scan(_ addedFolders: [URL]) -> [InstalledModel] {
        var found: [InstalledModel] = []
        var seen = Set<String>()
        for (folder, downloaded) in [(modelsRoot, true)] + addedFolders.map({ ($0, false) }) {
            for file in Self.ggufFiles(in: folder) where seen.insert(file.lowercased()).inserted {
                found.append(describe(file, downloaded: downloaded))
            }
        }
        return found.sorted { $0.modified > $1.modified }
    }

    func describe(_ file: String, downloaded: Bool = false) -> InstalledModel {
        let attributes = try? FileManager.default.attributesOfItem(atPath: file)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        let modified = attributes?[.modificationDate] as? Date ?? .distantPast
        lock.lock()
        if let cached = cache[file], cached.size == size, cached.modified == modified, cached.downloaded == downloaded {
            lock.unlock()
            return cached
        }
        lock.unlock()
        var info: GGUFInfo?
        var problem: String?
        do {
            info = try GGUFReader.read(URL(fileURLWithPath: file))
        } catch let error as GGUFFormatError {
            problem = error.message
        } catch {
            problem = "Couldn't open the file: \(error.localizedDescription)"
        }
        let model = InstalledModel(path: file, size: size, modified: modified, info: info, problem: problem, downloaded: downloaded)
        lock.lock()
        cache[file] = model
        lock.unlock()
        return model
    }

    /// Deletes a downloaded model's file, and its folders if that leaves them empty.
    func delete(_ model: InstalledModel) throws {
        let manager = FileManager.default
        try manager.removeItem(atPath: model.path)
        lock.lock()
        cache[model.path] = nil
        lock.unlock()
        guard isInside(model.path, modelsRoot.path) else { return }
        var folder = (model.path as NSString).deletingLastPathComponent
        while isInside(folder, modelsRoot.path), !samePath(folder, modelsRoot.path),
              let entries = try? manager.contentsOfDirectory(atPath: folder),
              entries.allSatisfy({ $0 == ".DS_Store" }) {
            try? manager.removeItem(atPath: folder)
            folder = (folder as NSString).deletingLastPathComponent
        }
    }

    /// The GGUF files in a folder and up to four levels of subfolders. Files the engine
    /// made for its own tests (F32 copies) are left out.
    static func ggufFiles(in folder: URL) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: folder, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var files: [String] = []
        let rootDepth = folder.standardizedFileURL.pathComponents.count
        for case let url as URL in enumerator {
            if url.standardizedFileURL.pathComponents.count - rootDepth > maxDepth {
                enumerator.skipDescendants()
                continue
            }
            let name = url.lastPathComponent
            guard name.lowercased().hasSuffix(".gguf"), name.range(of: "-F32", options: .caseInsensitive) == nil else { continue }
            files.append(url.standardizedFileURL.path)
        }
        return files
    }
}
