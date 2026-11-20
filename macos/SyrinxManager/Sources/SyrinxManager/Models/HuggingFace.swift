// Hugging Face's public API, for finding GGUF models: search (api/models), a repository's
// files with their sizes and SHA-256 (api/models/{repo}/tree/main), and the download address
// of a file. No account is needed; gated repositories are shown but can't be downloaded.
// The Windows app's copy is windows/SyrinxManager/Models/HuggingFaceClient.cs.
import Foundation

/// A repository found by a search. `architecture` is what Hugging Face read from its GGUF
/// files ("gemma4", "qwen35"…), nil when it gave none: the search keeps only the ones the
/// engine runs (`EngineSupport.runsArchitecture`).
struct HubModel: Identifiable, Hashable {
    let repository: String
    let downloads: Int64
    let likes: Int64
    let gated: Bool
    let architecture: String?
    let contextLength: Int64?

    var id: String { repository }
    var publisher: String { String(repository.split(separator: "/").first ?? "") }
    var name: String { String(repository.split(separator: "/").last ?? "") }
}

/// A GGUF file in a repository. Split models are one entry with every part.
struct HubFile: Identifiable, Hashable {
    let repository: String
    let path: String
    let size: Int64
    let sha256: String?
    var parts = 1

    var id: String { "\(repository)/\(path)" }
    var fileName: String { (path as NSString).lastPathComponent }
    var quantization: String { HuggingFace.quantization(of: fileName) }
    var isVisionProjector: Bool { fileName.range(of: "mmproj", options: .caseInsensitive) != nil }
}

struct HubError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum HuggingFace {
    static let base = URL(string: "https://huggingface.co/")!

    static let session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.httpAdditionalHeaders = ["User-Agent": "Syrinx/0.1"]
        configuration.timeoutIntervalForRequest = 60
        return URLSession(configuration: configuration)
    }()

    static func resolveURL(_ repository: String, _ path: String) -> URL {
        let escaped = path.split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "/?#"))) ?? String($0) }
            .joined(separator: "/")
        return URL(string: "\(repository)/resolve/main/\(escaped)?download=true", relativeTo: base)!.absoluteURL
    }

    static func pageURL(_ repository: String) -> URL {
        URL(string: repository, relativeTo: base)!.absoluteURL
    }

    /// GGUF repositories matching the words, most downloaded first, each with the
    /// architecture Hugging Face read from its GGUF files (`expand[]=gguf`; asking for one
    /// field means naming the others wanted too).
    static func search(_ query: String, limit: Int = 30) async throws -> [HubModel] {
        var components = URLComponents(url: base.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "search", value: query.trimmingCharacters(in: .whitespaces)),
            URLQueryItem(name: "filter", value: "gguf"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(limit)),
        ] + ["gguf", "downloads", "likes", "gated", "private", "lastModified"].map { URLQueryItem(name: "expand[]", value: $0) }
        let entries = try JSONDecoder().decode([SearchEntry].self, from: try await get(components.url!))
        return entries.compactMap { entry in
            guard let id = entry.id, !id.isEmpty, entry.private != true else { return nil }
            return HubModel(repository: id, downloads: entry.downloads ?? 0, likes: entry.likes ?? 0, gated: entry.gated?.isGated ?? false,
                            architecture: entry.gguf?.architecture, contextLength: entry.gguf?.context_length)
        }
    }

    /// The repository's GGUF files, split models joined into one entry.
    static func ggufFiles(_ repository: String) async throws -> [HubFile] {
        let url = base.appendingPathComponent("api/models/\(repository)/tree/main").appending(queryItems: [URLQueryItem(name: "recursive", value: "true")])
        let entries = try JSONDecoder().decode([TreeEntry].self, from: try await get(url))
        return ggufFiles(repository, entries)
    }

    private static func get(_ url: URL) async throws -> Data {
        let (data, response) = try await session.data(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw HubError(message: "Hugging Face answered \(http.statusCode).")
        }
        return data
    }

    static func ggufFiles(_ repository: String, _ entries: [TreeEntry]) -> [HubFile] {
        var files: [HubFile] = []
        var splits: [String: [TreeEntry]] = [:]
        for entry in entries {
            guard entry.type == "file", let path = entry.path, path.lowercased().hasSuffix(".gguf") else { continue }
            if let range = path.range(of: #"-\d{5}-of-\d{5}\.gguf$"#, options: [.regularExpression, .caseInsensitive]) {
                splits[String(path[..<range.lowerBound]), default: []].append(entry)
                continue
            }
            files.append(HubFile(repository: repository, path: path, size: entry.lfs?.size ?? entry.size ?? 0, sha256: entry.lfs?.oid))
        }
        for parts in splits.values {
            let first = parts.min { ($0.path ?? "") < ($1.path ?? "") }!
            let size = parts.reduce(Int64(0)) { $0 + ($1.lfs?.size ?? $1.size ?? 0) }
            files.append(HubFile(repository: repository, path: first.path!, size: size, sha256: first.lfs?.oid, parts: parts.count))
        }
        return files.sorted { ($0.isVisionProjector ? 1 : 0, $0.size) < ($1.isVisionProjector ? 1 : 0, $1.size) }
    }

    /// "Llama-3.2-1B-Instruct-Q4_K_M.gguf" → "Q4_K_M"; "gemma-4-E2B_q4_0-it.gguf" → "Q4_0".
    static func quantization(of fileName: String) -> String {
        var stem = ((fileName as NSString).deletingPathExtension)
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: ".", with: " ")
        // Underscores join a quantization's parts (Q4_K_M) but also separate words (E2B_q4_0),
        // so split words at underscores that precede a quantization.
        stem = stem.replacingOccurrences(of: #"_(?=(?:I?Q\d|BF16|F16|F32)(?:_|\b))"#, with: " ", options: [.regularExpression, .caseInsensitive])
        let pattern = try! NSRegularExpression(
            pattern: #"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d)(?![A-Za-z0-9])"#,
            options: .caseInsensitive)
        let matches = pattern.matches(in: stem, range: NSRange(stem.startIndex..., in: stem))
        guard let last = matches.last, let range = Range(last.range, in: stem) else { return "GGUF" }
        return stem[range].uppercased()
    }

    struct SearchEntry: Decodable {
        let id: String?
        let downloads: Int64?
        let likes: Int64?
        let `private`: Bool?
        /// false, or "auto"/"manual" when access must be requested.
        let gated: Gated?
        let gguf: GGUFSummary?
    }

    struct Gated: Decodable {
        let isGated: Bool

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let flag = try? container.decode(Bool.self) {
                isGated = flag
            } else {
                isGated = (try? container.decode(String.self)) != nil
            }
        }
    }

    /// What Hugging Face read from a repository's GGUF files.
    struct GGUFSummary: Decodable {
        let architecture: String?
        let context_length: Int64?
    }

    struct TreeEntry: Decodable {
        let type: String?
        let path: String?
        let size: Int64?
        let lfs: LFS?

        struct LFS: Decodable {
            /// The file's SHA-256, in hex.
            let oid: String?
            let size: Int64?
        }
    }
}
