import XCTest
@testable import SyrinxManager

final class GGUFTests: XCTestCase {
    private func gemma(tokenizer: String = "gemma4", types: [UInt32] = [14, 0]) -> GGUFWriter {
        var writer = GGUFWriter()
            .adding("general.architecture", "gemma4")
            .adding("general.name", "Gemma 4 E2B It")
            .adding("gemma4.context_length", UInt32(131_072))
            .adding("general.file_type", UInt32(18))
            .adding("tokenizer.ggml.model", tokenizer)
            .adding("tokenizer.ggml.tokens", ["<pad>", "<bos>", "hello"])
            .adding("tokenizer.ggml.token_type", [Int32(1), 3, 1])
        for (i, type) in types.enumerated() {
            writer = writer.tensor("blk.\(i).weight", type: type, 256, 4)
        }
        return writer
    }

    func testReadsHeaderAndSkipsArrays() throws {
        let info = try GGUFReader.read(gemma().data())
        XCTAssertEqual(info.architecture, "gemma4")
        XCTAssertEqual(info.name, "Gemma 4 E2B It")
        XCTAssertEqual(info.contextLength, 131_072)
        XCTAssertEqual(info.quantization, "Q6_K")
        XCTAssertEqual(info.tensors.count, 2)
        XCTAssertEqual(info.parameterCount, 2048)
        XCTAssertEqual(info.tensorTypes, ["Q6_K", "F32"])
        XCTAssertNil(info.metadata["tokenizer.ggml.tokens"])
    }

    func testReadsFromAFileInPieces() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("syrinx-test-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        // A tokenizer array larger than the reader's 1 MB buffer, so it seeks past it.
        let tokens = (0..<120_000).map { "token-\($0)" }
        let data = GGUFWriter().adding("general.architecture", "qwen35").adding("tokenizer.ggml.tokens", tokens)
            .adding("tokenizer.ggml.model", "gpt2").adding("tokenizer.ggml.pre", "qwen35")
            .tensor("output.weight", type: 12, 256, 8).data()
        try data.write(to: url)
        let info = try GGUFReader.read(url)
        XCTAssertEqual(info.architecture, "qwen35")
        XCTAssertEqual(info.text("tokenizer.ggml.pre"), "qwen35")
        XCTAssertEqual(info.tensors.first?.name, "output.weight")
    }

    func testTruncatedAndForeignFiles() {
        let data = gemma().data()
        XCTAssertThrowsError(try GGUFReader.read(data.prefix(data.count - 10))) { error in
            XCTAssertEqual((error as? GGUFFormatError)?.message, "The file ends inside its header: it may be incomplete.")
        }
        XCTAssertThrowsError(try GGUFReader.read(Data("PK\u{3}\u{4}not a model".utf8)))
    }

    func testEngineSupport() throws {
        XCTAssertEqual(EngineSupport.check(try GGUFReader.read(gemma().data())).level, .gpu)
        // BF16 is uploaded as another type, so it still runs on the GPU.
        XCTAssertEqual(EngineSupport.check(try GGUFReader.read(gemma(types: [14, 30]).data())).level, .gpu)
        let iq = EngineSupport.check(try GGUFReader.read(gemma(types: [16]).data()))
        XCTAssertEqual(iq.level, .unsupported)
        XCTAssertEqual(iq.summary, "Uses IQ2_XXS, which Syrinx can't read yet")
        let llama2 = EngineSupport.check(try GGUFReader.read(GGUFWriter().adding("general.architecture", "llama")
            .adding("tokenizer.ggml.model", "llama").data()))
        XCTAssertEqual(llama2.summary, "Its tokenizer (\"llama\") isn't supported yet")
        let mistral = EngineSupport.check(try GGUFReader.read(GGUFWriter().adding("general.architecture", "mistral3").data()))
        XCTAssertEqual(mistral.summary, "The mistral3 architecture isn't supported yet")
        let projector = EngineSupport.check(try GGUFReader.read(GGUFWriter().adding("general.architecture", "clip").data()))
        XCTAssertFalse(projector.canRun)
    }

    /// The models in the checkout's models folder, if there are any, read as the engine does.
    func testModelsInTheCheckout() throws {
        let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../../../models").standardizedFileURL
        let files = ModelLibrary.ggufFiles(in: models).filter { !$0.contains("/mlx/") }
        guard !files.isEmpty else { throw XCTSkip("No models in \(models.path)") }
        for file in files {
            let info = try GGUFReader.read(URL(fileURLWithPath: file))
            let verdict = EngineSupport.check(info)
            XCTAssertTrue(verdict.canRun || info.isVisionProjector, "\(file): \(verdict.summary)")
        }
    }
}

final class HuggingFaceTests: XCTestCase {
    func testQuantizationNames() {
        XCTAssertEqual(HuggingFace.quantization(of: "Llama-3.2-1B-Instruct-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(HuggingFace.quantization(of: "gemma-4-E2B_q4_0-it.gguf"), "Q4_0")
        XCTAssertEqual(HuggingFace.quantization(of: "Qwen3.5-4B-Q6_K.gguf"), "Q6_K")
        XCTAssertEqual(HuggingFace.quantization(of: "Qwen3.5-4B-BF16.gguf"), "BF16")
        XCTAssertEqual(HuggingFace.quantization(of: "model-IQ4_XS.gguf"), "IQ4_XS")
        XCTAssertEqual(HuggingFace.quantization(of: "model.gguf"), "GGUF")
    }

    func testSplitFilesAreOneEntry() throws {
        let json = """
        [{"type":"file","path":"a/big-Q8_0-00001-of-00002.gguf","size":10,"lfs":{"oid":"aa","size":3000}},
         {"type":"file","path":"a/big-Q8_0-00002-of-00002.gguf","size":10,"lfs":{"oid":"bb","size":2000}},
         {"type":"file","path":"small-Q4_K_M.gguf","size":10,"lfs":{"oid":"cc","size":1000}},
         {"type":"file","path":"mmproj-F16.gguf","size":10,"lfs":{"oid":"dd","size":500}},
         {"type":"file","path":"README.md","size":10},
         {"type":"directory","path":"a"}]
        """
        let entries = try JSONDecoder().decode([HuggingFace.TreeEntry].self, from: Data(json.utf8))
        let files = HuggingFace.ggufFiles("me/repo", entries)
        XCTAssertEqual(files.map(\.path), ["small-Q4_K_M.gguf", "a/big-Q8_0-00001-of-00002.gguf", "mmproj-F16.gguf"])
        XCTAssertEqual(files[1].parts, 2)
        XCTAssertEqual(files[1].size, 5000)
        XCTAssertEqual(files[0].sha256, "cc")
        XCTAssertTrue(files[2].isVisionProjector)
    }

    func testSearchEntries() throws {
        let json = """
        [{"id":"unsloth/Qwen3.5-4B-GGUF","downloads":1200,"likes":3,"gated":false,"private":false,
          "gguf":{"architecture":"qwen35","context_length":262144,"chat_template":"..."}},
         {"id":"google/gemma","downloads":5,"gated":"manual","gguf":{"architecture":"gemma4"}},
         {"id":"x/private","private":true}]
        """
        let entries = try JSONDecoder().decode([HuggingFace.SearchEntry].self, from: Data(json.utf8))
        XCTAssertEqual(entries[0].gguf?.context_length, 262_144)
        XCTAssertEqual(entries[0].gated?.isGated, false)
        XCTAssertEqual(entries[1].gated?.isGated, true)
        XCTAssertEqual(entries[2].private, true)
    }

    func testResolveURLEscapesEachPart() {
        XCTAssertEqual(HuggingFace.resolveURL("me/repo", "sub dir/a+b.gguf").absoluteString,
                       "https://huggingface.co/me/repo/resolve/main/sub%20dir/a+b.gguf?download=true")
    }

    @MainActor
    func testLeftOutLine() {
        let results = [
            HubModel(repository: "a/a", downloads: 1, likes: 0, gated: false, architecture: "qwen35", contextLength: nil),
            HubModel(repository: "b/b", downloads: 1, likes: 0, gated: false, architecture: "mistral3", contextLength: nil),
            HubModel(repository: "c/c", downloads: 1, likes: 0, gated: false, architecture: nil, contextLength: nil),
        ]
        XCTAssertEqual(DiscoverState.describe(results),
                       "1 repository Syrinx runs, most downloaded first. Open one to see its files. 2 more aren't shown: mistral3, unknown, which Syrinx doesn't run yet.")
        XCTAssertEqual(DiscoverState.describe([]), "No GGUF models match.")
    }
}

final class FitTests: XCTestCase {
    func testFit() {
        let gb: Int64 = 1 << 30
        let mac = MemoryBudget(gpuBytes: 24 * gb, systemBytes: 32 * gb)
        XCTAssertEqual(ModelFit.assess(4 * gb, mac).level, .gpu)
        XCTAssertEqual(ModelFit.assess(23 * gb + gb / 2, mac).level, .tooLarge)
        // A model that runs on the CPU only is measured against memory alone.
        XCTAssertEqual(ModelFit.assess(20 * gb, MemoryBudget(gpuBytes: 0, systemBytes: 32 * gb)).level, .systemMemory)
        XCTAssertEqual(ModelFit.assess(40 * gb, mac).level, .tooLarge)
        XCTAssertEqual(ModelFit.assess(nil, mac).level, .unknown)
    }

    func testFormatting() {
        XCTAssertEqual(ModelFit.formatBytes(3_349_516_256), "3.1 GB")
        XCTAssertEqual(ModelFit.formatBytes(532_517_120), "508 MB")
        XCTAssertEqual(ModelFit.formatParameters(1_235_814_400), "1.2B")
        XCTAssertEqual(ModelFit.formatParameters(4_000_000_000), "4B")
        XCTAssertEqual(ModelFit.formatParameters(800_000_000), "800M")
    }
}

@MainActor
final class ServerArgumentTests: XCTestCase {
    private var file: URL!

    override func setUp() {
        file = FileManager.default.temporaryDirectory.appendingPathComponent("syrinx-settings-\(UUID().uuidString).json")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: file)
    }

    func testSettingsAreSavedWithTheWindowsNames() throws {
        let settings = Settings(file: file)
        settings.port = 9001
        settings.modelFolders = ["/x"]
        let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
        XCTAssertEqual(saved?["Port"] as? Int, 9001)
        XCTAssertEqual(saved?["ModelFolders"] as? [String], ["/x"])
        XCTAssertEqual(Settings(file: file).port, 9001)
    }

    func testArguments() {
        let settings = Settings(file: file)
        settings.model = "/m/model.gguf"
        settings.port = 9000
        settings.context = 32768
        settings.thinking = true
        settings.slots = 4
        settings.onNetwork = true
        settings.apiKey = " sx-key "
        let args = ServerProcess.arguments(settings, token: "abc")
        XCTAssertEqual(args, ["--standalone", "/m/model.gguf", "--port", "9000", "--context", "32768", "--thinking", "on",
                              "--token", "abc", "--with-app", "--slots", "4", "--host", "0.0.0.0", "--api-key", "sx-key"])
        // The token changes each start; a restart is needed only when the settings do.
        XCTAssertFalse(ServerProcess.fingerprint(settings).contains("abc"))
    }

    func testNetworkNeedsAKey() {
        let settings = Settings(file: file)
        let model = FileManager.default.temporaryDirectory.appendingPathComponent("syrinx-args-\(UUID().uuidString).gguf")
        FileManager.default.createFile(atPath: model.path, contents: Data())
        defer { try? FileManager.default.removeItem(at: model) }
        settings.model = model.path
        settings.onNetwork = true
        settings.apiKey = ""
        XCTAssertEqual(ServerProcess.problem(settings), "Other devices can connect only with an API key. Make one under Network.")
        settings.model = nil
        XCTAssertEqual(ServerProcess.problem(settings), "Choose a model first.")
    }

    func testKeysArePrefixedHex() {
        let key = Settings.newApiKey()
        XCTAssertTrue(key.hasPrefix("sx-"))
        XCTAssertEqual(key.count, 35)
        XCTAssertNotEqual(key, Settings.newApiKey())
    }

    func testPaths() {
        XCTAssertTrue(samePath("/a/b/../c", "/A/c"))
        XCTAssertTrue(isInside("/models/x/y.gguf", "/models/"))
        XCTAssertFalse(isInside("/models2/y.gguf", "/models"))
    }
}
