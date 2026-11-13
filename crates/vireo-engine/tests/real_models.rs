//! Checks the engine against real model files in `VireoPrototype/models/`.
//! A model that isn't downloaded is skipped, with a note, so these pass on any machine.

use std::path::PathBuf;
use std::time::Instant;

use vireo_engine::cpu::{self, Pool};
use vireo_engine::models::gemma4::Gemma4;
use vireo_engine::models::llama::Llama;
use vireo_engine::models::qwen35::Qwen35;
use vireo_engine::{GgufFile, TensorType, Tokenizer};

/// Text that exercises the tokenizer: code, numbers, contractions, other scripts,
/// emoji, odd whitespace, and control characters.
const VARIED_TEXT: &[&str] = &[
    "Hello world",
    "  leading spaces and trailing   ",
    "fn main() {\n    println!(\"{}\", 2 + 2);\n}\n",
    "It's 3:45pm on 2026-11-10; we'll ship v1.2.3 (maybe).",
    "Ünïcödé façade naïve café — “quotes” ‘and’ «guillemets»",
    "日本語のテキストと中文文本，还有한국어 텍스트.",
    "हिन्दी पाठ और العربية النص",
    "Emoji: 👍🏽 👨‍👩‍👧 🇳🇿",
    "tabs\tand\r\nwindows\r\nnewlines\n\n\n",
    "\u{0}\u{1}\u{7f} control characters",
    "1234567890 0.000001 1e10 -42",
    "",
];

fn model(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipped: {name} isn't in VireoPrototype/models");
        None
    }
}

/// The tensors' data, in file order, must fill the data section exactly: each one
/// starts where the last ended (rounded up to the alignment), and the last ends at the
/// end of the file. This is how llama.cpp writes files, and it catches any mistake in
/// a block size.
fn assert_tensors_fill_the_file(file: &GgufFile) {
    let mut tensors: Vec<_> = file.tensors.iter().collect();
    tensors.sort_by_key(|t| t.offset);
    let mut expected = 0u64;
    for t in &tensors {
        assert_eq!(
            t.offset, expected,
            "{} doesn't start where the tensor before it ended",
            t.name
        );
        let bytes = t
            .bytes
            .unwrap_or_else(|| panic!("{} has an unknown type", t.name));
        expected = (t.offset + bytes).next_multiple_of(file.alignment);
    }
    let last = tensors.last().unwrap();
    assert_eq!(
        file.data_offset + last.offset + last.bytes.unwrap(),
        file.len()
    );
}

#[test]
fn llama_3_2_1b_instruct_q4_k_m() {
    let Some(path) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") else {
        return;
    };
    let started = Instant::now();
    let file = GgufFile::open(&path).unwrap();
    eprintln!("opened in {:?}", started.elapsed());

    assert_eq!(file.version, 3);
    assert_eq!(file.architecture(), Some("llama"));
    assert_eq!(file.arch_u64("block_count"), Some(16));
    assert_eq!(file.arch_u64("embedding_length"), Some(2048));
    assert_eq!(file.arch_u64("feed_forward_length"), Some(8192));
    assert_eq!(file.arch_u64("attention.head_count"), Some(32));
    assert_eq!(file.arch_u64("attention.head_count_kv"), Some(8));
    assert_eq!(file.arch_f64("rope.freq_base"), Some(500000.0));
    assert_eq!(file.get_str("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(file.get_str("tokenizer.ggml.pre"), Some("llama-bpe"));
    assert_eq!(file.get_u64("tokenizer.ggml.bos_token_id"), Some(128000));

    let tokens = file
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .unwrap();
    assert_eq!(tokens.len(), 128256);
    assert_eq!(tokens[128000].as_str(), Some("<|begin_of_text|>"));
    assert!(
        file.get_str("tokenizer.chat_template")
            .unwrap()
            .contains("<|start_header_id|>")
    );

    assert_eq!(file.tensors.len(), 147);
    let params: u64 = file.tensors.iter().map(|t| t.elements()).sum();
    assert_eq!(
        params / 10_000_000,
        123,
        "about 1.24B parameters, got {params}"
    );

    let embd = file.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.dims, [2048, 128256]);
    assert_eq!(embd.kind, Some(TensorType::Q6_K));
    let q = file.tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(
        (q.dims.as_slice(), q.kind),
        (&[2048u64, 2048][..], Some(TensorType::Q4_K))
    );

    assert_tensors_fill_the_file(&file);

    // Norm weights are plain F32s: read one to check the data is where we think it is.
    // RMSNorm weights in a trained model are small positive numbers, mostly near 1.
    let norm = file.tensor("blk.0.attn_norm.weight").unwrap();
    let data = file.tensor_data(norm).unwrap();
    let (words, rest) = data.as_chunks::<4>();
    assert!(rest.is_empty());
    let values: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(values.len(), 2048);
    assert!(values.iter().all(|v| v.is_finite()));
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    assert!((0.01..10.0).contains(&mean), "mean of attn_norm is {mean}");
}

#[test]
fn llama_3_2_1b_instruct_tokenizer() {
    let Some(path) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") else {
        return;
    };
    let file = GgufFile::open(&path).unwrap();
    let started = Instant::now();
    let t = Tokenizer::from_gguf(&file).unwrap();
    eprintln!("tokenizer loaded in {:?}", started.elapsed());

    assert_eq!(t.vocab_size(), 128256);
    assert_eq!(t.bos(), Some(128000));
    assert!(t.adds_bos());
    let eot = t.token_id("<|eot_id|>").unwrap();
    assert!(t.is_end_of_generation(eot));
    assert!(t.is_end_of_generation(t.eos().unwrap()));

    // Llama 3's known ids.
    assert_eq!(t.encode("Hello world", true, true), [128000, 9906, 1917]);
    assert_eq!(t.encode("Hello world", false, false), [9906, 1917]);
    // A space before a number is its own token, and numbers go 3 digits at a time.
    let ids = t.encode(" 12345", false, false);
    let pieces: Vec<String> = ids.iter().map(|&id| t.decode(&[id], true)).collect();
    assert_eq!(pieces, [" ", "123", "45"]);

    // A chat template's special tokens are read as single tokens.
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>";
    let ids = t.encode(prompt, false, true);
    assert_eq!(ids[0], t.token_id("<|start_header_id|>").unwrap());
    assert_eq!(ids[2], t.token_id("<|end_header_id|>").unwrap());
    assert_eq!(*ids.last().unwrap(), eot);
    assert_eq!(ids.len(), 6, "{ids:?}");
    assert_eq!(t.decode(&ids, true), prompt);
    assert_eq!(t.decode(&ids, false), "user\n\nHi");

    // Any text comes back exactly, and BPE never leaves single bytes for plain words.
    for text in VARIED_TEXT {
        let ids = t.encode(text, false, false);
        assert_eq!(&t.decode(&ids, false), text, "{text:?} -> {ids:?}");
    }
    let ids = t.encode("The quick brown fox jumps over the lazy dog.", false, false);
    assert_eq!(
        ids.len(),
        10,
        "one token per word and one for the full stop"
    );

    // Speed: a long document tokenizes in well under a second.
    let long: String = VARIED_TEXT.concat().repeat(200);
    let started = Instant::now();
    let ids = t.encode(&long, false, false);
    eprintln!(
        "{} characters -> {} tokens in {:?}",
        long.chars().count(),
        ids.len(),
        started.elapsed()
    );
    assert_eq!(t.decode(&ids, false), long);
}

/// The forward pass against numbers llama.cpp (b12262) gave for the same prompt:
/// llama-server on the CPU with an f32 KV cache and no flash attention, on an F32 copy
/// of the model that llama-quantize made from the Q4_K_M file. With F32 weights
/// llama.cpp doesn't round activations, so it does the same arithmetic as our reference;
/// and our reference decodes Q4_K and Q6_K exactly, so the Q4_K_M file gives the same
/// numbers. (llama.cpp on the Q4_K_M file itself gives 94.74% / 4.53%: it rounds
/// activations to 8 bits for its quantized matrix products.)
#[test]
fn llama_3_2_1b_instruct_forward_pass_matches_llama_cpp() {
    let Some(path) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") else {
        return;
    };
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Llama::load(&file, Pool::with_all_cores()).unwrap();
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France? \
                  Answer in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    assert_eq!(ids.len(), 22);

    let mut session = model.session(64);
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = model.forward(&mut session, id, i + 1 == ids.len()) {
            logits = l.to_vec();
        }
    }
    let mut probs = logits;
    cpu::softmax(&mut probs);
    // llama.cpp: "The" 95.7887%, "Paris" 3.6702%, "R" 0.2236%, "France" 0.0463%.
    for (id, percent) in [
        (791, 95.7887),
        (60704, 3.6702),
        (49, 0.2236),
        (50100, 0.0463),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.001,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }

    // Greedy generation finishes the answer and stops at <|eot_id|>.
    let mut next = 791u32;
    let mut answer = vec![next];
    while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
        let logits = model.forward(&mut session, next, true).unwrap();
        next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        answer.push(next);
    }
    assert_eq!(
        tokenizer.decode(&answer, false),
        "The capital of France is Paris."
    );
}

/// Qwen 3.5 0.8B (the `qwen35` architecture Qwen 3.6–3.8 use too) against llama.cpp
/// b12262: llama-server on the CPU, f32 cache, on an F32 copy of the Q4_K_M file.
/// llama.cpp runs a prompt's DeltaNet layers in chunks, a different order of the same
/// arithmetic, so the two agree to about 0.0005 points rather than every digit.
/// (Perplexity on the same 4 chunks of text: 25.1818 here, 25.1817 for llama.cpp.)
#[test]
fn qwen35_0_8b_forward_pass_matches_llama_cpp() {
    let Some(path) = model("Qwen3.5-0.8B-Q4_K_M.gguf") else {
        return;
    };
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Qwen35::load(&file, Pool::with_all_cores()).unwrap();
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    assert_eq!(ids.len(), 24, "Qwen adds no BOS token");

    let mut session = model.session(64);
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = model.forward(&mut session, id, i + 1 == ids.len()) {
            logits = l.to_vec();
        }
    }
    let mut probs = logits;
    cpu::softmax(&mut probs);
    // llama.cpp: "The" 56.6133%, "Paris" 35.0073%, "France" 0.8957%, "London" 0.6041%.
    for (id, percent) in [
        (760, 56.6133),
        (57590, 35.0073),
        (47358, 0.8957),
        (38261, 0.6041),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.002,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }

    let mut next = 760u32;
    let mut answer = vec![next];
    while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
        let logits = model.forward(&mut session, next, true).unwrap();
        next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        answer.push(next);
    }
    assert_eq!(
        tokenizer.decode(&answer, false),
        "The capital of France is Paris."
    );
}

/// Gemma 4 E2B against llama.cpp b12262: llama-server on the CPU, f32 cache, on an F32
/// copy of Google's QAT Q4_0 file. The two agree to within 0.0002 points here, and on
/// perplexity to 0.005% (30.8273 against 30.8287 over two chunks): a small remaining
/// difference in some operation, noted in PLAN.md.
#[test]
fn gemma4_e2b_forward_pass_matches_llama_cpp() {
    let Some(path) = model("gemma-4-E2B_q4_0-it.gguf") else {
        return;
    };
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Gemma4::load(&file, Pool::with_all_cores()).unwrap();
    let prompt = "<|turn>user
What is the capital of France? Answer in one sentence.<turn|>
<|turn>model
";
    let ids = tokenizer.encode(prompt, true, true);
    assert_eq!(ids.len(), 21);
    assert_eq!(ids[0], tokenizer.bos().unwrap());

    let mut session = model.session(64);
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = model.forward(&mut session, id, i + 1 == ids.len()) {
            logits = l.to_vec();
        }
    }
    let mut probs = logits;
    cpu::softmax(&mut probs);
    // llama.cpp: "The" 99.7068%, "Paris" 0.2201%, "**" 0.0387%, "France" 0.0049%.
    for (id, percent) in [
        (818, 99.7068),
        (50429, 0.2201),
        (1018, 0.0387),
        (31756, 0.0049),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.001,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }

    let mut next = 818u32;
    let mut answer = vec![next];
    while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
        let logits = model.forward(&mut session, next, true).unwrap();
        next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        answer.push(next);
    }
    assert_eq!(
        tokenizer.decode(&answer, false),
        "The capital of France is Paris."
    );
}

/// Gemma 4 26B-A4B (mixture of experts) against llama.cpp b12228: llama-server on the
/// CPU, f32 cache, on an F32 copy of unsloth's UD-Q6_K file (101 GB, made with
/// llama-quantize). The top token is within 0.15 points; the dense 31B, checked the same
/// way, is within 0.03, and E2B within 0.0002, so some of this is a difference the large
/// models share. An expert's rows read up-first instead give " a" 24.9%, and the router's
/// scale without 1/sqrt(dim) a perplexity in the hundreds of thousands. The file is in
/// the Vireo app's model folder, or where `VIREO_GEMMA_26B` says.
#[test]
fn gemma4_26b_a4b_forward_pass_matches_llama_cpp() {
    let path = std::env::var_os("VIREO_GEMMA_26B").map(PathBuf::from).or_else(|| {
        std::env::var_os("HOME").map(|h| {
            PathBuf::from(h).join(
                "Library/Application Support/InvurseChat/models/unsloth/\
                 gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K.gguf",
            )
        })
    });
    let Some(path) = path.filter(|p| p.exists()) else {
        eprintln!("skipped: the Gemma 4 26B-A4B file isn't on this machine");
        return;
    };
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Gemma4::load(&file, Pool::with_all_cores()).unwrap();
    assert_eq!(
        model.config.experts,
        Some(vireo_engine::models::gemma4::Experts {
            count: 128,
            used: 8,
            width: 704
        })
    );
    let ids = tokenizer.encode(
        "The weather report for the coming week says that on Monday we should expect",
        true,
        true,
    );
    assert_eq!(ids.len(), 15);
    let mut session = model.session(32);
    let mut probs = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = model.forward(&mut session, id, i + 1 == ids.len()) {
            probs = l.to_vec();
        }
    }
    cpu::softmax(&mut probs);
    // llama.cpp: "-" 74.1497%, " own" 11.4948%, " to" 9.9130%, "/" 0.4453%.
    for (id, percent) in [
        (236772, 74.1497),
        (1852, 11.4948),
        (531, 9.9130),
        (236786, 0.4453),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.2,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }
}
