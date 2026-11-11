//! Constrained generation on the real models' vocabularies and, with an NVIDIA GPU, a
//! real tool call written under its grammar. Skipped (with a note) without the models.

use std::path::PathBuf;
use std::time::Instant;

use vireo_engine::grammar::{Grammar, Matcher, Property, Schema, TokenTrie};
use vireo_engine::models::Logits;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
use vireo_engine::{GgufFile, Tokenizer};

const MODELS: [&str; 3] = [
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "gemma-4-E2B_q4_0-it.gguf",
];

fn model(name: &str) -> Option<GgufFile> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name);
    if !path.exists() {
        eprintln!("skipped: {name} isn't in VireoPrototype/models");
        return None;
    }
    Some(GgufFile::open(path).unwrap())
}

fn file_tools() -> Schema {
    Schema::tool_call(&[
        (
            "read_file".into(),
            Schema::Object(vec![
                Property::required("path", Schema::string()),
                Property::optional("max_lines", Schema::Integer),
            ]),
        ),
        (
            "list_directory".into(),
            Schema::Object(vec![Property::required("path", Schema::string())]),
        ),
        (
            "search_files".into(),
            Schema::Object(vec![
                Property::required("query", Schema::string()),
                Property::optional("folder", Schema::string()),
            ]),
        ),
        ("get_current_datetime".into(), Schema::Object(vec![])),
    ])
}

fn is_set(bits: &[u64], id: u32) -> bool {
    bits[id as usize / 64] >> (id % 64) & 1 == 1
}

/// On each real vocabulary, at every byte of a call, the trie's tokens are exactly the
/// tokens that fit when each is tried on its own; and how long finding them takes.
#[test]
fn allowed_tokens_on_real_vocabularies() {
    let text = r#"{"name": "search_files", "arguments": {"query": "Résumé \"draft\" 2026\n", "folder": "C:\\Users\\me"}}"#;
    let g = Grammar::new(&file_tools());
    for name in MODELS {
        let Some(file) = model(name) else { continue };
        let tokenizer = Tokenizer::from_gguf(&file).unwrap();
        let started = Instant::now();
        let trie = TokenTrie::new(&tokenizer);
        let built = started.elapsed();
        let tokens: Vec<(u32, Vec<u8>)> = (0..tokenizer.vocab_size() as u32)
            .filter(|&id| {
                matches!(
                    tokenizer.token_type(id),
                    Some(
                        vireo_engine::tokenizer::TokenType::Normal
                            | vireo_engine::tokenizer::TokenType::Byte
                    )
                )
            })
            .map(|id| {
                let mut b = Vec::new();
                tokenizer.token_bytes(id, false, &mut b);
                (id, b)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect();
        let mut m = g.matcher();
        let (mut worst, mut total) = (0f64, 0f64);
        for i in 0..=text.len() {
            let started = Instant::now();
            let allowed = m.allowed(&trie);
            let took = started.elapsed().as_secs_f64();
            worst = worst.max(took);
            total += took;
            for (id, bytes) in &tokens {
                assert_eq!(
                    is_set(&allowed, *id),
                    m.allows(bytes),
                    "{name}: token {id} {:?} after {:?}",
                    String::from_utf8_lossy(bytes),
                    &text[..i]
                );
            }
            if i < text.len() {
                assert!(m.advance(text.as_bytes()[i]));
            }
        }
        assert!(m.is_complete());
        eprintln!(
            "{name}: {} tokens, trie built in {:.0} ms; allowed tokens in {:.2} ms on average, {:.2} ms at most",
            tokens.len(),
            built.as_secs_f64() * 1000.0,
            total / (text.len() + 1) as f64 * 1000.0,
            worst * 1000.0
        );
    }
}

/// A model on the GPU, behind one interface: read tokens, get the last one's logits.
trait Run {
    fn read(&mut self, ids: &[u32]) -> Vec<f32>;
}

struct Llama<'a>(LlamaCuda<'a>, vireo_engine::models::llama_cuda::CudaSession);
struct Qwen<'a>(
    Qwen35Cuda<'a>,
    vireo_engine::models::qwen35_cuda::Qwen35CudaSession,
);
struct Gemma<'a>(
    Gemma4Cuda<'a>,
    vireo_engine::models::gemma4_cuda::Gemma4CudaSession,
);

impl Run for Llama<'_> {
    fn read(&mut self, ids: &[u32]) -> Vec<f32> {
        self.0
            .forward_batch(&mut self.1, ids, Logits::Last)
            .unwrap()
            .unwrap()
            .to_vec()
    }
}
impl Run for Qwen<'_> {
    fn read(&mut self, ids: &[u32]) -> Vec<f32> {
        self.0
            .forward_batch(&mut self.1, ids, Logits::Last)
            .unwrap()
            .unwrap()
            .to_vec()
    }
}
impl Run for Gemma<'_> {
    fn read(&mut self, ids: &[u32]) -> Vec<f32> {
        self.0
            .forward_batch(&mut self.1, ids, Logits::Last)
            .unwrap()
            .unwrap()
            .to_vec()
    }
}

/// The likeliest allowed token.
fn pick(logits: &[f32], allowed: &[u64]) -> u32 {
    (0..logits.len() as u32)
        .filter(|&id| is_set(allowed, id))
        .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
        .expect("some token is allowed")
}

/// Writes a call under `grammar` after `prompt`, greedily, with forced text added without
/// asking the model. The call's text, the tokens it took, and the model's passes.
fn constrained_call(
    run: &mut dyn Run,
    tokenizer: &Tokenizer,
    trie: &TokenTrie,
    grammar: &Grammar,
    prompt: &[u32],
) -> (String, usize, usize) {
    let mut m: Matcher = grammar.matcher();
    let mut text = Vec::new();
    let (mut tokens, mut passes) = (0, 0);
    let mut pending: Vec<u32> = prompt.to_vec();
    loop {
        let forced = m.forced();
        if !forced.is_empty() {
            let ids = tokenizer.encode(std::str::from_utf8(&forced).unwrap(), false, false);
            assert!(m.advance_all(&forced));
            text.extend_from_slice(&forced);
            tokens += ids.len();
            pending.extend(ids);
        }
        if m.is_complete() {
            break;
        }
        let logits = run.read(&pending);
        passes += 1;
        pending.clear();
        let id = pick(&logits, &m.allowed(trie));
        let mut bytes = Vec::new();
        tokenizer.token_bytes(id, false, &mut bytes);
        assert!(m.advance_all(&bytes));
        text.extend_from_slice(&bytes);
        tokens += 1;
        pending.push(id);
        assert!(
            tokens < 200,
            "a runaway call: {}",
            String::from_utf8_lossy(&text)
        );
    }
    (String::from_utf8(text).unwrap(), tokens, passes)
}

/// Each model, asked something a tool answers, writes a call that fits the grammar, and
/// the one a person would expect; forced text saves passes.
#[test]
fn a_constrained_tool_call_on_each_model() {
    let system = "You can call these tools. Answer with one call as JSON: {\"name\": ..., \"arguments\": {...}}.\n\
                  read_file(path, max_lines?): a file's text\n\
                  list_directory(path): the files in a folder\n\
                  search_files(query, folder?): files whose names or text match\n\
                  get_current_datetime(): the date and time now";
    let question = r"What does C:\Users\me\notes.txt say?";
    let grammar = Grammar::new(&file_tools());
    for name in MODELS {
        let Some(file) = model(name) else { continue };
        let gpu = match vireo_engine::cuda::Gpu::open() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipped: {e}");
                return;
            }
        };
        let tokenizer = Tokenizer::from_gguf(&file).unwrap();
        let trie = TokenTrie::new(&tokenizer);
        let (prompt, mut run): (String, Box<dyn Run>) = match name {
            n if n.starts_with("Llama") => {
                let m = LlamaCuda::load(&file, gpu).unwrap();
                let s = m.session(2048).unwrap();
                (
                    format!(
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
                         <|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\
                         <|start_header_id|>assistant<|end_header_id|>\n\n"
                    ),
                    Box::new(Llama(m, s)),
                )
            }
            n if n.starts_with("Qwen") => {
                let m = Qwen35Cuda::load(&file, gpu).unwrap();
                let s = m.session(2048).unwrap();
                (
                    format!(
                        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n\
                         <|im_start|>assistant\n<think>\n\n</think>\n\n"
                    ),
                    Box::new(Qwen(m, s)),
                )
            }
            _ => {
                let m = Gemma4Cuda::load(&file, gpu).unwrap();
                let s = m.session(2048).unwrap();
                (
                    format!(
                        "<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{question}<turn|>\n<|turn>model\n"
                    ),
                    Box::new(Gemma(m, s)),
                )
            }
        };
        let ids = tokenizer.encode(&prompt, false, true);
        let started = Instant::now();
        let (call, tokens, passes) =
            constrained_call(run.as_mut(), &tokenizer, &trie, &grammar, &ids);
        eprintln!(
            "{name}: {call}  ({tokens} tokens in {passes} passes, {:.0} ms with the prompt)",
            started.elapsed().as_secs_f64() * 1000.0
        );
        assert!(
            call.starts_with(r#"{"name": "read_file", "arguments": {"path": ""#),
            "{name}: {call}"
        );
        assert!(call.contains("notes.txt"), "{name}: {call}");
        assert!(passes < tokens, "{name}: forced text should save passes");
    }
}
