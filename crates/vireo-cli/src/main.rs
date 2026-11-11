//! `vireo-cli`: runs the engine from the command line, so it can be tested and
//! benchmarked against llama.cpp before the server exists.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use vireo_engine::cpu::Pool;
use vireo_engine::cuda::Gpu;
use vireo_engine::models::gemma4::{self, Gemma4};
use vireo_engine::models::gemma4_cuda::{Gemma4Cuda, Gemma4CudaSession};
use vireo_engine::models::llama::{Llama, Session};
use vireo_engine::models::llama_cuda::{CudaSession, LlamaCuda};
use vireo_engine::models::qwen35::{self, Qwen35};
use vireo_engine::models::qwen35_cuda::{Qwen35Cuda, Qwen35CudaSession};
use vireo_engine::{GgufFile, Tokenizer};

mod json;
mod serve;

const USAGE: &str = "\
Usage:
  vireo-cli inspect <model.gguf> [--tensors]   Show a GGUF file's metadata and tensors
  vireo-cli tokenize <model.gguf> <text> [options]
                                               Show how the model's tokenizer splits text
      --file          <text> is a file to read the text from
      --raw           No BOS token; special tokens in the text are plain text
      --ids           Print only the ids, as llama.cpp's llama-tokenize --ids does
  vireo-cli generate <model.gguf> <prompt> [options]
                                               Continue the prompt, picking the likeliest
                                               token each time (greedy)
      -n <count>      Tokens to generate (default 64)
      --file          <prompt> is a file to read the prompt from
      --raw           No BOS token; special tokens in the prompt are plain text
      --top <k>       Also show the k likeliest first tokens and their probabilities
      --threads <n>   CPU threads (default: every core)
      --gpu           Run on the NVIDIA GPU instead of the CPU
  vireo-cli perplexity <model.gguf> <text-file> [-c <context>] [--chunks <n>] [--gpu]
                                               How well the model predicts the text, scored
                                               as llama.cpp's llama-perplexity scores it
  vireo-cli serve <model.gguf> [options]       Keep the model loaded and answer chats on
                                               http://127.0.0.1 (for the Windows app)
      --port <n>      Port to listen on (default 39281)
      --token <t>     Answer only requests that carry this bearer token
      --gpu           Run on the NVIDIA GPU
      --threads <n>   CPU threads (default: every core)
      --context <n>   Context length in tokens (default 16384, capped by the model's)
      --temperature <t> --top-p <p> --top-k <k> --min-p <m> --presence-penalty <p>
                      Sampling defaults; a chat's own settings win
      --no-thinking   Answer without thinking first (Qwen, Gemma)
      --thinking-budget <n>
                      Close the thinking after this many tokens
  vireo-cli --version                          Show the version";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["--version"] => {
            println!("vireo-cli {}", vireo_engine::VERSION);
            ExitCode::SUCCESS
        }
        ["inspect", path, rest @ ..] if rest.iter().all(|a| *a == "--tensors") => {
            inspect(path, !rest.is_empty())
        }
        ["tokenize", path, text, rest @ ..]
            if rest
                .iter()
                .all(|a| matches!(*a, "--file" | "--raw" | "--ids")) =>
        {
            tokenize(path, text, rest)
        }
        ["generate", path, prompt, rest @ ..] => match GenerateOptions::parse(rest) {
            Some(options) => generate(path, prompt, &options),
            None => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        ["serve", path, rest @ ..] => match serve::ServeOptions::parse(rest) {
            Some(options) => serve::serve(path, options),
            None => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        ["perplexity", path, text, rest @ ..] => {
            let mut context = 512;
            let mut chunks = usize::MAX;
            let mut gpu = false;
            let mut args = rest.iter();
            while let Some(&arg) = args.next() {
                let mut value = || args.next().and_then(|v| v.parse().ok());
                match arg {
                    "--gpu" => gpu = true,
                    "-c" if let Some(v) = value() => context = v,
                    "--chunks" if let Some(v) = value() => chunks = v,
                    _ => {
                        eprintln!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            perplexity(path, text, context, chunks, gpu)
        }
        [] | ["help" | "--help" | "-h"] => {
            println!("vireo-cli {}\n\n{USAGE}", vireo_engine::VERSION);
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// The model on the CPU or the GPU, behind one interface for the commands.
enum Engine<'a> {
    Cpu(Llama<'a>),
    Gpu(LlamaCuda<'a>),
    Qwen35(Qwen35<'a>),
    Gemma4(Gemma4<'a>),
    Qwen35Gpu(Qwen35Cuda<'a>),
    Gemma4Gpu(Gemma4Cuda<'a>),
}

enum State {
    Cpu(Session),
    Gpu(CudaSession),
    Qwen35(qwen35::Session),
    Gemma4(gemma4::Session),
    Qwen35Gpu(Qwen35CudaSession),
    Gemma4Gpu(Gemma4CudaSession),
}

impl State {
    fn len(&self) -> usize {
        match self {
            State::Cpu(s) => s.len(),
            State::Gpu(s) => s.len(),
            State::Qwen35(s) => s.len(),
            State::Gemma4(s) => s.len(),
            State::Qwen35Gpu(s) => s.len(),
            State::Gemma4Gpu(s) => s.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            State::Cpu(s) => s.capacity(),
            State::Gpu(s) => s.capacity(),
            State::Qwen35(s) => s.capacity(),
            State::Gemma4(s) => s.capacity(),
            State::Qwen35Gpu(s) => s.capacity(),
            State::Gemma4Gpu(s) => s.capacity(),
        }
    }
}

impl<'a> Engine<'a> {
    fn load(file: &'a GgufFile, gpu: bool, threads: Option<usize>) -> Result<Self, String> {
        if file.architecture() == Some("gemma4") {
            if gpu {
                let device = Gpu::open().map_err(|e| e.to_string())?;
                return Gemma4Cuda::load(file, device)
                    .map(Engine::Gemma4Gpu)
                    .map_err(|e| e.to_string());
            }
            let pool = threads.map_or_else(Pool::with_all_cores, Pool::new);
            return Gemma4::load(file, pool)
                .map(Engine::Gemma4)
                .map_err(|e| e.to_string());
        }
        if file.architecture() == Some("qwen35") {
            if gpu {
                let device = Gpu::open().map_err(|e| e.to_string())?;
                return Qwen35Cuda::load(file, device)
                    .map(Engine::Qwen35Gpu)
                    .map_err(|e| e.to_string());
            }
            let pool = threads.map_or_else(Pool::with_all_cores, Pool::new);
            return Qwen35::load(file, pool)
                .map(Engine::Qwen35)
                .map_err(|e| e.to_string());
        }
        if gpu {
            let device = Gpu::open().map_err(|e| e.to_string())?;
            return LlamaCuda::load(file, device)
                .map(Engine::Gpu)
                .map_err(|e| e.to_string());
        }
        let pool = threads.map_or_else(Pool::with_all_cores, Pool::new);
        Llama::load(file, pool)
            .map(Engine::Cpu)
            .map_err(|e| e.to_string())
    }

    fn describe(&self) -> String {
        match self {
            Engine::Cpu(m) => format!(
                "{} layers, {} wide, on the CPU ({} threads)",
                m.config.layers,
                m.config.dim,
                m.threads()
            ),
            Engine::Gpu(m) => format!(
                "{} layers, {} wide, on the GPU ({})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name
            ),
            Engine::Gemma4(m) => format!(
                "Gemma 4: {} layers ({} global), {} wide, on the CPU ({} threads)",
                m.config.layers,
                m.config.is_swa.iter().filter(|s| !**s).count(),
                m.config.dim,
                m.threads()
            ),
            Engine::Qwen35Gpu(m) => format!(
                "Qwen 3.5: {} layers, {} wide, on the GPU ({})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name
            ),
            Engine::Gemma4Gpu(m) => format!(
                "Gemma 4: {} layers, {} wide, on the GPU ({})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name
            ),
            Engine::Qwen35(m) => format!(
                "Qwen 3.5: {} layers ({} full attention, the rest DeltaNet), {} wide, on the CPU ({} threads)",
                m.config.layers,
                (0..m.config.layers)
                    .filter(|&l| m.config.is_full_attention(l))
                    .count(),
                m.config.dim,
                m.threads()
            ),
        }
    }

    fn session(&self, capacity: usize) -> State {
        match self {
            Engine::Cpu(m) => State::Cpu(m.session(capacity)),
            Engine::Qwen35(m) => State::Qwen35(m.session(capacity)),
            Engine::Gemma4(m) => State::Gemma4(m.session(capacity)),
            Engine::Qwen35Gpu(m) => State::Qwen35Gpu(
                m.session(capacity)
                    .unwrap_or_else(|e| panic!("couldn't set up the GPU session: {e}")),
            ),
            Engine::Gemma4Gpu(m) => State::Gemma4Gpu(
                m.session(capacity)
                    .unwrap_or_else(|e| panic!("couldn't set up the GPU session: {e}")),
            ),
            Engine::Gpu(m) => State::Gpu(
                m.session(capacity)
                    .unwrap_or_else(|e| panic!("couldn't set up the GPU session: {e}")),
            ),
        }
    }

    fn forward<'s>(
        &self,
        state: &'s mut State,
        token: u32,
        want_logits: bool,
    ) -> Option<&'s [f32]> {
        match (self, state) {
            (Engine::Cpu(m), State::Cpu(s)) => m.forward(s, token, want_logits),
            (Engine::Qwen35(m), State::Qwen35(s)) => m.forward(s, token, want_logits),
            (Engine::Gemma4(m), State::Gemma4(s)) => m.forward(s, token, want_logits),
            (Engine::Qwen35Gpu(m), State::Qwen35Gpu(s)) => m
                .forward(s, token, want_logits)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            (Engine::Gemma4Gpu(m), State::Gemma4Gpu(s)) => m
                .forward(s, token, want_logits)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            (Engine::Gpu(m), State::Gpu(s)) => m
                .forward(s, token, want_logits)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            _ => unreachable!("a session belongs to the engine that made it"),
        }
    }
}

struct GenerateOptions {
    tokens: usize,
    file: bool,
    raw: bool,
    top: usize,
    threads: Option<usize>,
    gpu: bool,
}

impl GenerateOptions {
    fn parse(args: &[&str]) -> Option<Self> {
        let mut o = Self {
            tokens: 64,
            file: false,
            raw: false,
            top: 0,
            threads: None,
            gpu: false,
        };
        let mut args = args.iter();
        while let Some(&arg) = args.next() {
            match arg {
                "-n" => o.tokens = args.next()?.parse().ok()?,
                "--file" => o.file = true,
                "--raw" => o.raw = true,
                "--gpu" => o.gpu = true,
                "--top" => o.top = args.next()?.parse().ok()?,
                "--threads" => o.threads = Some(args.next()?.parse().ok()?),
                _ => return None,
            }
        }
        Some(o)
    }
}

fn generate(path: &str, prompt: &str, options: &GenerateOptions) -> ExitCode {
    let prompt = if options.file {
        match std::fs::read_to_string(prompt) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Couldn't read {prompt}: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        prompt.to_string()
    };
    let fail = |what: &str, e: &dyn std::fmt::Display| {
        eprintln!("Couldn't {what}: {e}");
        ExitCode::FAILURE
    };

    let started = Instant::now();
    let file = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => return fail(&format!("open {path}"), &e),
    };
    let tokenizer = match Tokenizer::from_gguf(&file) {
        Ok(t) => t,
        Err(e) => return fail("load the tokenizer", &e),
    };
    let model = match Engine::load(&file, options.gpu, options.threads) {
        Ok(m) => m,
        Err(e) => return fail("load the model", &e),
    };
    eprintln!("Loaded in {:.1?}: {}.", started.elapsed(), model.describe());

    let ids = tokenizer.encode(&prompt, !options.raw, !options.raw);
    if ids.is_empty() {
        eprintln!("The prompt has no tokens.");
        return ExitCode::FAILURE;
    }
    let mut session = model.session(ids.len() + options.tokens);

    // The prompt: every token but the last needs no logits.
    let started = Instant::now();
    for &id in &ids[..ids.len() - 1] {
        model.forward(&mut session, id, false);
    }
    let mut logits = model
        .forward(&mut session, ids[ids.len() - 1], true)
        .expect("asked for logits")
        .to_vec();
    let prompt_time = started.elapsed();

    if options.top > 0 {
        let mut probs = logits.clone();
        vireo_engine::cpu::softmax(&mut probs);
        let mut ranked: Vec<usize> = (0..probs.len()).collect();
        ranked.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        eprintln!("Likeliest first tokens:");
        for &id in ranked.iter().take(options.top) {
            let piece = tokenizer.decode(&[id as u32], true);
            eprintln!("  {id:>7}  {:>8.4}%  {piece:?}", probs[id] * 100.0);
        }
    }

    // Greedy generation, printing text as soon as its bytes make whole characters.
    let started = Instant::now();
    let mut pending = Vec::new();
    let mut generated = Vec::new();
    let mut out = std::io::stdout().lock();
    for _ in 0..options.tokens {
        let next = argmax(&logits);
        generated.push(next);
        if tokenizer.is_end_of_generation(next) {
            break;
        }
        tokenizer.token_bytes(next, false, &mut pending);
        let whole = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(), // a character still coming
            Err(_) => pending.len(),                              // invalid: print it lossily
        };
        let _ = out.write_all(String::from_utf8_lossy(&pending[..whole]).as_bytes());
        let _ = out.flush();
        pending.drain(..whole);
        if session.len() == session.capacity() {
            break;
        }
        logits.copy_from_slice(
            model
                .forward(&mut session, next, true)
                .expect("asked for logits"),
        );
    }
    let _ = out.write_all(String::from_utf8_lossy(&pending).as_bytes());
    let _ = writeln!(out);
    let gen_time = started.elapsed();

    let ids_text: Vec<String> = generated.iter().map(u32::to_string).collect();
    eprintln!("\nGenerated ids: [{}]", ids_text.join(", "));
    eprintln!(
        "Prompt: {} tokens in {prompt_time:.2?} ({:.1} tokens/s). Generated: {} tokens in {gen_time:.2?} ({:.1} tokens/s).",
        ids.len(),
        ids.len() as f64 / prompt_time.as_secs_f64(),
        generated.len(),
        generated.len() as f64 / gen_time.as_secs_f64()
    );
    ExitCode::SUCCESS
}

/// Perplexity the way llama-perplexity computes it, so the two can be compared: the
/// text (with a BOS token first) is cut into chunks of `context` tokens; each chunk
/// starts with BOS in place of its first token and runs from an empty context; and only
/// the second half of each chunk is scored, so every scored token has at least half a
/// context before it.
fn perplexity(
    path: &str,
    text_path: &str,
    context: usize,
    max_chunks: usize,
    gpu: bool,
) -> ExitCode {
    let text = match std::fs::read_to_string(text_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Couldn't read {text_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Couldn't open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (tokenizer, model) = match Tokenizer::from_gguf(&file)
        .map_err(|e| e.to_string())
        .and_then(|t| Ok((t, Engine::load(&file, gpu, None)?)))
    {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("Couldn't load {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let tokens = tokenizer.encode(&text, true, false);
    let chunks = (tokens.len() / context).min(max_chunks);
    if chunks == 0 {
        eprintln!(
            "The text has {} tokens: it needs at least {context}.",
            tokens.len()
        );
        return ExitCode::FAILURE;
    }
    eprintln!(
        "{} tokens; scoring {chunks} chunks of {context}; {}.",
        tokens.len(),
        model.describe()
    );
    let started = Instant::now();
    let (mut nll, mut count) = (0f64, 0usize);
    for chunk in 0..chunks {
        let tokens = &tokens[chunk * context..(chunk + 1) * context];
        let mut session = model.session(context);
        for (j, &token) in tokens.iter().enumerate() {
            let token = match (j, tokenizer.bos()) {
                (0, Some(bos)) => bos,
                _ => token,
            };
            let scored = j >= context / 2 && j + 1 < context;
            let logits = model.forward(&mut session, token, scored);
            if let Some(logits) = logits {
                // -log softmax(logits)[next], computed in f64 for a stable sum.
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
                let next = tokens[j + 1] as usize;
                nll += max + sum.ln() - logits[next] as f64;
                count += 1;
            }
        }
        eprintln!("  [{}] {:.4}", chunk + 1, (nll / count as f64).exp());
    }
    println!(
        "Final estimate: PPL = {:.4} ({count} tokens scored in {:.1?})",
        (nll / count as f64).exp(),
        started.elapsed()
    );
    ExitCode::SUCCESS
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best as u32
}

fn tokenize(path: &str, text: &str, options: &[&str]) -> ExitCode {
    let has = |option: &str| options.contains(&option);
    let text = if has("--file") {
        match std::fs::read_to_string(text) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Couldn't read {text}: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        text.to_string()
    };

    let started = Instant::now();
    let tokenizer = match GgufFile::open(path)
        .map_err(|e| e.to_string())
        .and_then(|file| Tokenizer::from_gguf(&file).map_err(|e| e.to_string()))
    {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!("Couldn't load the tokenizer from {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let loaded = started.elapsed();

    let special = !has("--raw");
    let started = Instant::now();
    let ids = tokenizer.encode(&text, special, special);
    let encoded = started.elapsed();

    if has("--ids") {
        let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
        println!("[{}]", ids.join(", "));
        return ExitCode::SUCCESS;
    }
    for &id in &ids {
        let mut bytes = Vec::new();
        tokenizer.token_bytes(id, true, &mut bytes);
        let piece = String::from_utf8_lossy(&bytes);
        println!("{id:>7}  {:?}", piece.as_ref());
    }
    println!(
        "\n{} tokens from {} characters. Tokenizer loaded in {loaded:.1?}, text tokenized in {encoded:.1?}.",
        ids.len(),
        text.chars().count()
    );
    ExitCode::SUCCESS
}

fn inspect(path: &str, list_tensors: bool) -> ExitCode {
    let file = match GgufFile::open(path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Couldn't open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("{path}");
    println!("  {}, GGUF version {}", size(file.len()), file.version);
    println!(
        "  Architecture  {}",
        file.architecture().unwrap_or("(not given)")
    );
    if let Some(name) = file.get_str("general.name") {
        println!("  Name          {name}");
    }
    let params: u64 = file.tensors.iter().map(|t| t.elements()).sum();
    println!("  Parameters    {}", count(params));
    println!(
        "  Tensors       {}, data from byte {} (aligned to {})",
        file.tensors.len(),
        file.data_offset,
        file.alignment
    );

    println!("\nMetadata ({} keys)", file.metadata.len());
    let key_width = file
        .metadata
        .iter()
        .map(|(k, _)| k.len())
        .max()
        .unwrap_or(0);
    for (key, value) in &file.metadata {
        println!("  {key:key_width$}  {value}");
    }

    // Tensor types by how much of the file they take.
    let mut by_type: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for t in &file.tensors {
        let name = t.kind.map_or_else(
            || format!("unknown type {}", t.type_id),
            |k| k.name().to_string(),
        );
        let entry = by_type.entry(name).or_default();
        entry.0 += 1;
        entry.1 += t.bytes.unwrap_or(0);
        entry.2 += t.elements();
    }
    let mut by_type: Vec<_> = by_type.into_iter().collect();
    by_type.sort_by_key(|(_, (_, bytes, _))| std::cmp::Reverse(*bytes));
    println!("\nTensor types");
    for (name, (tensors, bytes, elements)) in &by_type {
        let share = 100.0 * *elements as f64 / params.max(1) as f64;
        println!(
            "  {name:8} {tensors:>5} tensors  {:>10}  {share:>5.1}% of parameters",
            size(*bytes)
        );
    }
    let unknown = file.tensors.iter().filter(|t| t.kind.is_none()).count();
    if unknown > 0 {
        println!(
            "  {unknown} tensors have types this engine doesn't know, so their data can't be read."
        );
    }

    if list_tensors {
        println!("\nTensors");
        let name_width = file.tensors.iter().map(|t| t.name.len()).max().unwrap_or(0);
        for t in &file.tensors {
            let kind = t
                .kind
                .map_or_else(|| format!("type {}", t.type_id), |k| k.name().to_string());
            let dims: Vec<String> = t.dims.iter().map(u64::to_string).collect();
            let bytes = t.bytes.map_or_else(|| "?".to_string(), size);
            println!(
                "  {:name_width$}  {kind:8} [{}]  {bytes}",
                t.name,
                dims.join(" × ")
            );
        }
    } else {
        println!("\n(Add --tensors to list every tensor.)");
    }
    ExitCode::SUCCESS
}

/// "808.0 MB", "1.32 GB", in the decimal units Windows Explorer and Hugging Face use.
fn size(bytes: u64) -> String {
    let b = bytes as f64;
    match bytes {
        0..1_000 => format!("{bytes} bytes"),
        1_000..1_000_000 => format!("{:.1} KB", b / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1} MB", b / 1e6),
        _ => format!("{:.2} GB", b / 1e9),
    }
}

/// "1.24B", "596M": how parameter counts are usually written.
fn count(n: u64) -> String {
    let f = n as f64;
    match n {
        0..1_000_000 => n.to_string(),
        1_000_000..1_000_000_000 => format!("{:.0}M", f / 1e6),
        _ => format!("{:.2}B", f / 1e9),
    }
}
