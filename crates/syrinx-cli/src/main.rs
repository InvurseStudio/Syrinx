//! `syrinx-cli`: runs the engine from the command line, so it can be tested and
//! benchmarked against llama.cpp before the server exists.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use syrinx_engine::cpu::Pool;
use syrinx_engine::cuda::Gpu;
use syrinx_engine::models::gemma4::{self, Gemma4};
use syrinx_engine::models::gemma4_cuda::{Gemma4Cuda, Gemma4CudaSession};
use syrinx_engine::models::inkling2::{self, Inkling2};
use syrinx_engine::models::inkling2_cuda::{Inkling2Cuda, Inkling2CudaSession};
use syrinx_engine::models::llama::{Llama, Session};
use syrinx_engine::models::llama_cuda::{CudaSession, LlamaCuda};
use syrinx_engine::models::qwen35::{self, Qwen35};
use syrinx_engine::models::qwen35_cuda::{Qwen35Cuda, Qwen35CudaSession};
use syrinx_engine::models::{Logits, Precision};
use syrinx_engine::{GgufFile, Tokenizer};

const USAGE: &str = "\
Usage:
  syrinx-cli inspect <model.gguf> [--tensors]   Show a GGUF file's metadata and tensors
  syrinx-cli tokenize <model.gguf> <text> [options]
                                               Show how the model's tokenizer splits text
      --file          <text> is a file to read the text from
      --raw           No BOS token; special tokens in the text are plain text
      --ids           Print only the ids, as llama.cpp's llama-tokenize --ids does
  syrinx-cli generate <model.gguf> <prompt> [options]
                                               Continue the prompt, picking the likeliest
                                               token each time (greedy)
      -n <count>      Tokens to generate (default 64)
      --file          <prompt> is a file to read the prompt from
      --raw           No BOS token; special tokens in the prompt are plain text
      --top <k>       Also show the k likeliest first tokens and their probabilities
      --threads <n>   CPU threads (default: every core)
      --gpu           Run on the NVIDIA GPU instead of the CPU
      --exact         On the GPU, exact f32 arithmetic instead of 8-bit activations
                      (slower; the CPU's numbers)
  syrinx-cli perplexity <model.gguf> <text-file> [-c <context>] [--chunks <n>] [--gpu] [--exact]
                                               How well the model predicts the text, scored
                                               as llama.cpp's llama-perplexity scores it
  syrinx-cli batches <model.gguf> <text-file> [--depth <tokens>]
  syrinx-cli together <model.gguf> <text-file> [--depth <tokens>] [--chats <n>]
                                               On the GPU: what a batch of 1 to 64 tokens
                                               costs with <tokens> already read (4000 by
                                               default): a draft's guesses checked at once,
                                               a short tool result. Gemma 4 and Qwen 3.5
  syrinx-cli --version                          Show the version";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["--version"] => {
            println!("syrinx-cli {}", syrinx_engine::VERSION);
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
        ["batches", path, text, rest @ ..] => {
            let depth = match rest {
                [] => Some(4000),
                ["--depth", n] => n.parse().ok(),
                _ => None,
            };
            match depth {
                Some(depth) => batches(path, text, depth),
                None => {
                    eprintln!("{USAGE}");
                    ExitCode::from(2)
                }
            }
        }
        ["together", path, text, rest @ ..] => {
            let (mut depth, mut chats, mut ok) = (4000, 8, rest.len().is_multiple_of(2));
            for pair in rest.chunks(2) {
                match pair {
                    ["--depth", n] => depth = n.parse().unwrap_or(0),
                    ["--chats", n] => chats = n.parse().unwrap_or(0),
                    _ => ok = false,
                }
            }
            if ok && depth > 0 && chats > 0 {
                together(path, text, depth, chats)
            } else {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        }
        ["perplexity", path, text, rest @ ..] => {
            let mut context = 512;
            let mut chunks = usize::MAX;
            let mut gpu = false;
            let mut exact = false;
            let mut args = rest.iter();
            while let Some(&arg) = args.next() {
                let mut value = || args.next().and_then(|v| v.parse().ok());
                match arg {
                    "--gpu" => gpu = true,
                    "--exact" => exact = true,
                    "-c" if let Some(v) = value() => context = v,
                    "--chunks" if let Some(v) = value() => chunks = v,
                    _ => {
                        eprintln!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            let precision = if exact {
                Precision::Exact
            } else {
                Precision::Int8
            };
            perplexity(path, text, context, chunks, gpu, precision)
        }
        [] | ["help" | "--help" | "-h"] => {
            println!("syrinx-cli {}\n\n{USAGE}", syrinx_engine::VERSION);
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// The model on the CPU or the GPU, behind one interface for the commands.
#[allow(clippy::large_enum_variant)] // one, for the life of the command
enum Engine<'a> {
    Cpu(Llama<'a>),
    Gpu(LlamaCuda<'a>),
    Qwen35(Qwen35<'a>),
    Gemma4(Gemma4<'a>),
    Inkling2(Inkling2<'a>),
    Inkling2Gpu(Inkling2Cuda<'a>),
    Qwen35Gpu(Qwen35Cuda<'a>),
    Gemma4Gpu(Gemma4Cuda<'a>),
}

enum State {
    Cpu(Session),
    Gpu(CudaSession),
    Qwen35(qwen35::Session),
    Gemma4(gemma4::Session),
    Inkling2(inkling2::Session),
    Inkling2Gpu(Inkling2CudaSession),
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
            State::Inkling2(s) => s.len(),
            State::Inkling2Gpu(s) => s.len(),
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
            State::Inkling2(s) => s.capacity(),
            State::Inkling2Gpu(s) => s.capacity(),
            State::Qwen35Gpu(s) => s.capacity(),
            State::Gemma4Gpu(s) => s.capacity(),
        }
    }
}

impl<'a> Engine<'a> {
    fn load(
        file: &'a GgufFile,
        gpu: bool,
        threads: Option<usize>,
        precision: Precision,
    ) -> Result<Self, String> {
        if matches!(file.architecture(), Some("gemma4" | "muse-glimmer")) {
            if gpu {
                let device = Gpu::open().map_err(|e| e.to_string())?;
                return Gemma4Cuda::load(file, device)
                    .map(|mut m| {
                        m.set_precision(precision);
                        Engine::Gemma4Gpu(m)
                    })
                    .map_err(|e| e.to_string());
            }
            let pool = threads.map_or_else(Pool::with_all_cores, Pool::new);
            return Gemma4::load(file, pool)
                .map(Engine::Gemma4)
                .map_err(|e| e.to_string());
        }
        if matches!(file.architecture(), Some("inkling2" | "inkling2_loop")) {
            if gpu {
                let device = Gpu::open().map_err(|e| e.to_string())?;
                return Inkling2Cuda::load(file, device)
                    .map(|mut m| {
                        m.set_precision(precision);
                        Engine::Inkling2Gpu(m)
                    })
                    .map_err(|e| e.to_string());
            }
            let pool = threads.map_or_else(Pool::with_all_cores, Pool::new);
            return Inkling2::load(file, pool)
                .map(Engine::Inkling2)
                .map_err(|e| e.to_string());
        }
        if file.architecture() == Some("qwen35") {
            if gpu {
                let device = Gpu::open().map_err(|e| e.to_string())?;
                return Qwen35Cuda::load(file, device)
                    .map(|mut m| {
                        m.set_precision(precision);
                        Engine::Qwen35Gpu(m)
                    })
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
                .map(|mut m| {
                    m.set_precision(precision);
                    Engine::Gpu(m)
                })
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
                "{} layers, {} wide, on the GPU ({}, {:?})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name,
                m.precision()
            ),
            Engine::Gemma4(m) => format!(
                "Gemma 4: {} layers ({} global), {} wide, on the CPU ({} threads)",
                m.config.layers,
                m.config.is_swa.iter().filter(|s| !**s).count(),
                m.config.dim,
                m.threads()
            ),
            Engine::Inkling2(m) => format!(
                "Inkling 2: {} layers ({} visits a token), {} wide, on the CPU ({} threads)",
                m.config.layers,
                m.config.slots(),
                m.config.dim,
                m.threads()
            ),
            Engine::Inkling2Gpu(m) => format!(
                "Inkling 2: {} layers ({} visits a token), {} wide, on the GPU ({}, {:?})",
                m.config.layers,
                m.config.slots(),
                m.config.dim,
                m.gpu().info.name,
                m.precision()
            ),
            Engine::Qwen35Gpu(m) => format!(
                "Qwen 3.5: {} layers, {} wide, on the GPU ({}, {:?})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name,
                m.precision()
            ),
            Engine::Gemma4Gpu(m) => format!(
                "Gemma 4: {} layers, {} wide, on the GPU ({}, {:?})",
                m.config.layers,
                m.config.dim,
                m.gpu().info.name,
                m.precision()
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
            Engine::Inkling2(m) => State::Inkling2(m.session(capacity)),
            Engine::Inkling2Gpu(m) => State::Inkling2Gpu(
                m.session(capacity)
                    .unwrap_or_else(|e| panic!("couldn't set up the GPU session: {e}")),
            ),
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
            (Engine::Inkling2(m), State::Inkling2(s)) => m.forward(s, token, want_logits),
            (Engine::Inkling2Gpu(m), State::Inkling2Gpu(s)) => m
                .forward(s, token, want_logits)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
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

    fn forward_greedy(&self, state: &mut State, token: u32) -> u32 {
        match (self, &mut *state) {
            (Engine::Qwen35Gpu(m), State::Qwen35Gpu(s)) => m
                .forward_greedy(s, token)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            (Engine::Gemma4Gpu(m), State::Gemma4Gpu(s)) => m
                .forward_greedy(s, token)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            (Engine::Inkling2Gpu(m), State::Inkling2Gpu(s)) => m
                .forward_greedy(s, token)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            (Engine::Gpu(m), State::Gpu(s)) => m
                .forward_greedy(s, token)
                .unwrap_or_else(|e| panic!("the GPU failed: {e}")),
            _ => argmax(self.forward(state, token, true).expect("asked for logits")),
        }
    }

    /// Runs `tokens` in order: in batches on the GPU, one by one on the CPU. Hands back
    /// the logits asked for (the last token's, or a row of the vocabulary per token).
    fn forward_batch(&self, state: &mut State, tokens: &[u32], logits: Logits) -> Option<Vec<f32>> {
        let gpu = |r: Result<Option<&[f32]>, syrinx_engine::models::Error>| {
            r.unwrap_or_else(|e| panic!("the GPU failed: {e}"))
                .map(<[f32]>::to_vec)
        };
        match (self, &mut *state) {
            (Engine::Gpu(m), State::Gpu(s)) => return gpu(m.forward_batch(s, tokens, logits)),
            (Engine::Qwen35Gpu(m), State::Qwen35Gpu(s)) => {
                return gpu(m.forward_batch(s, tokens, logits));
            }
            (Engine::Gemma4Gpu(m), State::Gemma4Gpu(s)) => {
                return gpu(m.forward_batch(s, tokens, logits));
            }
            (Engine::Inkling2Gpu(m), State::Inkling2Gpu(s)) => {
                return gpu(m.forward_batch(s, tokens, logits));
            }
            _ => {}
        }
        let mut all = Vec::new();
        let mut last = None;
        for (i, &token) in tokens.iter().enumerate() {
            let want = logits == Logits::All || (logits == Logits::Last && i + 1 == tokens.len());
            if let Some(l) = self.forward(state, token, want) {
                if logits == Logits::All {
                    all.extend_from_slice(l);
                } else {
                    last = Some(l.to_vec());
                }
            }
        }
        match logits {
            Logits::None => None,
            Logits::Last => last,
            Logits::All => Some(all),
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
    exact: bool,
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
            exact: false,
        };
        let mut args = args.iter();
        while let Some(&arg) = args.next() {
            match arg {
                "-n" => o.tokens = args.next()?.parse().ok()?,
                "--file" => o.file = true,
                "--raw" => o.raw = true,
                "--gpu" => o.gpu = true,
                "--exact" => o.exact = true,
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
    let precision = if options.exact {
        Precision::Exact
    } else {
        Precision::Int8
    };
    let model = match Engine::load(&file, options.gpu, options.threads, precision) {
        Ok(m) => m,
        Err(e) => return fail("load the model", &e),
    };
    // As syrinx-server does after a load, so the first prompt's time is the prompt's.
    match &model {
        Engine::Gpu(m) => m.gpu().settle(),
        Engine::Qwen35Gpu(m) => m.gpu().settle(),
        Engine::Gemma4Gpu(m) => m.gpu().settle(),
        Engine::Inkling2Gpu(m) => m.gpu().settle(),
        _ => {}
    }
    eprintln!("Loaded in {:.1?}: {}.", started.elapsed(), model.describe());

    let ids = tokenizer.encode(&prompt, !options.raw, !options.raw);
    if ids.is_empty() {
        eprintln!("The prompt has no tokens.");
        return ExitCode::FAILURE;
    }
    let mut session = model.session(ids.len() + options.tokens);

    // The prompt: only the last token's logits are needed.
    let started = Instant::now();
    let logits = model
        .forward_batch(&mut session, &ids, Logits::Last)
        .expect("asked for logits");
    let prompt_time = started.elapsed();

    if options.top > 0 {
        let mut probs = logits.clone();
        syrinx_engine::cpu::softmax(&mut probs);
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
    let mut next = argmax(&logits);
    for _ in 0..options.tokens {
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
        next = model.forward_greedy(&mut session, next);
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
    precision: Precision,
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
        .and_then(|t| Ok((t, Engine::load(&file, gpu, None, precision)?)))
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
    let mut dump = std::env::var_os("VIREO_PPL_DUMP")
        .and_then(|p| std::fs::File::create(p).ok())
        .map(std::io::BufWriter::new);
    let (mut nll, mut count) = (0f64, 0usize);
    for chunk in 0..chunks {
        let tokens = &tokens[chunk * context..(chunk + 1) * context];
        let mut session = model.session(context);
        let mut input = tokens.to_vec();
        // As llama-perplexity: a chunk starts with BOS only for a model that adds one
        // (Qwen 3.8 27B names a BOS token but doesn't add it).
        if let (true, Some(bos)) = (tokenizer.adds_bos(), tokenizer.bos()) {
            input[0] = bos;
        }
        // The first half only gives context; the second half's predictions are scored.
        let half = context / 2;
        model.forward_batch(&mut session, &input[..half], Logits::None);
        let all = model
            .forward_batch(&mut session, &input[half..], Logits::All)
            .expect("asked for logits");
        let vocab = all.len() / (context - half);
        for j in half..context - 1 {
            let logits = &all[(j - half) * vocab..(j - half + 1) * vocab];
            // -log softmax(logits)[next], computed in f64 for a stable sum.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
            let next = tokens[j + 1] as usize;
            nll += max + sum.ln() - logits[next] as f64;
            // VIREO_PPL_DUMP=<file>: each scored token's -log p and likeliest token, one a
            // line, to compare two modes token by token.
            if let Some(d) = dump.as_mut() {
                use std::io::Write;
                let _ = writeln!(d, "{} {}", max + sum.ln() - logits[next] as f64, argmax(logits));
            }
            count += 1;
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

/// The batch sizes `batches` times.
const BATCH_SIZES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// What batches of a few tokens cost `depth` tokens into a chat, on the GPU: each size
/// read with its logits five times from the same place, the quickest kept. One line of
/// `size: milliseconds` pairs on standard output. A draft's guesses are checked in such
/// batches, so their cost against one token's decides what drafting gains; no test timed
/// them until a kernel change doubled a check of 4 to 16 tokens unnoticed (15-17 November
/// 2026). Vireo Bench's S1 runs this (`b4` to `b64`).
fn batches(path: &str, text_path: &str, depth: usize) -> ExitCode {
    let load = || -> Result<Vec<f64>, String> {
        let text = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;
        let file = GgufFile::open(path).map_err(|e| format!("{path}: {e}"))?;
        let tokenizer = Tokenizer::from_gguf(&file).map_err(|e| e.to_string())?;
        let ids = tokenizer.encode(&text, true, false);
        let most = BATCH_SIZES[BATCH_SIZES.len() - 1];
        if ids.len() < depth + most {
            return Err(format!(
                "The text has {} tokens: it needs {}.",
                ids.len(),
                depth + most
            ));
        }
        let gpu = Gpu::open().map_err(e)?;
        fn e<E: std::fmt::Display>(e: E) -> String {
            e.to_string()
        }
        let mut ms = Vec::new();
        let qwen = file.architecture().is_some_and(|a| a.starts_with("qwen"));
        if qwen {
            let m = Qwen35Cuda::load(&file, gpu).map_err(e)?;
            let mut s = m.session(depth + 512).map_err(e)?;
            // As the server reads a round of guesses: the DeltaNet steps kept.
            if m.can_rewind() {
                m.keep_steps(&mut s).map_err(e)?;
            }
            m.forward_batch(&mut s, &ids[..depth], Logits::Last).map_err(e)?;
            let mut checkpoint = None;
            m.save(&s, &mut checkpoint).map_err(e)?;
            m.gpu().synchronize().map_err(e)?;
            for n in BATCH_SIZES {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    m.forward_batch(&mut s, &ids[depth..depth + n], Logits::Last).map_err(e)?;
                    m.gpu().synchronize().map_err(e)?;
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    m.restore(&mut s, checkpoint.as_ref().unwrap()).map_err(e)?;
                    m.gpu().synchronize().map_err(e)?;
                }
                ms.push(best);
            }
        } else {
            let m = Gemma4Cuda::load(&file, gpu).map_err(e)?;
            let mut s = m.session(depth + 512).map_err(e)?;
            m.forward_batch(&mut s, &ids[..depth], Logits::Last).map_err(e)?;
            for n in BATCH_SIZES {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    m.forward_batch(&mut s, &ids[depth..depth + n], Logits::Last).map_err(e)?;
                    m.gpu().synchronize().map_err(e)?;
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    if s.truncate(depth) != depth {
                        return Err("The session couldn't go back over a batch.".into());
                    }
                }
                ms.push(best);
            }
        }
        Ok(ms)
    };
    match load() {
        Ok(ms) => {
            let cells: Vec<String> = BATCH_SIZES
                .iter()
                .zip(&ms)
                .map(|(n, t)| format!("{n}: {t:.2}"))
                .collect();
            println!("Batches at {depth} tokens, ms: {}", cells.join("  "));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// How far apart in the text the chats `together` reads start.
const TOGETHER_APART: usize = 300;
/// How many tokens each chat writes in a timed run of `together`.
const TOGETHER_STEPS: usize = 8;

/// What a token costs each of several chats written together (Gemma 4, on the GPU):
/// `chats` conversations, each `depth` tokens of a different stretch of the text, then
/// one token for each in one pass (`Gemma4Cuda::forward_each`), every chat's logits read
/// back. First that each chat's logits are what it gets alone; then, for 1, 2, 4...
/// chats, the milliseconds a step (the quickest of five runs of a few steps) and the
/// tokens a second each chat and all of them get.
fn together(path: &str, text_path: &str, depth: usize, chats: usize) -> ExitCode {
    fn e<E: std::fmt::Display>(e: E) -> String {
        e.to_string()
    }
    let run = || -> Result<(), String> {
        let text = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;
        let file = GgufFile::open(path).map_err(|e| format!("{path}: {e}"))?;
        if file.architecture().is_some_and(|a| a.starts_with("qwen")) {
            return Err("`together` is for Gemma 4 models so far.".into());
        }
        let tokenizer = Tokenizer::from_gguf(&file).map_err(|e| e.to_string())?;
        let ids = tokenizer.encode(&text, true, false);
        let needs = depth + chats * TOGETHER_APART + TOGETHER_STEPS;
        if ids.len() < needs {
            return Err(format!("The text has {} tokens: it needs {needs}.", ids.len()));
        }
        let m = Gemma4Cuda::load(&file, Gpu::open().map_err(e)?).map_err(e)?;
        let vocab = m.config.vocab;
        let mut sessions = Vec::new();
        for c in 0..chats {
            let mut s = m.session(depth + 512).map_err(e)?;
            let from = c * TOGETHER_APART;
            m.forward_batch(&mut s, &ids[from..from + depth], Logits::Last).map_err(e)?;
            sessions.push(s);
        }
        let mut work = m.session_with_batch(chats.max(2), chats).map_err(e)?;
        let next = |c: usize, step: usize| ids[c * TOGETHER_APART + depth + step];

        // The same logits as alone.
        let mut alone = Vec::new();
        for (c, s) in sessions.iter_mut().enumerate() {
            alone.push(m.forward(s, next(c, 0), true).map_err(e)?.unwrap().to_vec());
            if s.truncate(depth) != depth {
                return Err("The session couldn't go back over a token.".into());
            }
        }
        let tokens: Vec<u32> = (0..chats).map(|c| next(c, 0)).collect();
        let mut refs: Vec<&mut _> = sessions.iter_mut().collect();
        m.forward_each(&mut work, &mut refs, &tokens).map_err(e)?;
        let mut row = vec![0.0f32; vocab];
        // The ten likeliest tokens alone: what a reply is picked from.
        let top = |alone: &[f32]| {
            let mut order: Vec<usize> = (0..alone.len()).collect();
            order.select_nth_unstable_by(10, |&a, &b| alone[b].total_cmp(&alone[a]));
            order.truncate(10);
            order
        };
        let (mut same, mut worst, mut worst_top) = (0, 0.0f32, 0.0f32);
        for (c, alone) in alone.iter().enumerate() {
            m.held_logits(&work, c, &mut row).map_err(e)?;
            same += usize::from(argmax(&row) == argmax(alone));
            worst = row.iter().zip(alone).fold(worst, |w, (a, b)| w.max((a - b).abs()));
            worst_top = top(alone).iter().fold(worst_top, |w, &i| w.max((row[i] - alone[i]).abs()));
        }
        println!(
            "Together against alone at {depth} tokens: the same likeliest token for {same} of {chats} chats; logits differ by {worst:.4} at most, {worst_top:.4} among each chat's ten likeliest."
        );
        for s in &mut sessions {
            s.truncate(depth);
        }
        // For scale: each chat's token as the first of a batch of as many of its own
        // tokens as there are chats (the same kernels, one chat's cache) against alone.
        let (mut same, mut worst, mut worst_top) = (0, 0.0f32, 0.0f32);
        for (c, alone) in alone.iter().enumerate() {
            let s = &mut sessions[c];
            let from = c * TOGETHER_APART + depth;
            let tokens = &ids[from..from + chats.min(TOGETHER_STEPS)];
            let batch = m.forward_batch(s, tokens, Logits::All).map_err(e)?.unwrap();
            let batch = &batch[..vocab];
            same += usize::from(argmax(batch) == argmax(alone));
            worst = batch.iter().zip(alone).fold(worst, |w, (a, b)| w.max((a - b).abs()));
            worst_top = top(alone).iter().fold(worst_top, |w, &i| w.max((batch[i] - alone[i]).abs()));
            s.truncate(depth);
        }
        println!(
            "One chat's batch against alone, for scale: the same likeliest token for {same} of {chats}; logits differ by {worst:.4} at most, {worst_top:.4} among the ten likeliest."
        );

        let mut n = 1;
        let mut cells = Vec::new();
        while n <= chats {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for step in 0..TOGETHER_STEPS {
                    let tokens: Vec<u32> = (0..n).map(|c| next(c, step)).collect();
                    let mut refs: Vec<&mut _> = sessions[..n].iter_mut().collect();
                    m.forward_each(&mut work, &mut refs, &tokens).map_err(e)?;
                    for c in 0..n {
                        m.held_logits(&work, c, &mut row).map_err(e)?;
                    }
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e3 / TOGETHER_STEPS as f64);
                for s in &mut sessions[..n] {
                    s.truncate(depth);
                }
            }
            cells.push(format!("{n}: {best:.2} ms ({:.1} each, {:.0} in all)", 1e3 / best, n as f64 * 1e3 / best));
            // Every number up to 8 (where a GPU's kernels change shape), then doubling.
            n = if n < 8 { n + 1 } else { n * 2 };
        }
        // One chat the usual way, for the cost of going through `forward_each` at all.
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for step in 0..TOGETHER_STEPS {
                m.forward(&mut sessions[0], next(0, step), true).map_err(e)?;
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / TOGETHER_STEPS as f64);
            sessions[0].truncate(depth);
        }
        println!("Chats written together at {depth} tokens, a step (tokens/s): {}", cells.join("  "));
        println!("One chat alone, the usual way: {best:.2} ms ({:.1} tokens/s).", 1e3 / best);
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn argmax(values: &[f32]) -> u32 {
    // The largest value first (a branch-free pass), then the first place it's at.
    let top = values.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    values.iter().position(|&v| v == top).unwrap_or(0) as u32
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
