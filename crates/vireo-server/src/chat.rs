//! Each model family's chat format, written out from the template in its GGUF file
//! (`tokenizer.chat_template`) for plain chats: a system prompt, then user and assistant
//! turns, thinking off. And each family's recommended sampling, from
//! `server/model_profiles.py`.

use crate::sample::Settings;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Llama,
    Qwen35,
    Gemma4,
    /// Muse Glimmer: Gemma 4's layer with other choices, run by the same model code
    /// (`Gemma4Cuda`, `Config::muse`), with a chat format of its own.
    Muse,
}

impl Family {
    /// The family's name in a reply's record of its tool calls ([`Record::family`]).
    pub fn name(self) -> &'static str {
        match self {
            Family::Llama => "llama",
            Family::Qwen35 => "qwen35",
            Family::Gemma4 => "gemma4",
            Family::Muse => "muse",
        }
    }

    /// Whether the family thinks in a part of the reply set apart from its answer, which
    /// a request can turn on (`thinking`): Gemma 4's thought channel, Qwen's `<think>`.
    /// (Muse always reasons, to itself; thinking raises its strength.)
    pub fn thinks_aloud(self) -> bool {
        matches!(self, Family::Gemma4 | Family::Qwen35)
    }

    pub fn of(architecture: Option<&str>) -> Option<Self> {
        match architecture? {
            "llama" => Some(Family::Llama),
            "qwen35" => Some(Family::Qwen35),
            "gemma4" => Some(Family::Gemma4),
            "muse-glimmer" => Some(Family::Muse),
            _ => None,
        }
    }

    /// Recommended sampling: Gemma 4 and Muse 1.0 / 0.95 / 64, others 0.7 / 0.9. Muse's
    /// generation_config.json asks for greedy decoding; its model card's settings are
    /// these. Qwen's cards (3.5 and 3.8 alike) give a set for each mode: thinking 1.0 /
    /// 0.95 / top-k 20, thinking off ("instruct") 0.7 / 0.8 / top-k 20. (0.6 / 0.95, used
    /// for both until 2026-11-15, is Qwen 3.5's set for precise coding with thinking on;
    /// 3.8's card has none.) Qwen 3.8 also gets its card's presence penalty with thinking
    /// off, 1.5 (0 with it on); Qwen 3.5's, 1.5 in both modes, isn't applied.
    pub fn sampling(self, thinking: bool) -> Settings {
        match self {
            Family::Qwen35 if thinking => Settings {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                ..Settings::default()
            },
            Family::Qwen35 => Settings {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
                presence_penalty: if qwen_keeps_thought() { 1.5 } else { 0.0 },
                ..Settings::default()
            },
            Family::Gemma4 | Family::Muse => Settings {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 64,
                ..Settings::default()
            },
            Family::Llama => Settings {
                temperature: 0.7,
                top_p: 0.9,
                ..Settings::default()
            },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// A reply's tool calls and what they returned, as the app kept them from the reply
    /// (`turn_record`) and sent them back: written into the history where they were made,
    /// so a later turn still has what an earlier one read.
    pub record: Option<Record>,
}

/// The tool calls of one reply (see [`Message::record`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    /// The family whose form `call` is written in ([`Family::name`]). A chat that goes on
    /// with a model of another family gets its results as plain notes instead.
    pub family: String,
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    /// Where in the reply's text (thinking left out, trimmed) the call was made, in bytes.
    pub at: usize,
    pub tool: String,
    /// The call as the model wrote it, from its start token.
    pub call: String,
    /// Its result as the model read it.
    pub result: String,
    /// Whether the model wrote the next step's call straight after this one, before any
    /// result (Gemma 4's several calls in one step): both results then follow the calls.
    pub then: bool,
}

/// How much of earlier turns' tool results a prompt keeps: all of each, the start of
/// each, or none (the reply's text alone, as before 2026-11-13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Results {
    Whole,
    Short,
    Left,
}

/// The most of a result kept at [`Results::Short`], in characters.
const SHORT_RESULT: usize = 1200;

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Message {
            role: role.into(),
            content: content.into(),
            record: None,
        }
    }

    /// Its size in the prompt, in bytes, near enough: its text and its tools' results.
    pub fn bytes(&self) -> usize {
        self.content.len()
            + self.record.as_ref().map_or(0, |r| {
                r.steps.iter().map(|s| s.call.len() + s.result.len()).sum()
            })
    }

    /// The message with this much of its tools' results.
    pub fn with_results(&self, keep: Results) -> Message {
        let mut m = self.clone();
        match keep {
            Results::Whole => {}
            Results::Left => m.record = None,
            Results::Short => {
                for step in m.record.iter_mut().flat_map(|r| r.steps.iter_mut()) {
                    let chars = step.result.chars().count();
                    if chars > SHORT_RESULT {
                        let kept: String = step.result.chars().take(SHORT_RESULT).collect();
                        step.result = format!(
                            "{kept}\n[... {} more characters no longer kept: call the tool again for them]",
                            chars - SHORT_RESULT
                        );
                    }
                }
            }
        }
        m
    }
}

impl Record {
    /// A record as the app holds it: the JSON `to_json` wrote. `None` if it isn't one.
    pub fn parse(text: &str) -> Option<Record> {
        let v = crate::json::parse(text).ok()?;
        let steps: Vec<Step> = v
            .array("steps")
            .iter()
            .filter_map(|s| {
                Some(Step {
                    at: s.num("at").unwrap_or(0.0).max(0.0) as usize,
                    tool: s.str("tool")?.to_string(),
                    call: plain(s.str("call")?),
                    result: plain(s.str("result")?),
                    then: matches!(s.get("then"), Some(crate::json::Value::Bool(true))),
                })
            })
            .collect();
        (!steps.is_empty()).then(|| Record {
            family: v.str("family").unwrap_or("").to_string(),
            steps,
        })
    }

    pub fn to_json(&self) -> String {
        let steps: Vec<String> = self
            .steps
            .iter()
            .map(|s| {
                format!(
                    "{{\"at\":{},\"tool\":{},\"call\":{},\"result\":{}{}}}",
                    s.at,
                    crate::json::string(&s.tool),
                    crate::json::string(&s.call),
                    crate::json::string(&s.result),
                    if s.then { ",\"then\":true" } else { "" }
                )
            })
            .collect();
        format!(
            "{{\"family\":{},\"steps\":[{}]}}",
            crate::json::string(&self.family),
            steps.join(",")
        )
    }
}

/// Marks around a stretch of a prompt that's read as plain text, so nothing in it can pass
/// for a special token: a tool's result written back into the history ([`encode`]). They
/// are private-use characters, taken out of everything that comes from outside ([`plain`]).
pub const PLAIN_OPEN: char = '\u{E000}';
pub const PLAIN_CLOSE: char = '\u{E001}';

/// Text without the marks [`encode`] reads.
pub fn plain(text: &str) -> String {
    if text.contains([PLAIN_OPEN, PLAIN_CLOSE]) {
        text.replace([PLAIN_OPEN, PLAIN_CLOSE], "")
    } else {
        text.to_string()
    }
}

/// A prompt's tokens: special tokens read as such, but for the stretches between
/// [`PLAIN_OPEN`] and [`PLAIN_CLOSE`], which are read as plain text (as a tool's result is
/// when the reply that called it reads it, so the chat's cache matches token for token).
pub fn encode(tokenizer: &vireo_engine::Tokenizer, text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(PLAIN_OPEN) {
        if open > 0 {
            ids.extend(tokenizer.encode(&rest[..open], false, true));
        }
        let after = &rest[open + PLAIN_OPEN.len_utf8()..];
        let close = after.find(PLAIN_CLOSE).unwrap_or(after.len());
        if close > 0 {
            ids.extend(tokenizer.encode(&after[..close], false, false));
        }
        rest = after.get(close + PLAIN_CLOSE.len_utf8()..).unwrap_or("");
    }
    if !rest.is_empty() {
        ids.extend(tokenizer.encode(rest, false, true));
    }
    ids
}

/// A reply's text with its tool calls and their results where they were made, as the
/// family's template writes a turn that called tools: what follows the turn's opening, up
/// to its closing. (Muse's from just after `<|start|>assistant`: each message names its
/// recipient.) For a record from another family, the results as plain notes.
fn with_calls(family: Family, content: &str, record: &Record) -> String {
    let same = record.family == family.name();
    let mut out = String::new();
    let mut done = 0;
    let say = |out: &mut String, text: &str, last: bool| match family {
        // Muse's text is a message to the user; one with a call after it ends in <|eom|>.
        Family::Muse if same => {
            if last {
                out.push_str(" to=user<|message|>");
                out.push_str(text);
            } else if !text.trim().is_empty() {
                out.push_str(" to=user<|message|>");
                out.push_str(text);
                out.push_str("<|eom|><|start|>assistant");
            }
        }
        _ => out.push_str(text),
    };
    // Calls written together, waiting for the last of them: their results follow it.
    let mut waiting: Vec<&Step> = Vec::new();
    for step in &record.steps {
        let at = if step.at >= done && step.at <= content.len() && content.is_char_boundary(step.at)
        {
            step.at
        } else {
            done
        };
        say(&mut out, &content[done..at], false);
        done = at;
        if !same {
            out.push_str(&format!("\n[{} returned:\n", step.tool));
            out.push(PLAIN_OPEN);
            out.push_str(&step.result);
            out.push(PLAIN_CLOSE);
            out.push_str("]\n");
            continue;
        }
        let grouped = family == Family::Gemma4;
        if grouped && (step.then || !waiting.is_empty()) {
            out.push_str(&step.call);
            out.push_str(crate::tools::GEMMA_CALL_CLOSE);
            waiting.push(step);
            if step.then {
                continue;
            }
            for s in waiting.drain(..) {
                let (before, body, after) =
                    crate::tools::after_call(family, &s.tool, &s.result, false);
                out.push_str(
                    before
                        .strip_prefix(crate::tools::GEMMA_CALL_CLOSE)
                        .unwrap_or(&before),
                );
                out.push(PLAIN_OPEN);
                out.push_str(&body);
                out.push(PLAIN_CLOSE);
                out.push_str(&after);
            }
            continue;
        }
        let (before, body, after) =
            crate::tools::after_call(family, &step.tool, &step.result, false);
        // A past turn has no thought in it, not even Qwen's empty one, unless the
        // template keeps them (Qwen 3.8).
        let after = if qwen_keeps_thought() {
            &after[..]
        } else {
            after.strip_suffix(QWEN_EMPTY_THOUGHT).unwrap_or(&after)
        };
        out.push_str(&step.call);
        out.push_str(&before);
        out.push(PLAIN_OPEN);
        out.push_str(&body);
        out.push(PLAIN_CLOSE);
        out.push_str(after);
    }
    // A group cut short (the reply was stopped): its results as they came.
    for s in waiting.drain(..) {
        let (before, body, after) = crate::tools::after_call(family, &s.tool, &s.result, false);
        out.push_str(
            before
                .strip_prefix(crate::tools::GEMMA_CALL_CLOSE)
                .unwrap_or(&before),
        );
        out.push(PLAIN_OPEN);
        out.push_str(&body);
        out.push(PLAIN_CLOSE);
        out.push_str(&after);
    }
    say(&mut out, &content[done..], true);
    out
}

/// Whether a Gemma 4's turn opens with the empty thought when thinking is off: set from
/// the model's chat template when it's loaded (`follow_template`).
static GEMMA_EMPTY_THOUGHT: AtomicBool = AtomicBool::new(true);

/// Whether a Qwen's template keeps the thought in a past reply's turn (Qwen 3.8's does
/// unless told otherwise, `preserve_thinking`; Qwen 3.5's leaves past thoughts out). The
/// app keeps a reply's answer, not its thinking, so a past turn is written with the
/// empty thought, as the template writes one whose thinking wasn't sent back. With
/// thinking off that is the turn as the model wrote it, token for token: the next prompt
/// goes on from the end of the chat's cache and the last reply isn't read again.
static QWEN_KEEPS_THOUGHT: AtomicBool = AtomicBool::new(false);
const QWEN_EMPTY_THOUGHT: &str = "<think>\n\n</think>\n\n";

/// Notes what the loaded Qwen's chat template does with past thoughts.
pub fn note_qwen_template(template: Option<&str>) {
    QWEN_KEEPS_THOUGHT.store(
        template.is_some_and(|t| t.contains("preserve_thinking")),
        Ordering::Relaxed,
    );
}

/// What Qwen 3.8's template opens the system turn with when thinking is on and no
/// `reasoning_effort` is given (its default, xhigh; medium has no line, low another).
const QWEN_EFFORT: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

/// Whether the loaded Qwen is one whose template keeps past thoughts, names a reasoning
/// effort and whose card asks for a presence penalty: Qwen 3.8.
pub fn qwen_keeps_thought() -> bool {
    QWEN_KEEPS_THOUGHT.load(Ordering::Relaxed)
}

/// Follows the loaded model's chat template on the one thing it decides here: whether a
/// Gemma 4's turn, thinking off, opens with the empty thought. Gemma 4 31B's template
/// writes it; E2B's and E4B's open the turn with `<|turn>model\n` alone, and with the
/// empty thought written for them they wrote their reasoning as the reply (Windows eval,
/// 2026-11-13: E4B 39 -> 35 of 43, twice the tokens). No template: the empty thought.
pub fn follow_template(template: Option<&str>) {
    GEMMA_EMPTY_THOUGHT.store(template.is_none_or(writes_empty_thought), Ordering::Relaxed);
}

/// Whether a chat template writes Gemma's empty thought (`<|channel>thought` then
/// `<channel|>`, with the newline between as Jinja's `\n` or as itself).
fn writes_empty_thought(template: &str) -> bool {
    template.contains("<|channel>thought\\n<channel|>")
        || template.contains("<|channel>thought\n<channel|>")
}

/// The prompt for `messages` (oldest first) with the turn for the model's reply opened.
/// Special tokens are written as text, for `encode(.., parse_special: true)`.
#[cfg(test)]
pub fn prompt(family: Family, system: Option<&str>, messages: &[Message], today: &str) -> String {
    history(family, system, "", messages, today, false) + opening(family, false)
}

/// The turn for the model's reply, opened: what `prompt` adds after `history`. It
/// starts with a special token on every family, so the two can be tokenized apart.
/// The next turn's prompt writes the reply differently (Qwen's empty thought isn't
/// kept), so a chat's prompts can share their history and nothing after it.
pub fn opening(family: Family, thinking: bool) -> &'static str {
    match family {
        // Thinking off, for a Gemma whose template opens its turn with no thought
        // (Gemma 4 E2B and E4B): the turn as the template opens it.
        Family::Gemma4 if !thinking && !GEMMA_EMPTY_THOUGHT.load(Ordering::Relaxed) => {
            "<|turn>model\n"
        }
        Family::Llama => "<|start_header_id|>assistant<|end_header_id|>\n\n",
        // Thinking on: the thought is opened for the model, as the template opens it.
        Family::Qwen35 if thinking => "<|im_start|>assistant\n<think>\n",
        // Thinking off: the template's empty thought.
        Family::Qwen35 => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        // Thinking on, the model opens its own thought (the system turn says it may).
        Family::Gemma4 if thinking => "<|turn>model\n",
        // Thinking off: the template's empty thought (in the GGUF's template and Google's
        // July 2026 one alike), so the model goes straight to its answer.
        Family::Gemma4 => "<|turn>model\n<|channel>thought\n<channel|>",
        // Muse's turn names its recipient next: the reply loop has the model choose
        // between the user and a tool (`tools::call_grammar`).
        Family::Muse => "<|start|>assistant",
    }
}

/// The reply's turn opened where the server writes a call in before the model writes (the
/// skill it picked): no thought comes before that call. For Gemma 4 with thinking on that
/// is the bare turn, as its template writes a turn that starts with a call. The empty
/// thought is the template's sign that thinking is off, and written here it switched
/// thinking off for the whole reply: in the Nov 15 test gauntlet (Gemma 4 31B, thinking
/// on) 8 of 21 replies that followed a skill had no reasoning at all, against none of the
/// 11 without a skill, and 7 of those 8 were graded partial or fail.
pub fn opening_for_call(family: Family, thinking: bool) -> &'static str {
    match family {
        Family::Gemma4 if thinking => "<|turn>model\n",
        other => opening(other, false),
    }
}

/// What ends a message in `history`: its last message's text stops this far from the
/// end, which is where a note for this turn only can go (`memory::since`).
pub fn closing(family: Family) -> &'static str {
    match family {
        Family::Llama => "<|eot_id|>",
        Family::Qwen35 => "<|im_end|>\n",
        Family::Gemma4 => "<turn|>\n",
        Family::Muse => "<|eot|>",
    }
}

/// Where a chat too long for the context may start instead of at its first message:
/// each a user's message, the first one with `step` bytes of messages before it, then
/// with twice that, and so on. They depend only on the messages before them, so a chat
/// keeps the same start (and its cache) turn after turn, until it has grown by about a
/// step and moves on to the next.
pub fn cut_points(messages: &[Message], step: usize) -> Vec<usize> {
    let step = step.max(1);
    let mut cuts = Vec::new();
    let (mut before, mut k) = (0usize, 1usize);
    for (i, m) in messages.iter().enumerate() {
        if m.role == "user" && before >= k * step {
            cuts.push(i);
            k = before / step + 1;
        }
        before += m.bytes();
    }
    cuts
}

/// The reply's turn opened for a plain answer (for a question the server asks the model
/// itself): `opening`, with Muse's recipient the user.
pub fn answer_opening(family: Family) -> &'static str {
    match family {
        Family::Muse => "<|start|>assistant to=user<|message|>",
        other => opening(other, false),
    }
}

/// Muse thinks to itself in a message `to=self`; its template's switch is set as the
/// Mac's server sets it: low for thinking off, high for thinking on.
pub const MUSE_REASONING: &str = "Reasoning strength: low.";
pub const MUSE_REASONING_ON: &str = "Reasoning strength: high.";

/// What Gemma 4's template opens its system turn with when thinking is on.
const GEMMA_THINK: &str = "<|think|>\n";

/// What the Mac's server adds to a system prompt of Vireo's for Muse (model_profiles.py's
/// `system_note`): the knowledge cutoff its own default system turn gives, and a line for
/// replies it would otherwise run on.
pub const MUSE_NOTE: &str = "Knowledge cutoff: January 4, 2026.\n\nKeep the final reply short: the result, then the points the user must know. Do not restate rules, file contents or your steps, do not list the sources again, and do not offer a menu of next steps. Never shorten a draft, a document or a list the user asked for.";

/// The system prompt, the tools' declarations (`tools::declarations`, or "" for none) and
/// `messages`, without the reply's turn. The declarations go where each template puts
/// them, in the system turn, so a chat's prompts all start the same way. `thinking` is
/// the request's: Gemma 4's template says so in the system turn, Muse's sets its strength.
/// Read it with [`encode`]: a reply's kept tool results are marked as plain text.
pub fn history(
    family: Family,
    system: Option<&str>,
    tools: &str,
    messages: &[Message],
    today: &str,
    thinking: bool,
) -> String {
    let system = system.map(str::trim).filter(|s| !s.is_empty());
    let mut out = String::new();
    match family {
        Family::Llama => {
            out.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
            if !tools.is_empty() {
                out.push_str("Environment: ipython\n");
            }
            out.push_str("Cutting Knowledge Date: December 2023\nToday Date: ");
            out.push_str(today);
            out.push_str("\n\n");
            out.push_str(tools);
            out.push_str(system.unwrap_or(""));
            out.push_str("<|eot_id|>");
            for m in messages {
                out.push_str("<|start_header_id|>");
                out.push_str(&m.role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(&said(family, m));
                out.push_str("<|eot_id|>");
            }
        }
        Family::Qwen35 => {
            // Qwen 3.8, thinking on: the system turn opens with the reasoning effort.
            let effort = thinking && qwen_keeps_thought();
            if !tools.is_empty() {
                out.push_str("<|im_start|>system\n");
                if effort {
                    out.push_str(QWEN_EFFORT);
                    out.push_str("\n\n");
                }
                out.push_str(tools);
                if let Some(s) = system {
                    out.push_str("\n\n");
                    out.push_str(s);
                }
                out.push_str("<|im_end|>\n");
            } else if let Some(s) = system {
                out.push_str("<|im_start|>system\n");
                if effort {
                    out.push_str(QWEN_EFFORT);
                    out.push_str("\n\n");
                }
                out.push_str(s);
                out.push_str("<|im_end|>\n");
            } else if effort {
                out.push_str("<|im_start|>system\n");
                out.push_str(QWEN_EFFORT);
                out.push_str("<|im_end|>\n");
            }
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                if m.role == "assistant" && qwen_keeps_thought() {
                    out.push_str(QWEN_EMPTY_THOUGHT);
                }
                out.push_str(&said(family, m));
                out.push_str("<|im_end|>\n");
            }
        }
        Family::Gemma4 => {
            out.push_str("<bos>");
            if system.is_some() || !tools.is_empty() || thinking {
                out.push_str("<|turn>system\n");
                if thinking {
                    out.push_str(GEMMA_THINK);
                }
                out.push_str(system.unwrap_or(""));
                out.push_str(tools);
                out.push_str("<turn|>\n");
            }
            for m in messages {
                let role = if m.role == "assistant" {
                    "model"
                } else {
                    "user"
                };
                out.push_str("<|turn>");
                out.push_str(role);
                out.push('\n');
                out.push_str(&said(family, m));
                out.push_str("<turn|>\n");
            }
        }
        Family::Muse => {
            out.push_str("<|begin_of_text|><|start|>system<|message|>");
            match system {
                Some(s) => {
                    out.push_str(s);
                    out.push_str("\n\n");
                    out.push_str(MUSE_NOTE);
                }
                // The template's own system turn, when the chat has none.
                None => {
                    out.push_str(
                        "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: ",
                    );
                    out.push_str(&iso_date(today));
                    out.push('.');
                }
            }
            out.push_str("\n\n");
            out.push_str(if thinking {
                MUSE_REASONING_ON
            } else {
                MUSE_REASONING
            });
            out.push_str("\n\n");
            if tools.is_empty() {
                out.push_str("# Valid recipients: \"self\", \"user\".");
            } else {
                // The tools' declarations end with the recipients they add.
                out.push_str(tools);
            }
            out.push_str("<|eot|>");
            for m in messages {
                match (&m.record, m.role == "assistant") {
                    (Some(record), true) if record.family == family.name() => {
                        out.push_str("<|start|>assistant");
                        out.push_str(&with_calls(family, m.content.trim(), record));
                    }
                    (_, true) => {
                        out.push_str("<|start|>assistant to=user<|message|>");
                        out.push_str(&said(family, m));
                    }
                    (_, false) => {
                        out.push_str("<|start|>user<|message|>");
                        out.push_str(m.content.trim());
                    }
                }
                out.push_str("<|eot|>");
            }
        }
    }
    out
}

/// A message's text as its turn holds it: trimmed, and a reply that called tools with its
/// calls and their results ([`with_calls`]).
fn said(family: Family, m: &Message) -> String {
    match &m.record {
        Some(record) if m.role == "assistant" => with_calls(family, m.content.trim(), record),
        _ => m.content.trim().to_string(),
    }
}

/// "2026-11-12" (the Muse template's `strftime_now('%Y-%m-%d')`) from `today`'s
/// "12 Nov 2026"; `today` itself if it isn't one.
fn iso_date(today: &str) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let parts: Vec<&str> = today.split(' ').collect();
    match parts[..] {
        [day, month, year] => match MONTHS.iter().position(|m| *m == month) {
            Some(i) => format!("{year}-{:02}-{day}", i + 1),
            None => today.to_string(),
        },
        _ => today.to_string(),
    }
}

/// An earlier reply without any thinking in it (the templates leave past thoughts out).
pub fn without_thinking(text: &str) -> String {
    let mut out = text.to_string();
    for (open, close) in [("<think>", "</think>"), ("<|channel>", "<channel|>")] {
        while let Some(start) = out.find(open) {
            match out[start..].find(close) {
                Some(end) => out.replace_range(start..start + end + close.len(), ""),
                None => out.truncate(start),
            }
        }
        if let Some(end) = out.find(close) {
            out.replace_range(..end + close.len(), "");
        }
    }
    out.trim().to_string()
}

/// "11 Nov 2026", as the Llama 3 template's `strftime_now("%d %b %Y")`.
pub fn today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0) as i64;
    // Days to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!("{day:02} {} {year}", MONTHS[(month - 1) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_a_call_is_written_into_opens_without_the_thinking_off_sign() {
        // Thinking on: the bare turn, so the thoughts after the call are still the model's.
        assert_eq!(opening_for_call(Family::Gemma4, true), "<|turn>model\n");
        // Thinking off, and the other families: as a turn with no thought opens.
        for family in [Family::Gemma4, Family::Qwen35, Family::Llama, Family::Muse] {
            assert_eq!(opening_for_call(family, false), opening(family, false));
        }
        assert_eq!(opening_for_call(Family::Qwen35, true), opening(Family::Qwen35, false));
    }

    fn said(role: &str, bytes: usize) -> Message {
        Message::new(role, "x".repeat(bytes))
    }

    #[test]
    fn a_long_chat_starts_at_the_same_message_as_it_grows() {
        // 60 bytes a turn, a step of 100: a cut at the first user message with 100
        // bytes before it, the next with 200, and so on.
        let mut chat = Vec::new();
        for _ in 0..6 {
            chat.push(said("user", 20));
            chat.push(said("assistant", 40));
        }
        chat.push(said("user", 20));
        assert_eq!(cut_points(&chat, 100), [4, 8, 10]);
        // Two turns on, the earlier ones are where they were, and there's one more.
        for _ in 0..2 {
            chat.push(said("assistant", 40));
            chat.push(said("user", 20));
        }
        assert_eq!(cut_points(&chat, 100), [4, 8, 10, 14]);
        // One long message passes several steps: one cut after it, not one for each.
        let chat = [
            said("user", 350),
            said("assistant", 10),
            said("user", 5),
            said("assistant", 60),
            said("user", 5),
        ];
        assert_eq!(cut_points(&chat, 100), [2, 4]);
        assert!(cut_points(&chat[..1], 100).is_empty());
        assert!(cut_points(&[], 100).is_empty());
    }

    #[test]
    fn every_history_ends_with_its_closing() {
        for family in [Family::Llama, Family::Qwen35, Family::Gemma4, Family::Muse] {
            let mut chat = chat();
            chat.truncate(1);
            let text = history(family, Some("Be brief."), "", &chat, "12 Nov 2026", false);
            let end = text.len() - closing(family).len();
            assert_eq!(&text[end..], closing(family), "{family:?}");
            assert!(text[..end].ends_with("Hi"), "{family:?}");
            // A longer chat starts with a shorter one: its first messages' tokens can
            // be counted apart.
            let longer = history(
                family,
                Some("Be brief."),
                "",
                &self::chat(),
                "12 Nov 2026",
                false,
            );
            assert!(longer.starts_with(&text), "{family:?}");
        }
    }

    fn chat() -> Vec<Message> {
        vec![
            Message::new("user", "Hi"),
            Message::new("assistant", "Hello!"),
            Message::new("user", " And you? "),
        ]
    }

    #[test]
    fn writes_each_familys_turns() {
        assert_eq!(
            prompt(Family::Qwen35, Some("Be brief."), &chat(), "x"),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nAnd you?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            prompt(Family::Gemma4, None, &chat(), "x"),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nAnd you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        assert!(
            prompt(Family::Llama, None, &chat(), "11 Nov 2026").starts_with(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 Cutting Knowledge Date: December 2023\nToday Date: 11 Nov 2026\n\n<|eot_id|>"
            )
        );
    }

    #[test]
    fn writes_muses_turns_as_its_template_does() {
        // No system prompt: the template's own system turn, with the date.
        assert_eq!(
            prompt(Family::Muse, None, &chat(), "12 Nov 2026"),
            "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n\
             Knowledge cutoff: 2026-01-04.\nCurrent date: 2026-11-12.\n\nReasoning strength: low.\n\n\
             # Valid recipients: \"self\", \"user\".<|eot|>\
             <|start|>user<|message|>Hi<|eot|><|start|>assistant to=user<|message|>Hello!<|eot|>\
             <|start|>user<|message|>And you?<|eot|><|start|>assistant"
        );
        // A system prompt replaces the template's, with the Mac's note after it; the tools'
        // declarations bring their recipients.
        let text = history(
            Family::Muse,
            Some("Be brief."),
            "TOOLS",
            &chat()[..1],
            "x",
            false,
        );
        assert_eq!(
            text,
            format!(
                "<|begin_of_text|><|start|>system<|message|>Be brief.\n\n{MUSE_NOTE}\n\n\
                 Reasoning strength: low.\n\nTOOLS<|eot|><|start|>user<|message|>Hi<|eot|>"
            )
        );
        assert_eq!(
            answer_opening(Family::Muse),
            "<|start|>assistant to=user<|message|>"
        );
        assert_eq!(answer_opening(Family::Gemma4), opening(Family::Gemma4, false));
        assert_eq!(Family::of(Some("muse-glimmer")), Some(Family::Muse));
        assert_eq!(iso_date("05 Jan 2027"), "2027-01-05");
    }

    #[test]
    fn qwen_sampling_follows_the_mode() {
        let (on, off) = (Family::Qwen35.sampling(true), Family::Qwen35.sampling(false));
        assert_eq!((on.temperature, on.top_p, on.top_k), (1.0, 0.95, 20));
        assert_eq!((off.temperature, off.top_p, off.top_k), (0.7, 0.8, 20));
        // The others have one set.
        let g = Family::Gemma4;
        assert_eq!(g.sampling(true).temperature, g.sampling(false).temperature);
    }

    #[test]
    fn leaves_out_past_thinking() {
        assert_eq!(without_thinking("<think>\nhmm\n</think>\n\nYes."), "Yes.");
        assert_eq!(
            without_thinking("<|channel>thought\nx<channel|>Sure"),
            "Sure"
        );
        assert_eq!(without_thinking("plain"), "plain");
    }

    #[test]
    fn dates_like_strftime() {
        let t = today();
        assert_eq!(t.len(), 11, "{t}");
    }

    #[test]
    fn thinking_is_switched_on_as_each_template_does() {
        let chat = &chat()[..1];
        assert_eq!(
            history(Family::Gemma4, Some("Be brief."), "", chat, "x", true),
            "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n"
        );
        // Gemma's template writes the system turn for thinking alone.
        assert!(
            history(Family::Gemma4, None, "", chat, "x", true)
                .starts_with("<bos><|turn>system\n<|think|>\n<turn|>\n")
        );
        assert_eq!(
            opening(Family::Qwen35, true),
            "<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(opening(Family::Gemma4, true), "<|turn>model\n");
        assert_eq!(
            opening(Family::Gemma4, false),
            "<|turn>model\n<|channel>thought\n<channel|>"
        );
        assert!(
            history(Family::Muse, None, "", chat, "x", true).contains("Reasoning strength: high.")
        );
    }

    #[test]
    fn a_gemma_template_decides_the_empty_thought() {
        // 31B's: the generation prompt writes the empty thought.
        assert!(writes_empty_thought(
            "{{- '<|turn>model\\n' -}}{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}"
        ));
        // E2B's and E4B's: the turn alone, a thought only after a tool's response.
        assert!(!writes_empty_thought(
            "{{- '<|turn>model\\n' -}}{%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}{{- '<|channel>thought\\n' -}}"
        ));
    }

    fn read_step(at: usize, call: &str) -> Step {
        Step {
            at,
            tool: "read_file".into(),
            call: call.into(),
            result: "<turn|> 42".into(),
            then: false,
        }
    }

    #[test]
    fn a_replys_tool_results_are_written_back_where_they_were() {
        let record = |family: Family, call: &str| Record {
            family: family.name().into(),
            steps: vec![read_step(0, call)],
        };
        let mut chat = chat();
        chat[1].content = "It says 42.".into();
        chat[1].record = Some(record(
            Family::Gemma4,
            "<|tool_call>call:read_file{path:<|\"|>a.txt<|\"|>}",
        ));
        let text = history(Family::Gemma4, None, "", &chat, "x", false);
        assert_eq!(
            text,
            format!(
                "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|tool_call>call:read_file{{path:<|\"|>a.txt<|\"|>}}\
                 <tool_call|><|tool_response>response:read_file{{value:<|\"|>{PLAIN_OPEN}<turn|> 42{PLAIN_CLOSE}<|\"|>}}\
                 <tool_response|>It says 42.<turn|>\n<|turn>user\nAnd you?<turn|>\n"
            )
        );
        // Qwen's past turn has no thought after the result.
        chat[1].record = Some(record(
            Family::Qwen35,
            "<tool_call>\n<function=read_file>\n</function>\n",
        ));
        let text = history(Family::Qwen35, None, "", &chat, "x", false);
        assert!(
            text.contains(&format!(
                "<|im_start|>assistant\n<tool_call>\n<function=read_file>\n</function>\n</tool_call><|im_end|>\n\
                 <|im_start|>user\n<tool_response>\n{PLAIN_OPEN}<turn|> 42{PLAIN_CLOSE}\n</tool_response><|im_end|>\n\
                 <|im_start|>assistant\nIt says 42.<|im_end|>\n"
            )),
            "{text}"
        );
        // Muse names each message's recipient.
        chat[1].record = Some(record(Family::Muse, " to=read_file<|message|>CALL"));
        let text = history(Family::Muse, None, "", &chat, "x", false);
        assert!(
            text.contains(&format!(
                "<|start|>assistant to=read_file<|message|>CALL<|eot|><|start|>tool read_file<|message|>\
                 <tool_output name=\"read_file\">\n{PLAIN_OPEN}<turn|> 42{PLAIN_CLOSE}\n</tool_output><|eot|>\
                 <|start|>assistant to=user<|message|>It says 42.<|eot|>"
            )),
            "{text}"
        );
        // Another family's record: the result as a note, no call in a form it never wrote.
        let text = history(Family::Gemma4, None, "", &chat, "x", false);
        assert!(
            text.contains(&format!(
                "<|turn>model\n\n[read_file returned:\n{PLAIN_OPEN}<turn|> 42{PLAIN_CLOSE}]\nIt says 42.<turn|>"
            )),
            "{text}"
        );
    }

    #[test]
    fn a_call_goes_back_where_the_reply_made_it() {
        let mut m = Message::new("assistant", "One moment. Done.");
        m.record = Some(Record {
            family: "gemma4".into(),
            steps: vec![read_step(11, "CALL"), read_step(999, "LATER")],
        });
        let text = super::said(Family::Gemma4, &m);
        assert!(text.starts_with("One moment.CALL<tool_call|>"), "{text}");
        // A place past the text (the app cut it short): straight after the last one.
        assert!(text.ends_with("<tool_response|> Done."), "{text}");
        assert!(text.contains("<tool_response|>LATER<tool_call|>"), "{text}");
    }

    #[test]
    fn calls_written_together_get_their_results_after_the_last() {
        let together = |at, call: &str, then| Step {
            then,
            ..read_step(at, call)
        };
        let mut chat = chat();
        chat[1].content = "Both say 42.".into();
        chat[1].record = Some(Record {
            family: "gemma4".into(),
            steps: vec![
                together(0, "<|tool_call>A", true),
                together(0, "<|tool_call>B", false),
                together(0, "<|tool_call>C", false),
            ],
        });
        let text = history(Family::Gemma4, None, "", &chat, "x", false);
        let result = format!(
            "<|tool_response>response:read_file{{value:<|\"|>{PLAIN_OPEN}<turn|> 42{PLAIN_CLOSE}<|\"|>}}<tool_response|>"
        );
        assert!(
            text.contains(&format!(
                "<|turn>model\n<|tool_call>A<tool_call|><|tool_call>B<tool_call|>{result}{result}\
                 <|tool_call>C<tool_call|>{result}Both say 42.<turn|>\n"
            )),
            "{text}"
        );
        // A group cut short still has its results.
        chat[1].record.as_mut().unwrap().steps.truncate(1);
        let text = history(Family::Gemma4, None, "", &chat, "x", false);
        assert!(
            text.contains(&format!("<|tool_call>A<tool_call|>{result}Both")),
            "{text}"
        );
        // The app keeps the mark.
        let record = chat[1].record.clone().unwrap();
        assert_eq!(Record::parse(&record.to_json()), Some(record));
    }

    #[test]
    fn a_record_goes_to_the_app_and_back() {
        let record = Record {
            family: "gemma4".into(),
            steps: vec![read_step(3, "call \"x\"\n")],
        };
        assert_eq!(Record::parse(&record.to_json()), Some(record.clone()));
        assert_eq!(Record::parse("{\"steps\":[]}"), None);
        assert_eq!(Record::parse("nonsense"), None);
        // Kept short, and left out.
        let mut m = Message::new("assistant", "Yes.");
        m.record = Some(Record {
            family: "gemma4".into(),
            steps: vec![Step {
                result: "y".repeat(5000),
                then: false,
                ..read_step(0, "CALL")
            }],
        });
        assert_eq!(m.bytes(), 4 + 4 + 5000);
        let short = m.with_results(Results::Short);
        let kept = &short.record.as_ref().unwrap().steps[0].result;
        assert!(kept.starts_with(&"y".repeat(SHORT_RESULT)) && kept.contains("3800 more characters"));
        assert_eq!(m.with_results(Results::Left).record, None);
        assert_eq!(plain(&format!("a{PLAIN_OPEN}b{PLAIN_CLOSE}")), "ab");
    }
}
