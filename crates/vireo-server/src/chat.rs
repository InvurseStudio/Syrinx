//! Each model family's chat format, written out from the template in its GGUF file
//! (`tokenizer.chat_template`) for plain chats: a system prompt, then user and assistant
//! turns, thinking off. And each family's recommended sampling, from
//! `server/model_profiles.py`.

use crate::sample::Settings;

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
    pub fn of(architecture: Option<&str>) -> Option<Self> {
        match architecture? {
            "llama" => Some(Family::Llama),
            "qwen35" => Some(Family::Qwen35),
            "gemma4" => Some(Family::Gemma4),
            "muse-glimmer" => Some(Family::Muse),
            _ => None,
        }
    }

    /// Recommended sampling (model_profiles.py: Qwen 0.6 / 0.95 / top-k 20, Gemma 4 and
    /// Muse 1.0 / 0.95 / 64, others 0.7 / 0.9). Muse's generation_config.json asks for
    /// greedy decoding; its model card's settings are these.
    pub fn sampling(self) -> Settings {
        match self {
            Family::Qwen35 => Settings {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
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

pub struct Message {
    pub role: String,
    pub content: String,
}

/// The prompt for `messages` (oldest first) with the turn for the model's reply opened.
/// Special tokens are written as text, for `encode(.., parse_special: true)`.
#[cfg(test)]
pub fn prompt(family: Family, system: Option<&str>, messages: &[Message], today: &str) -> String {
    history(family, system, "", messages, today) + opening(family)
}

/// The turn for the model's reply, opened: what `prompt` adds after `history`. It
/// starts with a special token on every family, so the two can be tokenized apart.
/// The next turn's prompt writes the reply differently (Qwen's empty thought isn't
/// kept), so a chat's prompts can share their history and nothing after it.
pub fn opening(family: Family) -> &'static str {
    match family {
        Family::Llama => "<|start_header_id|>assistant<|end_header_id|>\n\n",
        // Thinking off: the template's empty thought.
        Family::Qwen35 => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        Family::Gemma4 => "<|turn>model\n",
        // Muse's turn names its recipient next: the reply loop has the model choose
        // between the user and a tool (`tools::call_grammar`).
        Family::Muse => "<|start|>assistant",
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
        before += m.content.len();
    }
    cuts
}

/// The reply's turn opened for a plain answer (for a question the server asks the model
/// itself): `opening`, with Muse's recipient the user.
pub fn answer_opening(family: Family) -> &'static str {
    match family {
        Family::Muse => "<|start|>assistant to=user<|message|>",
        other => opening(other),
    }
}

/// Muse thinks to itself in a message `to=self`; thinking off, it isn't offered one, and
/// its template's switch is set as the Mac's server sets it for thinking off.
pub const MUSE_REASONING: &str = "Reasoning strength: low.";

/// What the Mac's server adds to a system prompt of Vireo's for Muse (model_profiles.py's
/// `system_note`): the knowledge cutoff its own default system turn gives, and a line for
/// replies it would otherwise run on.
pub const MUSE_NOTE: &str = "Knowledge cutoff: January 4, 2026.\n\nKeep the final reply short: the result, then the points the user must know. Do not restate rules, file contents or your steps, do not list the sources again, and do not offer a menu of next steps. Never shorten a draft, a document or a list the user asked for.";

/// The system prompt, the tools' declarations (`tools::declarations`, or "" for none) and
/// `messages`, without the reply's turn. The declarations go where each template puts
/// them, in the system turn, so a chat's prompts all start the same way.
pub fn history(
    family: Family,
    system: Option<&str>,
    tools: &str,
    messages: &[Message],
    today: &str,
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
                out.push_str(m.content.trim());
                out.push_str("<|eot_id|>");
            }
        }
        Family::Qwen35 => {
            if !tools.is_empty() {
                out.push_str("<|im_start|>system\n");
                out.push_str(tools);
                if let Some(s) = system {
                    out.push_str("\n\n");
                    out.push_str(s);
                }
                out.push_str("<|im_end|>\n");
            } else if let Some(s) = system {
                out.push_str("<|im_start|>system\n");
                out.push_str(s);
                out.push_str("<|im_end|>\n");
            }
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(m.content.trim());
                out.push_str("<|im_end|>\n");
            }
        }
        Family::Gemma4 => {
            out.push_str("<bos>");
            if system.is_some() || !tools.is_empty() {
                out.push_str("<|turn>system\n");
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
                out.push_str(m.content.trim());
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
            out.push_str(MUSE_REASONING);
            out.push_str("\n\n");
            if tools.is_empty() {
                out.push_str("# Valid recipients: \"self\", \"user\".");
            } else {
                // The tools' declarations end with the recipients they add.
                out.push_str(tools);
            }
            out.push_str("<|eot|>");
            for m in messages {
                if m.role == "assistant" {
                    out.push_str("<|start|>assistant to=user<|message|>");
                } else {
                    out.push_str("<|start|>user<|message|>");
                }
                out.push_str(m.content.trim());
                out.push_str("<|eot|>");
            }
        }
    }
    out
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

    fn said(role: &str, bytes: usize) -> Message {
        Message {
            role: role.into(),
            content: "x".repeat(bytes),
        }
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
            let text = history(family, Some("Be brief."), "", &chat, "12 Nov 2026");
            let end = text.len() - closing(family).len();
            assert_eq!(&text[end..], closing(family), "{family:?}");
            assert!(text[..end].ends_with("Hi"), "{family:?}");
            // A longer chat starts with a shorter one: its first messages' tokens can
            // be counted apart.
            let longer = history(family, Some("Be brief."), "", &self::chat(), "12 Nov 2026");
            assert!(longer.starts_with(&text), "{family:?}");
        }
    }

    fn chat() -> Vec<Message> {
        vec![
            Message {
                role: "user".into(),
                content: "Hi".into(),
            },
            Message {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
            Message {
                role: "user".into(),
                content: " And you? ".into(),
            },
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
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nAnd you?<turn|>\n<|turn>model\n"
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
        let text = history(Family::Muse, Some("Be brief."), "TOOLS", &chat()[..1], "x");
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
        assert_eq!(answer_opening(Family::Gemma4), opening(Family::Gemma4));
        assert_eq!(Family::of(Some("muse-glimmer")), Some(Family::Muse));
        assert_eq!(iso_date("05 Jan 2027"), "2027-01-05");
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
}
