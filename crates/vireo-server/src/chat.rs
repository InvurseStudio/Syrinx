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
}

impl Family {
    pub fn of(architecture: Option<&str>) -> Option<Self> {
        match architecture? {
            "llama" => Some(Family::Llama),
            "qwen35" => Some(Family::Qwen35),
            "gemma4" => Some(Family::Gemma4),
            _ => None,
        }
    }

    /// Recommended sampling (model_profiles.py: Qwen 0.6 / 0.95 / top-k 20, Gemma 4 1.0
    /// / 0.95 / 64, others 0.7 / 0.9).
    pub fn sampling(self) -> Settings {
        match self {
            Family::Qwen35 => Settings {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                ..Settings::default()
            },
            Family::Gemma4 => Settings {
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
    }
}

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
    }
    out
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
