//! Usage since the server started, for the app's Statistics window (`GET /api/stats`, in
//! the shape the 0.9 Mac app's `StatisticsView` reads): who asked, how much the model read
//! and wrote, and how fast. Kept in memory only, and never what was asked.

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::json;

/// The latest requests kept, newest first.
const RECENT: usize = 100;
/// The latest refused attempts kept.
const REFUSED: usize = 20;
/// How far back the activity chart goes.
const HISTORY_SECONDS: f64 = 24.0 * 3600.0;

/// Who made a request: the app, a script on this computer (the app's token, another
/// client), or a paired device.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub id: String,
    pub name: String,
    /// "mac" (the app), "api" (a script) or "device", as the 0.9 view draws them.
    pub kind: &'static str,
}

const HERE: &str = if cfg!(target_os = "macos") {
    "this Mac"
} else {
    "this PC"
};

impl Source {
    /// The source of a request let in with the app's token (`app`) or a paired device's
    /// (`device`, its name). With the app's token, the app is told from a script by its
    /// User-Agent: the Mac app's URLSession sends CFNetwork's, the Windows app its own
    /// "VireoPrototype/…"; syrinx-eval sends none, Python's urllib its own.
    pub fn of(app: bool, device: &str, user_agent: Option<&str>) -> Self {
        if !app {
            return Source {
                id: format!("device:{device}"),
                name: device.to_string(),
                kind: "device",
            };
        }
        let ua = user_agent.unwrap_or("");
        if ua.contains("CFNetwork") || ua.starts_with("Vireo") {
            Source {
                id: "app".into(),
                name: format!("Vireo on {HERE}"),
                kind: "mac",
            }
        } else {
            Source {
                id: "script".into(),
                name: format!("Scripts on {HERE}"),
                kind: "api",
            }
        }
    }

    /// "api" for a script's request, as the 0.9 server marks requests through its API.
    fn request_kind(&self) -> &'static str {
        if self.kind == "api" { "api" } else { "chat" }
    }
}

/// What the engine knows of one reply, sent before it's done.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    /// The prompt's tokens, and those kept from the chat's cache (not read again).
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    /// How long reading the rest took.
    pub prefill_seconds: f64,
    /// The chat's tokens at the end of the reply: prompt, reply and tool results.
    pub context_tokens: usize,
    pub tool_calls: usize,
    /// Guesses made (a draft model, the chat itself or Qwen's prediction layer) and taken.
    pub guessed: usize,
    pub taken: usize,
}

impl Usage {
    fn prefill_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }
}

/// How a request ended.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// "ok", "error" or "cancelled".
    pub status: &'static str,
    pub completion_tokens: usize,
    /// The engine's time writing the reply (`None` when it never finished one).
    pub generation_seconds: Option<f64>,
    /// From the request's start to the first thing streamed back.
    pub first_token_seconds: Option<f64>,
    pub usage: Option<Usage>,
}

struct Active {
    source: Source,
    started_at: f64,
}

struct Finished {
    source: Source,
    started_at: f64,
    duration: f64,
    outcome: Outcome,
}

/// One source's sums.
#[derive(Default)]
struct Sums {
    requests: usize,
    errors: usize,
    completion_tokens: usize,
    prompt_tokens: usize,
    generation_seconds: f64,
    /// Over the requests the engine reported on (prompt tokens of those, for the cache share).
    reported_prompt_tokens: usize,
    cached_tokens: usize,
    prefill_tokens: usize,
    prefill_seconds: f64,
    first_token_seconds: f64,
    first_token_count: usize,
    context_tokens: usize,
    context_count: usize,
    peak_context_tokens: usize,
    tool_calls: usize,
    guessed: usize,
    taken: usize,
    last_seen_at: Option<f64>,
}

impl Sums {
    fn add(&mut self, f: &Finished, now: f64) {
        let o = &f.outcome;
        self.requests += 1;
        if o.status == "error" {
            self.errors += 1;
        }
        self.completion_tokens += o.completion_tokens;
        if let Some(s) = o
            .generation_seconds
            .filter(|s| *s > 0.0 && o.completion_tokens > 0)
        {
            self.generation_seconds += s;
        }
        if let Some(t) = o.first_token_seconds {
            self.first_token_seconds += t;
            self.first_token_count += 1;
        }
        if let Some(u) = &o.usage {
            self.prompt_tokens += u.prompt_tokens;
            self.reported_prompt_tokens += u.prompt_tokens;
            self.cached_tokens += u.cached_tokens;
            if u.prefill_seconds > 0.0 {
                self.prefill_tokens += u.prefill_tokens();
                self.prefill_seconds += u.prefill_seconds;
            }
            if u.context_tokens > 0 {
                self.context_tokens += u.context_tokens;
                self.context_count += 1;
                self.peak_context_tokens = self.peak_context_tokens.max(u.context_tokens);
            }
            self.tool_calls += u.tool_calls;
            self.guessed += u.guessed;
            self.taken += u.taken;
        }
        self.last_seen_at = Some(now);
    }

    fn merged<'a>(all: impl Iterator<Item = &'a Sums>) -> Sums {
        let mut t = Sums::default();
        for s in all {
            t.requests += s.requests;
            t.errors += s.errors;
            t.completion_tokens += s.completion_tokens;
            t.prompt_tokens += s.prompt_tokens;
            t.generation_seconds += s.generation_seconds;
            t.reported_prompt_tokens += s.reported_prompt_tokens;
            t.cached_tokens += s.cached_tokens;
            t.prefill_tokens += s.prefill_tokens;
            t.prefill_seconds += s.prefill_seconds;
            t.first_token_seconds += s.first_token_seconds;
            t.first_token_count += s.first_token_count;
            t.context_tokens += s.context_tokens;
            t.context_count += s.context_count;
            t.peak_context_tokens = t.peak_context_tokens.max(s.peak_context_tokens);
            t.tool_calls += s.tool_calls;
            t.guessed += s.guessed;
            t.taken += s.taken;
        }
        t
    }

    fn speed(&self) -> Option<f64> {
        rate(self.completion_tokens, self.generation_seconds)
    }

    /// The prompt-side figures, as JSON fields (with a leading comma).
    fn token_fields(&self) -> String {
        let reported = self.reported_prompt_tokens > 0;
        format!(
            ",\"cachedTokens\":{},\"cacheHitRate\":{},\"prefillTokens\":{},\"prefillTokensPerSecond\":{},\"averageFirstTokenSeconds\":{},\"averageContextTokens\":{},\"toolCalls\":{},\"draftGuesses\":{},\"draftTaken\":{},\"draftAcceptance\":{}",
            int_or_null(reported.then_some(self.cached_tokens)),
            num(
                reported.then(|| self.cached_tokens as f64 / self.reported_prompt_tokens as f64),
                4
            ),
            int_or_null((self.prefill_seconds > 0.0).then_some(self.prefill_tokens)),
            num(rate(self.prefill_tokens, self.prefill_seconds), 2),
            num(
                (self.first_token_count > 0)
                    .then(|| self.first_token_seconds / self.first_token_count as f64),
                2
            ),
            int_or_null((self.context_count > 0).then(|| {
                (self.context_tokens as f64 / self.context_count as f64).round() as usize
            })),
            self.tool_calls,
            self.guessed,
            self.taken,
            num(
                (self.guessed > 0).then(|| self.taken as f64 / self.guessed as f64),
                4
            ),
        )
    }
}

/// What the server knows of its use since it started.
pub struct Stats {
    started_at: f64,
    next_id: u64,
    active: HashMap<u64, Active>,
    /// By source id, in the order first seen, with each source's sums.
    sources: Vec<(Source, Sums)>,
    recent: VecDeque<Finished>,
    /// When each request finished, and the tokens it wrote: the activity chart.
    history: VecDeque<(f64, usize)>,
    refused_count: usize,
    /// When, from where and why, newest first.
    refused: VecDeque<(f64, String, String)>,
}

impl Stats {
    pub fn new(now: f64) -> Self {
        Stats {
            started_at: now,
            next_id: 0,
            active: HashMap::new(),
            sources: Vec::new(),
            recent: VecDeque::new(),
            history: VecDeque::new(),
            refused_count: 0,
            refused: VecDeque::new(),
        }
    }

    fn sums(&mut self, source: &Source) -> &mut Sums {
        let i = match self.sources.iter().position(|(s, _)| s.id == source.id) {
            Some(i) => i,
            None => {
                self.sources.push((source.clone(), Sums::default()));
                self.sources.len() - 1
            }
        };
        &mut self.sources[i].1
    }

    pub fn begin(&mut self, source: Source, now: f64) -> u64 {
        self.next_id += 1;
        self.sums(&source).last_seen_at = Some(now);
        self.active.insert(
            self.next_id,
            Active {
                source,
                started_at: now,
            },
        );
        self.next_id
    }

    pub fn finish(&mut self, id: u64, outcome: Outcome, now: f64) {
        let Some(active) = self.active.remove(&id) else {
            return;
        };
        let finished = Finished {
            source: active.source,
            started_at: active.started_at,
            duration: (now - active.started_at).max(0.0),
            outcome,
        };
        self.history
            .push_back((now, finished.outcome.completion_tokens));
        while self
            .history
            .front()
            .is_some_and(|(t, _)| *t < now - HISTORY_SECONDS)
        {
            self.history.pop_front();
        }
        let source = finished.source.clone();
        self.sums(&source).add(&finished, now);
        self.recent.push_front(finished);
        self.recent.truncate(RECENT);
    }

    pub fn refuse(&mut self, address: &str, reason: &str, now: f64) {
        self.refused_count += 1;
        self.refused
            .push_front((now, address.to_string(), reason.to_string()));
        self.refused.truncate(REFUSED);
    }

    /// `count` buckets of `seconds` each, the last holding `now`.
    fn buckets(&self, count: usize, seconds: f64, now: f64) -> String {
        let end = ((now / seconds).floor() + 1.0) * seconds;
        let start = end - count as f64 * seconds;
        let mut buckets = vec![(0usize, 0usize); count];
        for &(at, tokens) in &self.history {
            let i = ((at - start) / seconds).floor();
            if i >= 0.0 && (i as usize) < count {
                buckets[i as usize].0 += 1;
                buckets[i as usize].1 += tokens;
            }
        }
        let items: Vec<String> = buckets
            .iter()
            .enumerate()
            .map(|(i, (requests, tokens))| {
                format!(
                    "{{\"start\":{},\"requests\":{requests},\"tokens\":{tokens}}}",
                    (start + i as f64 * seconds) as i64
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// `GET /api/stats`.
    pub fn json(&self, model: &str, runtime: &str, context: usize, now: f64) -> String {
        let totals = Sums::merged(self.sources.iter().map(|(_, s)| s));
        let durations: Vec<f64> = self
            .recent
            .iter()
            .filter(|f| f.outcome.status == "ok")
            .map(|f| f.duration)
            .collect();
        let average_duration =
            (!durations.is_empty()).then(|| durations.iter().sum::<f64>() / durations.len() as f64);

        let mut active: Vec<&Active> = self.active.values().collect();
        active.sort_by(|a, b| a.started_at.total_cmp(&b.started_at));
        let active: Vec<String> = active
            .iter()
            .map(|a| {
                format!(
                    "{{\"sourceName\":{},\"sourceKind\":{},\"kind\":\"{}\",\"startedAt\":{},\"seconds\":{}}}",
                    json::string(&a.source.name),
                    json::string(a.source.kind),
                    a.source.request_kind(),
                    num(Some(a.started_at), 3),
                    num(Some(now - a.started_at), 1),
                )
            })
            .collect();

        let mut sources: Vec<&(Source, Sums)> = self.sources.iter().collect();
        sources.sort_by(|a, b| {
            b.1.last_seen_at
                .unwrap_or(0.0)
                .total_cmp(&a.1.last_seen_at.unwrap_or(0.0))
        });
        let sources: Vec<String> = sources
            .iter()
            .map(|(source, s)| {
                let busy = self
                    .active
                    .values()
                    .filter(|a| a.source.id == source.id)
                    .count();
                format!(
                    "{{\"id\":{},\"name\":{},\"kind\":{},\"requests\":{},\"errors\":{},\"completionTokens\":{},\"promptTokens\":{},\"tokensPerSecond\":{},\"lastSeenAt\":{},\"activeRequests\":{busy}{}}}",
                    json::string(&source.id),
                    json::string(&source.name),
                    json::string(source.kind),
                    s.requests,
                    s.errors,
                    s.completion_tokens,
                    s.prompt_tokens,
                    num(s.speed(), 2),
                    num(s.last_seen_at, 3),
                    s.token_fields(),
                )
            })
            .collect();

        let recent: Vec<String> = self
            .recent
            .iter()
            .map(|f| {
                let o = &f.outcome;
                let u = o.usage.as_ref();
                format!(
                    "{{\"source\":{},\"sourceName\":{},\"sourceKind\":{},\"kind\":\"{}\",\"status\":\"{}\",\"startedAt\":{},\"duration\":{},\"completionTokens\":{},\"promptTokens\":{},\"tokensPerSecond\":{},\"cachedTokens\":{},\"prefillTokens\":{},\"prefillTokensPerSecond\":{},\"firstTokenSeconds\":{},\"contextTokens\":{},\"toolCalls\":{},\"draftGuesses\":{},\"draftTaken\":{}}}",
                    json::string(&f.source.id),
                    json::string(&f.source.name),
                    json::string(f.source.kind),
                    f.source.request_kind(),
                    o.status,
                    num(Some(f.started_at), 3),
                    num(Some(f.duration), 2),
                    o.completion_tokens,
                    int_or_null(u.map(|u| u.prompt_tokens)),
                    num(o.generation_seconds.and_then(|s| rate(o.completion_tokens, s)), 2),
                    int_or_null(u.map(|u| u.cached_tokens)),
                    int_or_null(u.filter(|u| u.prefill_seconds > 0.0).map(Usage::prefill_tokens)),
                    num(u.and_then(|u| rate(u.prefill_tokens(), u.prefill_seconds)), 2),
                    num(o.first_token_seconds, 2),
                    int_or_null(u.map(|u| u.context_tokens).filter(|&c| c > 0)),
                    int_or_null(u.map(|u| u.tool_calls)),
                    int_or_null(u.map(|u| u.guessed)),
                    int_or_null(u.map(|u| u.taken)),
                )
            })
            .collect();

        let refused: Vec<String> = self
            .refused
            .iter()
            .map(|(at, address, reason)| {
                format!(
                    "{{\"at\":{},\"address\":{},\"reason\":{}}}",
                    num(Some(*at), 3),
                    json::string(address),
                    json::string(reason)
                )
            })
            .collect();

        format!(
            "{{\"startedAt\":{},\"uptimeSeconds\":{},\"model\":{},\"runtime\":{},\"contextLimit\":{context},\"totals\":{{\"requests\":{},\"errors\":{},\"completionTokens\":{},\"promptTokens\":{},\"tokensPerSecond\":{},\"averageDuration\":{}{},\"peakContextTokens\":{}}},\"active\":[{}],\"sources\":[{}],\"lastHour\":{},\"lastDay\":{},\"recent\":[{}],\"refused\":{{\"count\":{},\"recent\":[{}]}}}}",
            num(Some(self.started_at), 3),
            (now - self.started_at).max(0.0).round() as u64,
            json::string(model),
            json::string(runtime),
            totals.requests,
            totals.errors,
            totals.completion_tokens,
            totals.prompt_tokens,
            num(totals.speed(), 2),
            num(average_duration, 2),
            totals.token_fields(),
            int_or_null((totals.peak_context_tokens > 0).then_some(totals.peak_context_tokens)),
            active.join(","),
            sources.join(","),
            self.buckets(60, 60.0, now),
            self.buckets(24, 3600.0, now),
            recent.join(","),
            self.refused_count,
            refused.join(","),
        )
    }
}

fn rate(tokens: usize, seconds: f64) -> Option<f64> {
    (tokens > 0 && seconds > 0.0).then(|| tokens as f64 / seconds)
}

/// A number rounded to `places`, or null.
fn num(value: Option<f64>, places: usize) -> String {
    match value {
        Some(v) if v.is_finite() => {
            let text = format!("{v:.places$}");
            // "12.50" → "12.5", "3.00" → "3": shorter, and the same number.
            if text.contains('.') {
                text.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                text
            }
        }
        _ => "null".into(),
    }
}

fn int_or_null(value: Option<usize>) -> String {
    value.map_or("null".into(), |v| v.to_string())
}

// ---- The server's own -------------------------------------------------------------------

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

static STATS: LazyLock<Mutex<Stats>> = LazyLock::new(|| Mutex::new(Stats::new(now())));

/// Starts the clock: the server's start, not its first request.
pub fn start() {
    LazyLock::force(&STATS);
}

pub fn begin(source: Source) -> u64 {
    STATS.lock().unwrap().begin(source, now())
}

pub fn finish(id: u64, outcome: Outcome) {
    STATS.lock().unwrap().finish(id, outcome, now())
}

pub fn refuse(address: &str, reason: &str) {
    STATS.lock().unwrap().refuse(address, reason, now())
}

pub fn json(model: &str, runtime: &str, context: usize) -> String {
    STATS.lock().unwrap().json(model, runtime, context, now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Value;

    fn app() -> Source {
        Source::of(
            true,
            "This PC",
            Some("VireoMac/1 CFNetwork/3826 Darwin/25.0.0"),
        )
    }

    #[test]
    fn tells_the_app_from_scripts_and_devices() {
        assert_eq!(app().kind, "mac");
        assert_eq!(app().id, "app");
        assert_eq!(
            Source::of(true, "This PC", Some("VireoPrototype/0.9.0")).kind,
            "mac"
        );
        assert_eq!(Source::of(true, "This PC", None).kind, "api");
        assert_eq!(
            Source::of(true, "This PC", Some("Python-urllib/3.12")).id,
            "script"
        );
        let device = Source::of(false, "Sam's PC", Some("VireoPrototype/0.9.0"));
        assert_eq!((device.kind, device.name.as_str()), ("device", "Sam's PC"));
        assert_eq!(device.request_kind(), "chat");
        assert_eq!(Source::of(true, "", None).request_kind(), "api");
    }

    #[test]
    fn rounds_numbers_for_json() {
        assert_eq!(num(Some(12.5), 2), "12.5");
        assert_eq!(num(Some(3.0), 2), "3");
        assert_eq!(num(Some(0.4762), 4), "0.4762");
        assert_eq!(num(Some(f64::NAN), 2), "null");
        assert_eq!(num(None, 2), "null");
    }

    #[test]
    fn an_empty_server_has_the_whole_shape() {
        let stats = Stats::new(1000.0);
        let v = json::parse(&stats.json("Gemma", "Vireo engine", 8192, 1060.0)).unwrap();
        assert_eq!(v.num("uptimeSeconds"), Some(60.0));
        assert_eq!(v.num("contextLimit"), Some(8192.0));
        let totals = v.get("totals").unwrap();
        assert_eq!(totals.num("requests"), Some(0.0));
        assert!(matches!(totals.get("tokensPerSecond"), Some(Value::Null)));
        assert!(matches!(totals.get("cacheHitRate"), Some(Value::Null)));
        assert_eq!(v.array("lastHour").len(), 60);
        assert_eq!(v.array("lastDay").len(), 24);
        assert!(v.array("sources").is_empty());
        assert_eq!(v.get("refused").unwrap().num("count"), Some(0.0));
    }

    #[test]
    fn counts_replies_by_source() {
        let mut stats = Stats::new(1000.0);
        let first = stats.begin(app(), 1000.0);
        let second = stats.begin(Source::of(true, "This PC", None), 1001.0);
        // Both under way.
        let v = json::parse(&stats.json("m", "r", 10_000, 1002.0)).unwrap();
        assert_eq!(v.array("active").len(), 2);
        assert_eq!(v.array("active")[0].str("sourceKind"), Some("mac"));
        assert_eq!(v.array("active")[1].str("kind"), Some("api"));
        assert_eq!(v.array("active")[0].num("seconds"), Some(2.0));

        stats.finish(
            first,
            Outcome {
                status: "ok",
                completion_tokens: 100,
                generation_seconds: Some(2.0),
                first_token_seconds: Some(0.5),
                usage: Some(Usage {
                    prompt_tokens: 3000,
                    cached_tokens: 2000,
                    prefill_seconds: 0.5,
                    context_tokens: 3100,
                    tool_calls: 2,
                    guessed: 40,
                    taken: 30,
                }),
            },
            1003.0,
        );
        stats.finish(
            second,
            Outcome {
                status: "error",
                ..Outcome::default()
            },
            1004.0,
        );
        stats.refuse("192.168.1.50", "Not paired", 1004.5);

        let v = json::parse(&stats.json("m", "r", 10_000, 1005.0)).unwrap();
        assert!(v.array("active").is_empty());
        let t = v.get("totals").unwrap();
        assert_eq!(t.num("requests"), Some(2.0));
        assert_eq!(t.num("errors"), Some(1.0));
        assert_eq!(t.num("completionTokens"), Some(100.0));
        assert_eq!(t.num("promptTokens"), Some(3000.0));
        assert_eq!(t.num("tokensPerSecond"), Some(50.0));
        assert_eq!(t.num("averageDuration"), Some(3.0));
        assert_eq!(t.num("cachedTokens"), Some(2000.0));
        assert_eq!(t.num("cacheHitRate"), Some(0.6667));
        assert_eq!(t.num("prefillTokens"), Some(1000.0));
        assert_eq!(t.num("prefillTokensPerSecond"), Some(2000.0));
        assert_eq!(t.num("averageFirstTokenSeconds"), Some(0.5));
        assert_eq!(t.num("averageContextTokens"), Some(3100.0));
        assert_eq!(t.num("peakContextTokens"), Some(3100.0));
        assert_eq!(t.num("toolCalls"), Some(2.0));
        assert_eq!(t.num("draftAcceptance"), Some(0.75));

        // The script's failed request was seen last, so it comes first.
        let sources = v.array("sources");
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].str("id"), Some("script"));
        assert_eq!(sources[0].num("errors"), Some(1.0));
        assert!(matches!(sources[0].get("cacheHitRate"), Some(Value::Null)));
        assert_eq!(sources[1].str("kind"), Some("mac"));
        assert_eq!(sources[1].num("tokensPerSecond"), Some(50.0));

        let recent = v.array("recent");
        assert_eq!(recent[0].str("status"), Some("error"));
        assert_eq!(recent[0].str("kind"), Some("api"));
        assert!(matches!(recent[0].get("promptTokens"), Some(Value::Null)));
        assert_eq!(recent[1].num("duration"), Some(3.0));
        assert_eq!(recent[1].num("prefillTokensPerSecond"), Some(2000.0));
        assert_eq!(recent[1].num("firstTokenSeconds"), Some(0.5));

        // Both finished in the last minute's bucket.
        let hour = v.array("lastHour");
        assert_eq!(hour[59].num("requests"), Some(2.0));
        assert_eq!(hour[59].num("tokens"), Some(100.0));
        assert_eq!(hour[59].num("start"), Some(960.0));

        let refused = v.get("refused").unwrap();
        assert_eq!(refused.num("count"), Some(1.0));
        assert_eq!(refused.array("recent")[0].str("reason"), Some("Not paired"));
    }

    #[test]
    fn keeps_a_day_of_history_and_the_latest_requests() {
        let mut stats = Stats::new(0.0);
        for i in 0..(RECENT + 5) {
            let id = stats.begin(app(), i as f64);
            stats.finish(
                id,
                Outcome {
                    status: "ok",
                    ..Outcome::default()
                },
                i as f64,
            );
        }
        assert_eq!(stats.recent.len(), RECENT);
        let id = stats.begin(app(), HISTORY_SECONDS + 200.0);
        stats.finish(
            id,
            Outcome {
                status: "ok",
                ..Outcome::default()
            },
            HISTORY_SECONDS + 200.0,
        );
        assert_eq!(stats.history.len(), 1);
    }
}
