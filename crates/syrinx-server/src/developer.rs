//! The OpenAI-compatible API's settings and keys (`openai.rs` serves the API). As in
//! Vireo 0.9: it is off until it's turned on; requests from this computer need no key
//! unless the user asks for one (this server listens only on this computer); keys look
//! like `ick-…`, and only their SHA-256 is kept, in `developer.json` in the prototype's
//! folder (`store::data_dir`), the same shape as 0.9's.
//!
//! `--openai on` (or `off`) on the command line overrides whether it's on, for someone
//! running the server by hand.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::json::{self, Value};

/// Keys start with this, as 0.9's did.
pub const KEY_PREFIX: &str = "ick-";
pub const MAX_KEYS: usize = 50;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Key {
    pub id: String,
    pub name: String,
    /// The key's SHA-256, in hex.
    pub key_hash: String,
    /// The key's start and end, to tell keys apart in a list (`ick-AbCd…wxyz`).
    pub hint: String,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    pub enabled: bool,
    pub require_key_locally: bool,
    pub keys: Vec<Key>,
}

/// The command line's say over `enabled`: 0 none, 1 on, 2 off.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);
/// Settings are read and written under this.
static LOCK: Mutex<()> = Mutex::new(());

pub fn set_override(on: bool) {
    OVERRIDE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Serving on its own (`--standalone`): the key given on the command line (`--api-key`),
/// kept as its SHA-256 as the saved keys are. Once set, every request needs it, or a saved
/// key, or the app's token: the server may be listening on a network (`--host`).
static COMMAND_LINE_KEY: OnceLock<String> = OnceLock::new();
/// Serving on its own: the token of the app that started the server (`--token`, the Syrinx
/// manager), as its SHA-256, which may use the API and read the statistics.
static APP_TOKEN: OnceLock<String> = OnceLock::new();

pub fn set_command_line_key(key: &str) {
    let _ = COMMAND_LINE_KEY.set(sha256_hex(key.as_bytes()));
}

pub fn set_app_token(token: &str) {
    let _ = APP_TOKEN.set(sha256_hex(token.as_bytes()));
}

/// Whether a key from the command line is needed (`--api-key`).
pub fn command_line_key() -> bool {
    COMMAND_LINE_KEY.get().is_some()
}

fn file() -> PathBuf {
    std::env::var_os("VIREO_DEVELOPER_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::store::data_dir().join("developer.json"))
}

impl Settings {
    fn from_json(v: &Value) -> Settings {
        let keys = v
            .array("keys")
            .iter()
            .filter_map(|k| {
                Some(Key {
                    id: k.str("id")?.to_string(),
                    name: k.str("name").unwrap_or("").to_string(),
                    key_hash: k.str("keyHash")?.to_string(),
                    hint: k.str("hint").unwrap_or("").to_string(),
                    created_at: k.str("createdAt").unwrap_or("").to_string(),
                    last_used_at: k.str("lastUsedAt").map(str::to_string),
                })
            })
            .collect();
        Settings {
            enabled: matches!(v.get("enabled"), Some(Value::Bool(true))),
            require_key_locally: matches!(v.get("requireKeyLocally"), Some(Value::Bool(true))),
            keys,
        }
    }

    fn to_json(&self) -> String {
        let keys: Vec<String> = self
            .keys
            .iter()
            .map(|k| {
                format!(
                    "{{\"id\":{},\"name\":{},\"keyHash\":{},\"hint\":{},\"createdAt\":{},\"lastUsedAt\":{}}}",
                    json::string(&k.id),
                    json::string(&k.name),
                    json::string(&k.key_hash),
                    json::string(&k.hint),
                    json::string(&k.created_at),
                    k.last_used_at.as_deref().map_or("null".to_string(), json::string)
                )
            })
            .collect();
        format!(
            "{{\"enabled\":{},\"requireKeyLocally\":{},\"keys\":[{}]}}",
            self.enabled,
            self.require_key_locally,
            keys.join(",")
        )
    }

    /// What the app is shown: the settings without the keys' hashes, and the model.
    pub fn public_json(&self, model: &str) -> String {
        let keys: Vec<String> = self.keys.iter().map(public_key).collect();
        format!(
            "{{\"enabled\":{},\"requireKeyLocally\":{},\"keys\":[{}],\"model\":{}}}",
            self.enabled,
            self.require_key_locally,
            keys.join(","),
            json::string(model)
        )
    }
}

pub fn public_key(k: &Key) -> String {
    format!(
        "{{\"id\":{},\"name\":{},\"hint\":{},\"createdAt\":{},\"lastUsedAt\":{}}}",
        json::string(&k.id),
        json::string(&k.name),
        json::string(&k.hint),
        json::string(&k.created_at),
        k.last_used_at.as_deref().map_or("null".to_string(), json::string)
    )
}

/// The settings as saved, with the command line's say over `enabled`.
pub fn load() -> Settings {
    let _held = LOCK.lock().unwrap();
    load_unlocked()
}

fn load_unlocked() -> Settings {
    let mut settings = std::fs::read_to_string(file())
        .ok()
        .and_then(|t| json::parse(&t).ok())
        .map(|v| Settings::from_json(&v))
        .unwrap_or_default();
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => settings.enabled = true,
        2 => settings.enabled = false,
        _ => {}
    }
    settings
}

/// Changes the settings under the lock and saves them; what `change` returns.
fn update<T>(change: impl FnOnce(&mut Settings) -> Result<T, String>) -> Result<T, String> {
    let _held = LOCK.lock().unwrap();
    let mut settings = load_unlocked();
    let out = change(&mut settings)?;
    let path = file();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let temp = path.with_extension("json.writing");
    std::fs::write(&temp, settings.to_json()).map_err(|e| format!("Couldn't save the settings: {e}"))?;
    std::fs::rename(&temp, &path).map_err(|e| format!("Couldn't save the settings: {e}"))?;
    Ok(out)
}

/// The app's `PUT /api/developer`: `enabled` and `requireKeyLocally`, each if given.
pub fn set(body: &Value) -> Result<Settings, String> {
    update(|s| {
        for (field, slot) in [
            ("enabled", &mut s.enabled),
            ("requireKeyLocally", &mut s.require_key_locally),
        ] {
            match body.get(field) {
                None => {}
                Some(Value::Bool(b)) => *slot = *b,
                Some(_) => return Err(format!("{field} must be true or false.")),
            }
        }
        Ok(s.clone())
    })
}

/// A new key named `name`: the key itself (the only time it's shown) and what's kept.
pub fn create_key(name: &str) -> Result<(String, Key), String> {
    let name: String = name.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect();
    let name = if name.is_empty() { "Untitled key".to_string() } else { name };
    let token = format!("{KEY_PREFIX}{}", random_token());
    let key = Key {
        id: format!("{:016x}", crate::random_u64()),
        name,
        key_hash: sha256_hex(token.as_bytes()),
        hint: format!("{}…{}", &token[..8], &token[token.len() - 4..]),
        created_at: now_iso(),
        last_used_at: None,
    };
    update(|s| {
        if s.keys.len() >= MAX_KEYS {
            return Err(format!("There can be up to {MAX_KEYS} keys. Revoke one first."));
        }
        s.keys.push(key.clone());
        Ok(())
    })?;
    Ok((token, key))
}

/// Revokes a key; whether there was one with that id.
pub fn revoke_key(id: &str) -> Result<bool, String> {
    update(|s| {
        let before = s.keys.len();
        s.keys.retain(|k| k.id != id);
        Ok(s.keys.len() != before)
    })
}

/// Who may use the API: `Ok` with the key used (if one was), or an HTTP status and why.
/// A wrong key is refused only when a key is needed: OpenAI's client libraries insist on
/// sending one, so callers on this computer often send a placeholder.
pub fn authorize(settings: &Settings, bearer: Option<&str>) -> Result<Option<Key>, (u16, &'static str, String)> {
    authorize_with(
        settings,
        bearer,
        COMMAND_LINE_KEY.get().map(String::as_str),
        APP_TOKEN.get().map(String::as_str),
    )
}

/// [`authorize`], with the command line's key and the app's token (their hashes) given.
fn authorize_with(
    settings: &Settings,
    bearer: Option<&str>,
    command_line: Option<&str>,
    app: Option<&str>,
) -> Result<Option<Key>, (u16, &'static str, String)> {
    if !settings.enabled {
        return Err((
            403,
            "api_disabled",
            "The OpenAI-compatible API is turned off. Turn it on in Vireo's settings, or start syrinx-server with --openai on.".into(),
        ));
    }
    let hash = bearer.map(|token| sha256_hex(token.as_bytes()));
    // The command line's key and the app's token (`--standalone`).
    if let Some(hash) = &hash
        && [command_line, app].iter().flatten().any(|kept| same(kept, hash))
    {
        return Ok(None);
    }
    let key = hash.and_then(|hash| settings.keys.iter().find(|k| same(&k.key_hash, &hash)).cloned());
    if key.is_none() && (settings.require_key_locally || command_line.is_some()) {
        return Err((
            401,
            "invalid_api_key",
            if bearer.is_some() {
                "Incorrect API key provided.".into()
            } else if command_line.is_some() {
                "An API key is required: give this server's key as a bearer token (Authorization: Bearer <key>).".into()
            } else {
                "An API key is required. Create one in Vireo's settings.".into()
            },
        ));
    }
    Ok(key)
}

/// Notes a key's use, at most once a minute.
pub fn touch(id: &str) {
    let now = now_iso();
    let _ = update(|s| {
        if let Some(k) = s.keys.iter_mut().find(|k| k.id == id)
            && k.last_used_at.as_deref().is_none_or(|t| t.get(..16) != now.get(..16))
        {
            k.last_used_at = Some(now.clone());
        }
        Ok(())
    });
}

/// Compares two hashes in time that doesn't depend on where they differ.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// 43 URL-safe characters from the system's random source (256 bits).
fn random_token() -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut bytes = [0u8; 32];
    fill_random(&mut bytes);
    let mut out = String::new();
    let mut bits = 0u32;
    let mut held = 0;
    for b in bytes {
        bits = (bits << 8) | u32::from(b);
        held += 8;
        while held >= 6 {
            held -= 6;
            out.push(ALPHABET[((bits >> held) & 63) as usize] as char);
        }
    }
    if held > 0 {
        out.push(ALPHABET[((bits << (6 - held)) & 63) as usize] as char);
    }
    out
}

/// The system's random bytes: /dev/urandom, or on Windows the system's generator.
#[cfg(unix)]
fn fill_random(out: &mut [u8]) {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom").expect("/dev/urandom");
    f.read_exact(out).expect("random bytes");
}

#[cfg(windows)]
fn fill_random(out: &mut [u8]) {
    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(alg: *mut std::ffi::c_void, buf: *mut u8, len: u32, flags: u32) -> i32;
    }
    // BCRYPT_USE_SYSTEM_PREFERRED_RNG
    let status = unsafe { BCryptGenRandom(std::ptr::null_mut(), out.as_mut_ptr(), out.len() as u32, 2) };
    assert_eq!(status, 0, "BCryptGenRandom");
}

/// Now, in UTC, as `2026-11-19T21:34:02Z`.
pub fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    iso(secs)
}

/// Seconds since 1970 in UTC as `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant's days-to-date).
fn iso(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rest = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// SHA-256 (FIPS 180-4), in hex.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_line_key_is_needed_once_given_and_the_app_token_opens_too() {
        let on = Settings { enabled: true, ..Settings::default() };
        let cli = sha256_hex(b"sx-secret");
        let app = sha256_hex(b"app-token");
        let check = |bearer| authorize_with(&on, bearer, Some(&cli), Some(&app)).map(|k| k.is_none());
        assert_eq!(check(Some("sx-secret")), Ok(true));
        assert_eq!(check(Some("app-token")), Ok(true));
        assert_eq!(check(Some("wrong")).unwrap_err().0, 401);
        assert_eq!(check(None).unwrap_err().0, 401);
        // Without one, this computer needs no key (as before).
        assert!(authorize_with(&on, None, None, None).is_ok());
        // Turned off is off, whatever the key.
        let off = Settings::default();
        assert_eq!(authorize_with(&off, Some("sx-secret"), Some(&cli), None).unwrap_err().0, 403);
    }

    #[test]
    fn sha256_matches_the_standards_examples() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn dates_read_as_utc() {
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso(1_795_122_842), "2026-11-19T21:14:02Z");
    }

    #[test]
    fn keys_are_made_kept_hashed_checked_and_revoked() {
        let path = std::env::temp_dir().join(format!("syrinx-developer-{:x}.json", crate::random_u64()));
        // SAFETY: the tests in this module are the only ones that read this variable.
        unsafe { std::env::set_var("VIREO_DEVELOPER_FILE", &path) };
        let off = load();
        assert!(!off.enabled);
        assert_eq!(authorize(&off, None).unwrap_err().0, 403);
        let on = set(&json::parse("{\"enabled\":true}").unwrap()).unwrap();
        assert!(on.enabled && !on.require_key_locally);
        // No key needed: none, a placeholder or a real one all go.
        assert_eq!(authorize(&on, None), Ok(None));
        assert_eq!(authorize(&on, Some("lm-studio")), Ok(None));
        let (token, key) = create_key("  My   script ").unwrap();
        assert!(token.starts_with("ick-") && token.len() == 4 + 43, "{token}");
        assert_eq!(key.name, "My script");
        assert!(key.hint.starts_with("ick-") && key.hint.contains('…'));
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(!saved.contains(&token), "the key itself is never saved");
        assert!(saved.contains(&sha256_hex(token.as_bytes())));
        let needed = set(&json::parse("{\"requireKeyLocally\":true}").unwrap()).unwrap();
        assert_eq!(authorize(&needed, Some(&token)).unwrap().unwrap().id, key.id);
        assert_eq!(authorize(&needed, Some("wrong")).unwrap_err().0, 401);
        assert_eq!(authorize(&needed, None).unwrap_err().0, 401);
        assert!(set(&json::parse("{\"enabled\":1}").unwrap()).is_err());
        touch(&key.id);
        assert!(load().keys[0].last_used_at.is_some());
        assert!(!needed.public_json("m").contains("keyHash"));
        assert!(revoke_key(&key.id).unwrap());
        assert!(!revoke_key(&key.id).unwrap());
        assert_eq!(authorize(&load(), Some(&token)).unwrap_err().0, 401);
        let _ = std::fs::remove_file(&path);
    }
}
