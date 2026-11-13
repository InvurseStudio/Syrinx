//! Chats' caches saved for later: each one the tokens of a chat and its session's state
//! after them (`vireo_engine::models::SavedState`), so going back to a chat, or opening
//! it after a restart, reads only what's new in it.
//!
//! Saved in `chat-cache\<model>\` in the prototype's folder (`data_dir`), a file per chat, written
//! by a thread of their own so switching chats doesn't wait for the disk. The newest
//! states stay in memory as well. The oldest files go once they take more than the
//! budget. A file is:
//!
//!   "VIREOKV1", the number of tokens n (u64), the tokens (n × u32), the state's length
//!   in bytes (u64), the state's bytes; little-endian.
//!
//! The folder is per model file and precision (`model_key`): a state only fits a session
//! of the model and precision that made it, and loading checks its size.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;

use vireo_engine::models::SavedState;

const MAGIC: &[u8; 8] = b"VIREOKV1";

/// What the saved chats may take on disk, per model, at least.
pub const DISK_BUDGET: u64 = 4 << 30;
/// A large model's budget is room for this many chats of the whole context...
const FULL_CHATS: u64 = 4;
/// ...up to this much.
const MOST_DISK: u64 = 32 << 30;

/// The disk budget for a model whose chat of the whole context takes `full_chat` bytes in
/// the GPU's cache (about what it takes saved): 4 GB for the small models, and for a
/// large one room for four full chats. On Gemma 4 31B a 26K-token chat saves as 2.9 GB,
/// and two 11.5K-token chats took 3.6 of the 4 GB (`evals/long-run`): going back to the
/// second of two long chats would have read it all again.
pub fn budget_for(full_chat: u64) -> u64 {
    DISK_BUDGET.max(FULL_CHATS.saturating_mul(full_chat)).min(MOST_DISK)
}
/// How many saved states stay in memory too (the newest; on top of the one in use).
const IN_MEMORY: usize = 2;

/// The folder name for a model file in a precision: its name, size and modification time
/// (a new download of the same name is another model), hashed.
pub fn model_key(path: &Path, precision: &str) -> String {
    let meta = std::fs::metadata(path).ok();
    let size = meta.as_ref().map_or(0, std::fs::Metadata::len);
    let modified = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // FNV-1a: stable from one build to the next, unlike std's hasher.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in format!("{name}\0{size}\0{modified}\0{precision}").bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The prototype's own folder: `~/Library/Application Support/VireoPrototype` on the Mac
/// (where the Mac prototype app keeps its chats), `%LOCALAPPDATA%\VireoPrototype`
/// elsewhere.
pub fn data_dir() -> PathBuf {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map_or_else(
            || PathBuf::from("."),
            |home| {
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
            },
        )
    } else {
        std::env::var_os("LOCALAPPDATA").map_or_else(|| PathBuf::from("."), PathBuf::from)
    };
    base.join("VireoPrototype")
}

/// The default folder for `key`'s saved chats.
pub fn default_dir(key: &str) -> PathBuf {
    data_dir().join("chat-cache").join(key)
}

pub struct Entry {
    pub tokens: Vec<u32>,
    file: PathBuf,
    bytes: u64,
    /// The state, while it's in memory. The writer thread holds the other reference
    /// until the file is written.
    data: Option<Arc<SavedState>>,
    used: u64,
}

impl Entry {
    /// Whether its file is still being written (the writer thread holds its state).
    fn writing(&self) -> bool {
        self.data.as_ref().is_some_and(|d| Arc::strong_count(d) > 1)
    }
}

/// A file to write: where, the tokens, the state.
type ToWrite = (PathBuf, Vec<u32>, Arc<SavedState>);

pub struct Store {
    dir: PathBuf,
    entries: Vec<Entry>,
    budget: u64,
    clock: u64,
    next_name: u64,
    writer: Option<(Sender<ToWrite>, JoinHandle<()>)>,
}

impl Store {
    /// The chats saved in `dir` (made if it isn't there), oldest first.
    pub fn open(dir: PathBuf, budget: u64) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        let mut found = Vec::new();
        for item in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = item.path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("kv") => {}
                // A file a write didn't finish.
                Some("part") => {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                _ => continue,
            }
            let modified = item.metadata().and_then(|m| m.modified()).ok();
            match read_header(&path) {
                Some((tokens, bytes)) => found.push((modified, tokens, bytes, path)),
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        found.sort_by_key(|f| f.0);
        let mut store = Self {
            dir,
            entries: Vec::new(),
            budget,
            clock: 0,
            next_name: 0,
            writer: None,
        };
        for (_, tokens, bytes, file) in found {
            store.clock += 1;
            store.entries.push(Entry {
                tokens,
                file,
                bytes,
                data: None,
                used: store.clock,
            });
        }
        store
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Saves `state` after `tokens` (as many tokens as the state has positions). Saved
    /// chats that it continues (theirs are the first of its tokens) are replaced by it.
    pub fn put(&mut self, tokens: Vec<u32>, state: SavedState) {
        debug_assert_eq!(tokens.len(), state.len());
        if self
            .entries
            .iter()
            .any(|e| tokens.starts_with(&e.tokens) && e.writing())
        {
            // One it replaces is still being written: let that finish before removing it.
            self.flush();
        }
        self.entries.retain(|e| {
            let older = tokens.starts_with(&e.tokens);
            if older {
                let _ = std::fs::remove_file(&e.file);
            }
            !older
        });
        let file = self.new_file();
        let bytes = state.bytes().len() as u64;
        let data = Arc::new(state);
        self.write(file.clone(), tokens.clone(), data.clone());
        self.clock += 1;
        self.entries.push(Entry {
            tokens,
            file,
            bytes,
            data: Some(data),
            used: self.clock,
        });
        self.trim();
    }

    /// Saved chat `i`'s state, from memory or its file. `None` (and the chat forgotten)
    /// if the file can't be read.
    pub fn get(&mut self, i: usize) -> Option<Arc<SavedState>> {
        self.clock += 1;
        self.entries[i].used = self.clock;
        if let Some(d) = &self.entries[i].data {
            return Some(d.clone());
        }
        match read_state(&self.entries[i].file) {
            Some(state) => {
                let d = Arc::new(state);
                self.entries[i].data = Some(d.clone());
                self.trim();
                Some(d)
            }
            None => {
                let e = self.entries.remove(i);
                let _ = std::fs::remove_file(e.file);
                None
            }
        }
    }

    /// Forgets the saved chat `state` came from (`get` handed it out), and its file.
    pub fn forget(&mut self, state: &Arc<SavedState>) {
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.data.as_ref().is_some_and(|d| Arc::ptr_eq(d, state)))
        {
            let e = self.entries.remove(i);
            let _ = std::fs::remove_file(e.file);
        }
    }

    /// Waits for every file to be written.
    pub fn flush(&mut self) {
        if let Some((tx, handle)) = self.writer.take() {
            drop(tx);
            let _ = handle.join();
        }
    }

    fn new_file(&mut self) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_micros());
        self.next_name += 1;
        self.dir.join(format!("{stamp}-{}.kv", self.next_name))
    }

    fn write(&mut self, file: PathBuf, tokens: Vec<u32>, data: Arc<SavedState>) {
        if self.writer.is_none() {
            let (tx, rx) = channel::<ToWrite>();
            let handle = std::thread::spawn(move || {
                for (file, tokens, data) in rx {
                    if let Err(e) = write_file(&file, &tokens, &data) {
                        eprintln!("Couldn't save a chat's cache to {}: {e}", file.display());
                    }
                }
            });
            self.writer = Some((tx, handle));
        }
        let (tx, _) = self.writer.as_ref().unwrap();
        let _ = tx.send((file, tokens, data));
    }

    /// Keeps the files within the budget (the oldest go) and the newest few states in
    /// memory.
    fn trim(&mut self) {
        self.entries.sort_by_key(|e| e.used);
        let mut total: u64 = self.entries.iter().map(|e| e.bytes).sum();
        while total > self.budget && self.entries.len() > 1 {
            if self.entries[0].writing() {
                // Its file is still being written: let that finish before removing it.
                self.flush();
            }
            let e = self.entries.remove(0);
            total -= e.bytes;
            let _ = std::fs::remove_file(e.file);
        }
        let n = self.entries.len();
        for e in &mut self.entries[..n.saturating_sub(IN_MEMORY)] {
            // Only once its file is written (the writer has let go of it).
            if e.data.as_ref().is_some_and(|d| Arc::strong_count(d) == 1) {
                e.data = None;
            }
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.flush();
    }
}

fn write_file(file: &Path, tokens: &[u32], state: &SavedState) -> std::io::Result<()> {
    let part = file.with_extension("part");
    {
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&part)?);
        f.write_all(MAGIC)?;
        f.write_all(&(tokens.len() as u64).to_le_bytes())?;
        let mut t = Vec::with_capacity(tokens.len() * 4);
        for id in tokens {
            t.extend_from_slice(&id.to_le_bytes());
        }
        f.write_all(&t)?;
        f.write_all(&(state.bytes().len() as u64).to_le_bytes())?;
        f.write_all(state.bytes())?;
        f.flush()?;
    }
    std::fs::rename(&part, file)
}

fn read_u64(f: &mut impl Read) -> Option<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// The tokens and the state's size, checked against the file's size.
fn read_header(file: &Path) -> Option<(Vec<u32>, u64)> {
    let mut f = std::fs::File::open(file).ok()?;
    let size = f.metadata().ok()?.len();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let n = read_u64(&mut f)?;
    if n == 0 || n > size / 4 {
        return None;
    }
    let mut t = vec![0u8; n as usize * 4];
    f.read_exact(&mut t).ok()?;
    let bytes = read_u64(&mut f)?;
    if Some(size) != (16 + n * 4 + 8).checked_add(bytes) {
        return None;
    }
    let tokens = t
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    Some((tokens, bytes))
}

fn read_state(file: &Path) -> Option<SavedState> {
    let (tokens, bytes) = read_header(file)?;
    let mut f = std::fs::File::open(file).ok()?;
    let mut skip = vec![0u8; 16 + tokens.len() * 4 + 8];
    f.read_exact(&mut skip).ok()?;
    let mut data = vec![0u8; bytes as usize];
    f.read_exact(&mut data).ok()?;
    Some(SavedState::from_bytes(tokens.len(), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("vireo-store-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn state(tokens: &[u32], fill: u8, size: usize) -> SavedState {
        SavedState::from_bytes(tokens.len(), vec![fill; size])
    }

    #[test]
    fn saves_and_reads_back_after_a_restart() {
        let d = dir("restart");
        {
            let mut s = Store::open(d.clone(), DISK_BUDGET);
            s.put(vec![1, 2, 3], state(&[1, 2, 3], 7, 100));
            s.put(vec![9, 9], state(&[9, 9], 8, 50));
        }
        let mut s = Store::open(d.clone(), DISK_BUDGET);
        assert_eq!(s.entries().len(), 2);
        let i = s
            .entries()
            .iter()
            .position(|e| e.tokens == [1, 2, 3])
            .unwrap();
        let got = s.get(i).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got.bytes(), &[7u8; 100][..]);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_chat_that_goes_on_replaces_its_earlier_save() {
        let d = dir("replace");
        let mut s = Store::open(d.clone(), DISK_BUDGET);
        s.put(vec![1, 2], state(&[1, 2], 1, 10));
        s.put(vec![5], state(&[5], 1, 10));
        s.put(vec![1, 2, 3, 4], state(&[1, 2, 3, 4], 2, 10));
        let mut t: Vec<_> = s.entries().iter().map(|e| e.tokens.clone()).collect();
        t.sort();
        assert_eq!(t, vec![vec![1, 2, 3, 4], vec![5]]);
        s.flush();
        drop(s);
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_large_models_budget_holds_four_full_chats() {
        assert_eq!(budget_for(400 << 20), 4 << 30); // a small model: the least
        assert_eq!(budget_for(4400 << 20), 4 * (4400 << 20)); // Gemma 4 31B at 32K
        assert_eq!(budget_for(30 << 30), 32 << 30); // never more than the most
    }

    #[test]
    fn the_oldest_go_past_the_budget_and_bad_files_are_dropped() {
        let d = dir("budget");
        let mut s = Store::open(d.clone(), 250);
        s.put(vec![1], state(&[1], 1, 100));
        s.put(vec![2], state(&[2], 1, 100));
        let _ = s.get(0); // chat 1 used again: now chat 2 is the oldest
        s.put(vec![3], state(&[3], 1, 100));
        let mut t: Vec<_> = s.entries().iter().map(|e| e.tokens[0]).collect();
        t.sort();
        assert_eq!(t, vec![1, 3]);
        drop(s);
        std::fs::write(d.join("junk.kv"), b"VIREOKV1 not really").unwrap();
        std::fs::write(d.join("half.part"), b"x").unwrap();
        let s = Store::open(d.clone(), 250);
        assert_eq!(s.entries().len(), 2);
        assert!(!d.join("junk.kv").exists() && !d.join("half.part").exists());
        let _ = std::fs::remove_dir_all(d);
    }
}
