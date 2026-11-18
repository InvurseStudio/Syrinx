//! Maps a file into memory, read-only, with the operating system's own calls
//! (`CreateFileMappingW` / `MapViewOfFile` on Windows, `mmap` elsewhere).
//!
//! A model's weights are gigabytes. Mapping them means nothing is read until it's used,
//! the operating system can drop pages it needs back and reread them later, and a GPU
//! upload can copy straight from the mapping.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

/// A file's bytes, mapped read-only. Unmapped when dropped.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    /// The file, kept open for [`Mmap::release`].
    file: Option<File>,
}

// The mapping is read-only and owned by this value alone, so it can be shared and moved
// between threads like a `&[u8]` or a `Box<[u8]>`.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Maps the whole file at `path`.
    ///
    /// # Safety
    ///
    /// The file must not be changed or shortened while the mapping exists: the bytes
    /// would change under the caller, and reading past a shortened end crashes. Model
    /// files are only ever read, so this holds as long as nothing else rewrites them.
    pub unsafe fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::other("the file is too large to map on this computer"))?;
        if len == 0 {
            // Neither Windows nor POSIX maps an empty file.
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                file: None,
            });
        }
        let ptr = sys::map(&file, len)?;
        Ok(Self {
            ptr,
            len,
            file: Some(file),
        })
    }

    /// Gives the pages read so far back to the operating system; the bytes stay readable
    /// and are read from the file again when next used. For after a model's weights are
    /// copied to the GPU: until then the process holds the file's pages beside the copy,
    /// twice the model's size. Does nothing on Windows.
    pub fn release(&self) {
        if let Some(file) = &self.file {
            // Failing leaves the mapping as it was, pages and all.
            let _ = unsafe { sys::release(self.ptr, self.len, file) };
        }
    }
}

impl Deref for Mmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // The mapping covers `len` readable bytes for as long as `self` lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { sys::unmap(self.ptr, self.len) };
        }
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::ptr::{NonNull, null, null_mut};

    type Handle = *mut c_void;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            file: Handle,
            attributes: *mut c_void,
            protect: u32,
            maximum_size_high: u32,
            maximum_size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            desired_access: u32,
            file_offset_high: u32,
            file_offset_low: u32,
            bytes_to_map: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base_address: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    pub fn map(file: &File, _len: usize) -> io::Result<NonNull<u8>> {
        // Sizes of 0 mean "the whole file", for both the mapping and the view.
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                null_mut(),
                PAGE_READONLY,
                0,
                0,
                null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
        let error = view.is_null().then(io::Error::last_os_error);
        // The view keeps the mapping alive on its own.
        unsafe { CloseHandle(mapping) };
        match error {
            Some(error) => Err(error),
            None => Ok(NonNull::new(view.cast()).expect("checked above")),
        }
    }

    /// # Safety
    /// `ptr` must be a view returned by `map` and not unmapped yet.
    pub unsafe fn release(_ptr: NonNull<u8>, _len: usize, _file: &File) -> io::Result<()> {
        Ok(())
    }

    /// # Safety
    /// `ptr` must be a mapping returned by `map` and not unmapped yet.
    pub unsafe fn unmap(ptr: NonNull<u8>, _len: usize) {
        unsafe { UnmapViewOfFile(ptr.as_ptr().cast()) };
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::ptr::{NonNull, null_mut};

    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;
    const MAP_FIXED: c_int = 0x10;

    // `off_t` is 64 bits on macOS and 64-bit Linux, the only Unix systems Vireo targets.
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    pub fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        let ptr = unsafe { mmap(null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        NonNull::new(ptr.cast()).ok_or_else(|| io::Error::other("mmap returned a null address"))
    }

    /// Maps the file again over its own mapping, in one step: the same bytes at the same
    /// address, none of them resident.
    ///
    /// # Safety
    /// `ptr` and `len` must be a mapping of `file` returned by `map` and not unmapped yet.
    pub unsafe fn release(ptr: NonNull<u8>, len: usize, file: &File) -> io::Result<()> {
        let again = unsafe {
            mmap(
                ptr.as_ptr().cast(),
                len,
                PROT_READ,
                MAP_PRIVATE | MAP_FIXED,
                file.as_raw_fd(),
                0,
            )
        };
        if again as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// # Safety
    /// `ptr` and `len` must be a mapping returned by `map` and not unmapped yet.
    pub unsafe fn unmap(ptr: NonNull<u8>, len: usize) {
        unsafe { munmap(ptr.as_ptr().cast(), len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("vireo-mmap-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn maps_a_file() {
        let contents: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let path = temp_file("data", &contents);
        let map = unsafe { Mmap::open(&path) }.unwrap();
        assert_eq!(&map[..], &contents[..]);
        drop(map); // Windows can't delete a file that's still mapped.
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_the_same_bytes_after_a_release() {
        let contents: Vec<u8> = (0..300_000u32).map(|i| (i % 241) as u8).collect();
        let path = temp_file("release", &contents);
        let map = unsafe { Mmap::open(&path) }.unwrap();
        let before = map.as_ptr();
        assert_eq!(&map[..], &contents[..]);
        map.release();
        assert_eq!(map.as_ptr(), before);
        assert_eq!(&map[..], &contents[..]);
        map.release();
        map.release();
        assert_eq!(&map[..], &contents[..]);
        drop(map);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn maps_an_empty_file() {
        let path = temp_file("empty", b"");
        let map = unsafe { Mmap::open(&path) }.unwrap();
        assert!(map.is_empty());
        drop(map);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_a_missing_file() {
        let path = std::env::temp_dir().join("vireo-mmap-does-not-exist");
        let error = unsafe { Mmap::open(&path) }.err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }
}
