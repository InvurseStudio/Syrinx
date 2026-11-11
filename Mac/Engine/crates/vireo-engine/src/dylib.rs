//! Loads a shared library at runtime and finds functions in it, with the operating
//! system's own calls (`LoadLibraryW` / `GetProcAddress` on Windows, `dlopen` / `dlsym`
//! elsewhere).
//!
//! The engine loads GPU drivers this way, not by linking to them, so it starts on any
//! computer and uses a GPU only if its driver is there.

use std::ffi::{CString, c_void};

pub struct Library {
    handle: *mut c_void,
}

// A loaded library's functions can be called from any thread.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// Loads the library `name` ("nvcuda.dll", "libcuda.so.1"), or `None` if it isn't
    /// installed.
    pub fn open(name: &str) -> Option<Self> {
        let handle = sys::open(name);
        (!handle.is_null()).then_some(Self { handle })
    }

    /// The function `name`, as a function pointer of type `F`, or `None` if the library
    /// doesn't have it.
    ///
    /// # Safety
    /// `F` must be a function pointer type matching the function's real signature and
    /// calling convention.
    pub unsafe fn function<F: Copy>(&self, name: &str) -> Option<F> {
        assert_eq!(
            size_of::<F>(),
            size_of::<*mut c_void>(),
            "F must be a function pointer"
        );
        let name = CString::new(name).ok()?;
        let address = sys::symbol(self.handle, &name);
        if address.is_null() {
            return None;
        }
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        sys::close(self.handle);
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::{CStr, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub fn open(name: &str) -> *mut c_void {
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        unsafe { LoadLibraryW(wide.as_ptr()) }
    }

    pub fn symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { GetProcAddress(handle, name.as_ptr().cast()) }
    }

    pub fn close(handle: *mut c_void) {
        unsafe { FreeLibrary(handle) };
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    pub fn open(name: &str) -> *mut c_void {
        match CString::new(name) {
            Ok(name) => unsafe { dlopen(name.as_ptr(), RTLD_NOW) },
            Err(_) => std::ptr::null_mut(),
        }
    }

    pub fn symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { dlsym(handle, name.as_ptr()) }
    }

    pub fn close(handle: *mut c_void) {
        unsafe { dlclose(handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_system_function() {
        #[cfg(windows)]
        let (lib, function) = ("kernel32.dll", "GetTickCount64");
        #[cfg(unix)]
        let (lib, function) = ("libc.so.6", "getpid");
        let Some(library) = Library::open(lib) else {
            return; // a Unix without glibc
        };
        let f: Option<unsafe extern "system" fn() -> u64> = unsafe { library.function(function) };
        assert!(f.is_some());
        let missing: Option<unsafe extern "system" fn()> =
            unsafe { library.function("NoSuchFunction") };
        assert!(missing.is_none());
        assert!(Library::open("no-such-library-vireo.dll").is_none());
    }
}
