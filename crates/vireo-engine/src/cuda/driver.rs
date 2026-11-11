//! The CUDA driver API: the functions of `nvcuda.dll` (`libcuda.so.1` on Linux) the
//! engine uses, loaded at runtime. Each is declared from NVIDIA's `cuda.h`, which
//! documents them as stable across driver versions.

#![allow(non_snake_case)]

use std::ffi::{c_char, c_void};

use crate::dylib::Library;

pub type CuResult = i32;
pub type CuDevice = i32;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuGraph = *mut c_void;
pub type CuGraphExec = *mut c_void;
/// A GPU memory address.
pub type CuDevicePtr = u64;

pub const CUDA_SUCCESS: CuResult = 0;
pub const ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
/// A stream that doesn't wait for (or hold up) the legacy default stream.
pub const STREAM_NON_BLOCKING: u32 = 1;
/// How much dynamic shared memory a kernel may ask for (over 48 KB needs saying so).
pub const FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
/// Capturing on this stream doesn't restrict what other threads do meanwhile.
pub const STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;

macro_rules! driver_functions {
    ($($name:ident($($arg:ident: $ty:ty),*);)*) => {
        /// The driver's functions, found by name when the library loads.
        pub struct Driver {
            _library: Library,
            $(pub $name: unsafe extern "system" fn($($ty),*) -> CuResult,)*
        }

        impl Driver {
            /// Loads the NVIDIA driver, or says why it can't.
            pub fn load() -> Result<Self, String> {
                let name = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
                let library = Library::open(name)
                    .ok_or_else(|| format!("the NVIDIA driver ({name}) isn't installed"))?;
                Ok(Self {
                    $($name: unsafe { library.function(stringify!($name)) }
                        .ok_or_else(|| format!("the NVIDIA driver has no {}: it may be too old", stringify!($name)))?,)*
                    _library: library,
                })
            }
        }
    };
}

driver_functions! {
    cuInit(flags: u32);
    cuDriverGetVersion(version: *mut i32);
    cuDeviceGetCount(count: *mut i32);
    cuDeviceGet(device: *mut CuDevice, ordinal: i32);
    cuDeviceGetName(name: *mut c_char, len: i32, device: CuDevice);
    cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: CuDevice);
    cuDeviceTotalMem_v2(bytes: *mut usize, device: CuDevice);
    cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice);
    cuDevicePrimaryCtxRelease_v2(device: CuDevice);
    cuCtxSetCurrent(context: CuContext);
    cuModuleLoadData(module: *mut CuModule, image: *const c_void);
    cuModuleUnload(module: CuModule);
    cuModuleGetFunction(function: *mut CuFunction, module: CuModule, name: *const c_char);
    cuFuncSetAttribute(function: CuFunction, attribute: i32, value: i32);
    cuMemAlloc_v2(pointer: *mut CuDevicePtr, bytes: usize);
    cuMemFree_v2(pointer: CuDevicePtr);
    cuMemGetInfo_v2(free: *mut usize, total: *mut usize);
    cuMemcpyHtoDAsync_v2(destination: CuDevicePtr, source: *const c_void, bytes: usize, stream: CuStream);
    cuMemcpyDtoHAsync_v2(destination: *mut c_void, source: CuDevicePtr, bytes: usize, stream: CuStream);
    cuStreamCreate(stream: *mut CuStream, flags: u32);
    cuStreamDestroy_v2(stream: CuStream);
    cuStreamSynchronize(stream: CuStream);
    cuStreamBeginCapture_v2(stream: CuStream, mode: i32);
    cuStreamEndCapture(stream: CuStream, graph: *mut CuGraph);
    cuGraphInstantiateWithFlags(exec: *mut CuGraphExec, graph: CuGraph, flags: u64);
    cuGraphLaunch(exec: CuGraphExec, stream: CuStream);
    cuGraphExecDestroy(exec: CuGraphExec);
    cuGraphDestroy(graph: CuGraph);
    cuLaunchKernel(
        function: CuFunction,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_bytes: u32, stream: CuStream,
        params: *mut *mut c_void, extra: *mut *mut c_void
    );
    cuGetErrorName(error: CuResult, name: *mut *const c_char);
}
