//! The CUDA backend: Vireo's own kernels (`kernels/vireo.cu`) on NVIDIA GPUs, run
//! through the driver API directly.
//!
//! No CUDA runtime, cuBLAS or other NVIDIA library is linked or shipped: the engine
//! loads the driver (`nvcuda.dll`) when asked for the GPU, hands it the kernels' PTX,
//! and the driver compiles them for whatever NVIDIA GPU it finds. On a computer
//! without one, [`Gpu::open`] says why and the CPU backend is used instead.
//!
//! All GPU work goes through the thread that opened the [`Gpu`]: the driver's context
//! is made current on that thread.

mod driver;

use std::ffi::{CString, c_void};
use std::fmt;
use std::sync::Arc;

use driver::*;

use crate::tensor_type::TensorType;

pub use driver::CuDevicePtr;

#[derive(Debug)]
pub enum Error {
    /// No usable NVIDIA GPU: no driver, no device, or a build without kernels.
    Unavailable(String),
    /// A driver call failed.
    Driver {
        call: &'static str,
        code: i32,
        name: String,
    },
    /// A tensor type without a GPU kernel yet.
    Unsupported(TensorType),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unavailable(why) => write!(f, "no GPU: {why}"),
            Error::Driver { call, code, name } => {
                write!(f, "the GPU driver's {call} failed: {name} ({code})")
            }
            Error::Unsupported(kind) => write!(f, "{kind} has no GPU kernel yet"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The driver, its context on the GPU, and the loaded kernels. Shared by everything
/// that holds GPU memory, so it's released last.
struct Context {
    driver: Driver,
    device: CuDevice,
    module: CuModule,
}

// The context is only used from the thread that opened it (see the module docs);
// the raw pointers inside are handles, not memory this process reads.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    fn check(&self, call: &'static str, result: CuResult) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name = std::ptr::null();
        unsafe { (self.driver.cuGetErrorName)(result, &mut name) };
        let name = if name.is_null() {
            "unknown error".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Driver {
            call,
            code: result,
            name,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.cuModuleUnload)(self.module);
            (self.driver.cuDevicePrimaryCtxRelease_v2)(self.device);
        }
    }
}

/// GPU memory. Freed when dropped.
pub struct Buffer {
    context: Arc<Context>,
    ptr: CuDevicePtr,
    bytes: usize,
}

impl Buffer {
    /// The GPU address of the start.
    pub fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }

    /// The GPU address of f32 number `index`.
    pub fn f32_at(&self, index: usize) -> CuDevicePtr {
        debug_assert!(index * 4 <= self.bytes);
        self.ptr + (index * 4) as u64
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Copies `data` to the start of the buffer.
    pub fn write_f32(&self, data: &[f32]) -> Result<()> {
        assert!(data.len() * 4 <= self.bytes);
        let r = unsafe {
            (self.context.driver.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr().cast(), data.len() * 4)
        };
        self.context.check("cuMemcpyHtoD", r)
    }

    /// Copies the start of the buffer into `out`, waiting for queued GPU work first.
    pub fn read_f32(&self, out: &mut [f32]) -> Result<()> {
        assert!(out.len() * 4 <= self.bytes);
        let r = unsafe {
            (self.context.driver.cuMemcpyDtoH_v2)(out.as_mut_ptr().cast(), self.ptr, out.len() * 4)
        };
        self.context.check("cuMemcpyDtoH", r)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (self.context.driver.cuMemFree_v2)(self.ptr) };
    }
}

struct Kernels {
    matvec_f32: CuFunction,
    matvec_f16: CuFunction,
    matvec_q8_0: CuFunction,
    matvec_q4_k: CuFunction,
    matvec_q6_k: CuFunction,
    rms_norm: CuFunction,
    rope_norm: CuFunction,
    attention: CuFunction,
    swiglu: CuFunction,
    add: CuFunction,
}

/// What the driver reports about the GPU.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
    pub memory_bytes: usize,
    /// The driver's CUDA version, as 13050 for 13.5.
    pub driver_version: i32,
}

/// An NVIDIA GPU with Vireo's kernels loaded.
pub struct Gpu {
    context: Arc<Context>,
    kernels: Kernels,
    pub info: DeviceInfo,
}

#[cfg(vireo_cuda)]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vireo.ptx"));

/// The number of threads the per-row kernels use per block: 4 warps, 4 rows.
const ROWS_PER_BLOCK: u32 = 4;

impl Gpu {
    /// Opens the first NVIDIA GPU.
    pub fn open() -> Result<Self> {
        #[cfg(not(vireo_cuda))]
        return Err(Error::Unavailable(
            "this build has no GPU kernels (the CUDA Toolkit wasn't found when it was built)"
                .into(),
        ));
        #[cfg(vireo_cuda)]
        Self::open_device(0)
    }

    #[cfg(vireo_cuda)]
    fn open_device(ordinal: i32) -> Result<Self> {
        let driver = Driver::load().map_err(Error::Unavailable)?;
        let call = |name: &'static str, r: CuResult| {
            if r == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(Error::Driver {
                    call: name,
                    code: r,
                    name: format!("error {r}"),
                })
            }
        };
        unsafe {
            call("cuInit", (driver.cuInit)(0))?;
            let mut count = 0;
            call("cuDeviceGetCount", (driver.cuDeviceGetCount)(&mut count))?;
            if count <= ordinal {
                return Err(Error::Unavailable(
                    "the NVIDIA driver is installed, but there's no NVIDIA GPU".into(),
                ));
            }
            let mut device = 0;
            call("cuDeviceGet", (driver.cuDeviceGet)(&mut device, ordinal))?;
            let mut name = [0 as std::ffi::c_char; 256];
            call(
                "cuDeviceGetName",
                (driver.cuDeviceGetName)(name.as_mut_ptr(), 256, device),
            )?;
            let attribute = |attribute| {
                let mut value = 0;
                (driver.cuDeviceGetAttribute)(&mut value, attribute, device);
                value
            };
            let mut memory = 0;
            call(
                "cuDeviceTotalMem",
                (driver.cuDeviceTotalMem_v2)(&mut memory, device),
            )?;
            let mut version = 0;
            (driver.cuDriverGetVersion)(&mut version);
            let info = DeviceInfo {
                name: std::ffi::CStr::from_ptr(name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                compute_capability: (
                    attribute(ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                    attribute(ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
                ),
                multiprocessors: attribute(ATTRIBUTE_MULTIPROCESSOR_COUNT),
                memory_bytes: memory,
                driver_version: version,
            };

            let mut context = std::ptr::null_mut();
            call(
                "cuDevicePrimaryCtxRetain",
                (driver.cuDevicePrimaryCtxRetain)(&mut context, device),
            )?;
            call("cuCtxSetCurrent", (driver.cuCtxSetCurrent)(context))?;
            let ptx = CString::new(PTX).expect("PTX has no NUL bytes");
            let mut module = std::ptr::null_mut();
            let loaded = (driver.cuModuleLoadData)(&mut module, ptx.as_ptr().cast());
            if loaded != CUDA_SUCCESS {
                (driver.cuDevicePrimaryCtxRelease_v2)(device);
                return Err(Error::Driver {
                    call: "cuModuleLoadData (compiling the kernels for this GPU)",
                    code: loaded,
                    name: format!("error {loaded}"),
                });
            }
            let context = Arc::new(Context {
                driver,
                device,
                module,
            });

            let function = |name: &str| -> Result<CuFunction> {
                let c_name = CString::new(name).unwrap();
                let mut f = std::ptr::null_mut();
                let r =
                    (context.driver.cuModuleGetFunction)(&mut f, context.module, c_name.as_ptr());
                context.check("cuModuleGetFunction", r).map(|_| f)
            };
            let kernels = Kernels {
                matvec_f32: function("matvec_f32")?,
                matvec_f16: function("matvec_f16")?,
                matvec_q8_0: function("matvec_q8_0")?,
                matvec_q4_k: function("matvec_q4_k")?,
                matvec_q6_k: function("matvec_q6_k")?,
                rms_norm: function("rms_norm")?,
                rope_norm: function("rope_norm")?,
                attention: function("attention")?,
                swiglu: function("swiglu")?,
                add: function("add")?,
            };
            Ok(Self {
                context,
                kernels,
                info,
            })
        }
    }

    /// Whether a weight of this type can be multiplied on the GPU.
    pub fn supports(kind: TensorType) -> bool {
        matches!(
            kind,
            TensorType::F32
                | TensorType::F16
                | TensorType::Q8_0
                | TensorType::Q4_K
                | TensorType::Q6_K
        )
    }

    /// Free GPU memory, in bytes.
    pub fn free_memory(&self) -> usize {
        let (mut free, mut total) = (0, 0);
        unsafe { (self.context.driver.cuMemGetInfo_v2)(&mut free, &mut total) };
        free
    }

    /// `bytes` of uninitialized GPU memory.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let mut ptr = 0;
        let r = unsafe { (self.context.driver.cuMemAlloc_v2)(&mut ptr, bytes.max(4)) };
        self.context.check("cuMemAlloc", r)?;
        Ok(Buffer {
            context: Arc::clone(&self.context),
            ptr,
            bytes,
        })
    }

    /// A copy of `data` in GPU memory.
    pub fn upload(&self, data: &[u8]) -> Result<Buffer> {
        let buffer = self.alloc(data.len())?;
        let r = unsafe {
            (self.context.driver.cuMemcpyHtoD_v2)(buffer.ptr, data.as_ptr().cast(), data.len())
        };
        self.context.check("cuMemcpyHtoD", r)?;
        Ok(buffer)
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<Buffer> {
        let buffer = self.alloc(data.len() * 4)?;
        buffer.write_f32(data)?;
        Ok(buffer)
    }

    /// Waits for all queued GPU work, and reports any error a kernel hit.
    pub fn synchronize(&self) -> Result<()> {
        let r = unsafe { (self.context.driver.cuCtxSynchronize)() };
        self.context.check("cuCtxSynchronize", r)
    }

    /// Queues `kernel` on `grid` blocks of `block` threads. `params` point at the
    /// kernel's arguments, in order; the driver copies them before returning.
    fn launch(
        &self,
        kernel: CuFunction,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let r = unsafe {
            (self.context.driver.cuLaunchKernel)(
                kernel,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.context.check("cuLaunchKernel", r)
    }

    /// `y = W x`: `w` holds `rows` rows of `cols` numbers of type `kind`.
    pub fn matvec(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let kernel = match kind {
            TensorType::F32 => self.kernels.matvec_f32,
            TensorType::F16 => self.kernels.matvec_f16,
            TensorType::Q8_0 => self.kernels.matvec_q8_0,
            TensorType::Q4_K => self.kernels.matvec_q4_k,
            TensorType::Q6_K => self.kernels.matvec_q6_k,
            other => return Err(Error::Unsupported(other)),
        };
        let (mut w, mut x, mut y) = (w, x, y);
        let (mut rows_arg, mut cols_arg) = (rows as i32, cols as i32);
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK);
        self.launch(
            kernel,
            grid,
            ROWS_PER_BLOCK * 32,
            &mut [
                arg(&mut w),
                arg(&mut x),
                arg(&mut y),
                arg(&mut rows_arg),
                arg(&mut cols_arg),
            ],
        )
    }

    pub fn rms_norm(
        &self,
        x: CuDevicePtr,
        weight: CuDevicePtr,
        out: CuDevicePtr,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let (mut x, mut weight, mut out, mut n, mut eps) = (x, weight, out, n as i32, eps);
        self.launch(
            self.kernels.rms_norm,
            1,
            256,
            &mut [
                arg(&mut x),
                arg(&mut weight),
                arg(&mut out),
                arg(&mut n),
                arg(&mut eps),
            ],
        )
    }

    /// RoPE (adjacent pairs) in place; `factors` is 0 for none.
    pub fn rope_norm(
        &self,
        x: CuDevicePtr,
        heads: usize,
        head_dim: usize,
        pos: usize,
        base: f32,
        factors: CuDevicePtr,
    ) -> Result<()> {
        let threads = (heads * head_dim / 2) as u32;
        let (mut x, mut heads, mut head_dim, mut pos, mut base, mut factors) =
            (x, heads as i32, head_dim as i32, pos as i32, base, factors);
        self.launch(
            self.kernels.rope_norm,
            threads.div_ceil(128),
            128,
            &mut [
                arg(&mut x),
                arg(&mut heads),
                arg(&mut head_dim),
                arg(&mut pos),
                arg(&mut base),
                arg(&mut factors),
            ],
        )
    }

    /// Attention of `heads` query heads over `len` cached positions.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: CuDevicePtr,
        keys: CuDevicePtr,
        values: CuDevicePtr,
        out: CuDevicePtr,
        scores: CuDevicePtr,
        heads: usize,
        len: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        capacity: usize,
        scale: f32,
    ) -> Result<()> {
        let (mut q, mut keys, mut values, mut out, mut scores) = (q, keys, values, out, scores);
        let (mut len, mut head_dim, mut kv_dim, mut group, mut capacity, mut scale) = (
            len as i32,
            head_dim as i32,
            kv_dim as i32,
            group as i32,
            capacity as i32,
            scale,
        );
        self.launch(
            self.kernels.attention,
            heads as u32,
            128,
            &mut [
                arg(&mut q),
                arg(&mut keys),
                arg(&mut values),
                arg(&mut out),
                arg(&mut scores),
                arg(&mut len),
                arg(&mut head_dim),
                arg(&mut kv_dim),
                arg(&mut group),
                arg(&mut capacity),
                arg(&mut scale),
            ],
        )
    }

    pub fn swiglu(&self, gate: CuDevicePtr, up: CuDevicePtr, n: usize) -> Result<()> {
        let (mut gate, mut up, mut n_arg) = (gate, up, n as i32);
        self.launch(
            self.kernels.swiglu,
            (n as u32).div_ceil(256),
            256,
            &mut [arg(&mut gate), arg(&mut up), arg(&mut n_arg)],
        )
    }

    pub fn add(&self, x: CuDevicePtr, y: CuDevicePtr, n: usize) -> Result<()> {
        let (mut x, mut y, mut n_arg) = (x, y, n as i32);
        self.launch(
            self.kernels.add,
            (n as u32).div_ceil(256),
            256,
            &mut [arg(&mut x), arg(&mut y), arg(&mut n_arg)],
        )
    }
}

/// A kernel argument, as the pointer to it that `cuLaunchKernel` takes.
fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}
