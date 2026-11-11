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
    matvec_q4_0: CuFunction,
    matvec_q8_0: CuFunction,
    matvec_q4_k: CuFunction,
    matvec_q5_k: CuFunction,
    matvec_q6_k: CuFunction,
    rms_norm: CuFunction,
    l2_norm: CuFunction,
    rope: CuFunction,
    attention: CuFunction,
    swiglu: CuFunction,
    geglu: CuFunction,
    add: CuFunction,
    scale: CuFunction,
    softcap: CuFunction,
    split_query: CuFunction,
    apply_gate: CuFunction,
    conv_silu: CuFunction,
    delta_net: CuFunction,
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
                matvec_q4_0: function("matvec_q4_0")?,
                matvec_q8_0: function("matvec_q8_0")?,
                matvec_q4_k: function("matvec_q4_k")?,
                matvec_q5_k: function("matvec_q5_k")?,
                matvec_q6_k: function("matvec_q6_k")?,
                rms_norm: function("rms_norm")?,
                l2_norm: function("l2_norm")?,
                rope: function("rope")?,
                attention: function("attention")?,
                swiglu: function("swiglu")?,
                geglu: function("geglu")?,
                add: function("add")?,
                scale: function("scale")?,
                softcap: function("softcap")?,
                split_query: function("split_query")?,
                apply_gate: function("apply_gate")?,
                conv_silu: function("conv_silu")?,
                delta_net: function("delta_net")?,
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
                | TensorType::Q4_0
                | TensorType::Q8_0
                | TensorType::Q4_K
                | TensorType::Q5_K
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

    /// Queues `kernel` with `args`, in the kernel's parameter order.
    fn run(&self, kernel: CuFunction, grid: u32, block: u32, args: &[Arg]) -> Result<()> {
        let mut values = args.to_vec();
        let mut params: Vec<*mut c_void> = values
            .iter_mut()
            .map(|a| match a {
                Arg::Ptr(v) => (v as *mut u64).cast(),
                Arg::I32(v) => (v as *mut i32).cast(),
                Arg::F32(v) => (v as *mut f32).cast(),
            })
            .collect();
        self.launch(kernel, grid, block, &mut params)
    }

    /// One thread per number, 256 to a block.
    fn run_per_number(&self, kernel: CuFunction, n: usize, args: &[Arg]) -> Result<()> {
        self.run(kernel, (n as u32).div_ceil(256), 256, args)
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
            TensorType::Q4_0 => self.kernels.matvec_q4_0,
            TensorType::Q8_0 => self.kernels.matvec_q8_0,
            TensorType::Q4_K => self.kernels.matvec_q4_k,
            TensorType::Q5_K => self.kernels.matvec_q5_k,
            TensorType::Q6_K => self.kernels.matvec_q6_k,
            other => return Err(Error::Unsupported(other)),
        };
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK);
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
        ];
        self.run(kernel, grid, ROWS_PER_BLOCK * 32, &args)
    }

    /// RMSNorm of `n` numbers: `out = x / rms(x) × weight`.
    pub fn rms_norm(
        &self,
        x: CuDevicePtr,
        weight: CuDevicePtr,
        out: CuDevicePtr,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_rows(x, weight, out, 1, n, eps)
    }

    /// RMSNorm of each of `rows` rows of `n` numbers (heads, layer slices); `weight` is
    /// the same for every row, or 0 for none. In place is fine.
    pub fn rms_norm_rows(
        &self,
        x: CuDevicePtr,
        weight: CuDevicePtr,
        out: CuDevicePtr,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(x),
            Arg::Ptr(weight),
            Arg::Ptr(out),
            Arg::I32(n as i32),
            Arg::F32(eps),
        ];
        self.run(self.kernels.rms_norm, rows as u32, 256, &args)
    }

    /// Each of `rows` rows of `n` numbers divided by its length (at least `eps`), in place.
    pub fn l2_norm_rows(&self, x: CuDevicePtr, rows: usize, n: usize, eps: f32) -> Result<()> {
        let args = [Arg::Ptr(x), Arg::I32(n as i32), Arg::F32(eps)];
        self.run(self.kernels.l2_norm, rows as u32, 128, &args)
    }

    /// RoPE (adjacent pairs, Llama's) over whole heads, in place; `factors` is 0 for none.
    pub fn rope_norm(
        &self,
        x: CuDevicePtr,
        heads: usize,
        head_dim: usize,
        pos: usize,
        base: f32,
        factors: CuDevicePtr,
    ) -> Result<()> {
        self.rope(x, heads, head_dim, head_dim, pos, base, factors, false)
    }

    /// RoPE on the first `rope_dims` numbers of each head, in place: adjacent pairs, or
    /// halves with `neox`. `factors` is 0 for none.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: CuDevicePtr,
        heads: usize,
        head_dim: usize,
        rope_dims: usize,
        pos: usize,
        base: f32,
        factors: CuDevicePtr,
        neox: bool,
    ) -> Result<()> {
        let threads = (heads * rope_dims / 2) as u32;
        let args = [
            Arg::Ptr(x),
            Arg::I32(heads as i32),
            Arg::I32(head_dim as i32),
            Arg::I32(rope_dims as i32),
            Arg::I32(pos as i32),
            Arg::F32(base),
            Arg::Ptr(factors),
            Arg::I32(neox as i32),
        ];
        self.run(self.kernels.rope, threads.div_ceil(128), 128, &args)
    }

    /// Attention of `heads` query heads over cached positions `first..len`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: CuDevicePtr,
        keys: CuDevicePtr,
        values: CuDevicePtr,
        out: CuDevicePtr,
        scores: CuDevicePtr,
        heads: usize,
        first: usize,
        len: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        capacity: usize,
        scale: f32,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(q),
            Arg::Ptr(keys),
            Arg::Ptr(values),
            Arg::Ptr(out),
            Arg::Ptr(scores),
            Arg::I32(first as i32),
            Arg::I32(len as i32),
            Arg::I32(head_dim as i32),
            Arg::I32(kv_dim as i32),
            Arg::I32(group as i32),
            Arg::I32(capacity as i32),
            Arg::F32(scale),
        ];
        self.run(self.kernels.attention, heads as u32, 128, &args)
    }

    /// `gate = SiLU(gate) × up`
    pub fn swiglu(&self, gate: CuDevicePtr, up: CuDevicePtr, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.swiglu,
            n,
            &[Arg::Ptr(gate), Arg::Ptr(up), Arg::I32(n as i32)],
        )
    }

    /// `gate = GELU(gate) × up` (tanh approximation)
    pub fn geglu(&self, gate: CuDevicePtr, up: CuDevicePtr, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.geglu,
            n,
            &[Arg::Ptr(gate), Arg::Ptr(up), Arg::I32(n as i32)],
        )
    }

    /// `x += y`
    pub fn add(&self, x: CuDevicePtr, y: CuDevicePtr, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.add,
            n,
            &[Arg::Ptr(x), Arg::Ptr(y), Arg::I32(n as i32)],
        )
    }

    /// `x *= s`
    pub fn scale(&self, x: CuDevicePtr, s: f32, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.scale,
            n,
            &[Arg::Ptr(x), Arg::F32(s), Arg::I32(n as i32)],
        )
    }

    /// `x = cap × tanh(x / cap)`
    pub fn softcap(&self, x: CuDevicePtr, cap: f32, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.softcap,
            n,
            &[Arg::Ptr(x), Arg::F32(cap), Arg::I32(n as i32)],
        )
    }

    /// The queries out of gated attention's `[query | gate]` per head.
    pub fn split_query(
        &self,
        query_gate: CuDevicePtr,
        q: CuDevicePtr,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(query_gate),
            Arg::Ptr(q),
            Arg::I32(heads as i32),
            Arg::I32(head_dim as i32),
        ];
        self.run_per_number(self.kernels.split_query, heads * head_dim, &args)
    }

    /// `out ×= sigmoid(gate)`, the gates being the second halves of `[query | gate]`.
    pub fn apply_gate(
        &self,
        out: CuDevicePtr,
        query_gate: CuDevicePtr,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(out),
            Arg::Ptr(query_gate),
            Arg::I32(heads as i32),
            Arg::I32(head_dim as i32),
        ];
        self.run_per_number(self.kernels.apply_gate, heads * head_dim, &args)
    }

    /// The causal convolution and SiLU of Gated DeltaNet, moving the history along.
    pub fn conv_silu(
        &self,
        current: CuDevicePtr,
        history: CuDevicePtr,
        weights: CuDevicePtr,
        out: CuDevicePtr,
        channels: usize,
        kernel: usize,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(current),
            Arg::Ptr(history),
            Arg::Ptr(weights),
            Arg::Ptr(out),
            Arg::I32(channels as i32),
            Arg::I32(kernel as i32),
        ];
        self.run_per_number(self.kernels.conv_silu, channels, &args)
    }

    /// One token of Gated DeltaNet for `v_heads` heads with states of `n` × `n` (n ≤ 256),
    /// including the gated RMSNorm.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_net(
        &self,
        q: CuDevicePtr,
        k: CuDevicePtr,
        v: CuDevicePtr,
        beta_raw: CuDevicePtr,
        alpha_raw: CuDevicePtr,
        dt_bias: CuDevicePtr,
        a: CuDevicePtr,
        state: CuDevicePtr,
        z: CuDevicePtr,
        norm: CuDevicePtr,
        out: CuDevicePtr,
        v_heads: usize,
        n: usize,
        k_heads: usize,
        eps: f32,
    ) -> Result<()> {
        assert!(
            n <= 256 && n.is_multiple_of(32),
            "state size {n}: the kernel takes multiples of 32 up to 256"
        );
        let args = [
            Arg::Ptr(q),
            Arg::Ptr(k),
            Arg::Ptr(v),
            Arg::Ptr(beta_raw),
            Arg::Ptr(alpha_raw),
            Arg::Ptr(dt_bias),
            Arg::Ptr(a),
            Arg::Ptr(state),
            Arg::Ptr(z),
            Arg::Ptr(norm),
            Arg::Ptr(out),
            Arg::I32(n as i32),
            Arg::I32(k_heads as i32),
            Arg::F32(eps),
        ];
        self.run(self.kernels.delta_net, v_heads as u32, n as u32, &args)
    }

    /// Fills `bytes` of GPU memory with zeros.
    pub fn zero(&self, buffer: &Buffer) -> Result<()> {
        let zeros = vec![0f32; buffer.len_bytes() / 4];
        buffer.write_f32(&zeros)
    }
}

/// A kernel argument, by type.
#[derive(Clone, Copy)]
enum Arg {
    Ptr(CuDevicePtr),
    I32(i32),
    F32(f32),
}
