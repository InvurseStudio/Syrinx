//! The CUDA backend: Vireo's own kernels (`kernels/vireo.cu`) on NVIDIA GPUs, run
//! through the driver API directly.
//!
//! No CUDA runtime, cuBLAS or other NVIDIA library is linked or shipped: the engine
//! loads the driver (`nvcuda.dll`) when asked for the GPU, hands it the kernels' PTX,
//! and the driver compiles them for whatever NVIDIA GPU it finds. On a computer
//! without one, [`Gpu::open`] says why and the CPU backend is used instead.
//!
//! All GPU work goes, in order, through one stream, from the thread that opened the
//! [`Gpu`]: the driver's context is made current on that thread. A run of launches can
//! be recorded once as a [`Graph`] and replayed with a single call, which is how a
//! token is generated: launching each of its few hundred kernels one by one costs more
//! than running them.

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

/// The driver, its context on the GPU, the loaded kernels and the stream all work goes
/// through. Shared by everything that holds GPU memory, so it's released last.
struct Context {
    driver: Driver,
    device: CuDevice,
    module: CuModule,
    stream: CuStream,
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

    /// Copies `bytes` from the host to `to`, after the work already queued. The data is
    /// staged before this returns, so `from` can be reused straight away.
    fn upload(&self, to: CuDevicePtr, from: *const c_void, bytes: usize) -> Result<()> {
        let r = unsafe { (self.driver.cuMemcpyHtoDAsync_v2)(to, from, bytes, self.stream) };
        self.check("cuMemcpyHtoDAsync", r)
    }

    /// Copies `bytes` from `from` to the host once the queued work is done.
    fn download(&self, to: *mut c_void, from: CuDevicePtr, bytes: usize) -> Result<()> {
        let r = unsafe { (self.driver.cuMemcpyDtoHAsync_v2)(to, from, bytes, self.stream) };
        self.check("cuMemcpyDtoHAsync", r)?;
        let r = unsafe { (self.driver.cuStreamSynchronize)(self.stream) };
        self.check("cuStreamSynchronize", r)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.cuStreamDestroy_v2)(self.stream);
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

    /// Copies `data` to the start of the buffer, after the GPU work already queued.
    pub fn write_f32(&self, data: &[f32]) -> Result<()> {
        self.write_f32_at(0, data)
    }

    /// Copies `data` to the buffer from f32 number `index` on.
    pub fn write_f32_at(&self, index: usize, data: &[f32]) -> Result<()> {
        assert!((index + data.len()) * 4 <= self.bytes);
        self.context
            .upload(self.f32_at(index), data.as_ptr().cast(), data.len() * 4)
    }

    /// Copies `data` to the start of the buffer.
    pub fn write_i32(&self, data: &[i32]) -> Result<()> {
        assert!(data.len() * 4 <= self.bytes);
        self.context
            .upload(self.ptr, data.as_ptr().cast(), data.len() * 4)
    }

    /// Copies the start of the buffer into `out`, waiting for queued GPU work first.
    pub fn read_f32(&self, out: &mut [f32]) -> Result<()> {
        self.read_f32_at(0, out)
    }

    /// Copies `out.len()` bytes from byte `offset` on into `out`, waiting for queued GPU
    /// work first.
    pub fn read_bytes_at(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        assert!(offset + out.len() <= self.bytes);
        self.context
            .download(out.as_mut_ptr().cast(), self.ptr + offset as u64, out.len())
    }

    /// Copies `data` to the buffer from byte `offset` on, after the GPU work already
    /// queued.
    pub fn write_bytes_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        assert!(offset + data.len() <= self.bytes);
        self.context
            .upload(self.ptr + offset as u64, data.as_ptr().cast(), data.len())
    }

    /// Copies from f32 number `index` on into `out`, waiting for queued GPU work first.
    pub fn read_f32_at(&self, index: usize, out: &mut [f32]) -> Result<()> {
        assert!((index + out.len()) * 4 <= self.bytes);
        self.context
            .download(out.as_mut_ptr().cast(), self.f32_at(index), out.len() * 4)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (self.context.driver.cuMemFree_v2)(self.ptr) };
    }
}

/// A recorded run of kernel launches, replayed with [`Graph::launch`]. It holds the
/// GPU addresses it was recorded with: the buffers they belong to must outlive it.
pub struct Graph {
    context: Arc<Context>,
    graph: CuGraph,
    exec: CuGraphExec,
}

impl Graph {
    /// Queues the whole recorded run.
    pub fn launch(&self) -> Result<()> {
        let r = unsafe { (self.context.driver.cuGraphLaunch)(self.exec, self.context.stream) };
        self.context.check("cuGraphLaunch", r)
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            (self.context.driver.cuGraphExecDestroy)(self.exec);
            (self.context.driver.cuGraphDestroy)(self.graph);
        }
    }
}

/// The tensor-core attention kernels' shapes, as `kernels/vireo.cu` builds them: (head
/// size, rows of 16 tokens per block, warps per row, key/value positions per chunk).
const ATTENTION_MMA: [(u32, u32, u32, u32); 4] = [
    (64, 4, 1, 32),
    (128, 4, 1, 32),
    (256, 4, 1, 32),
    (512, 2, 2, 16),
];

/// Shared memory for a tensor-core attention kernel: the queries, a chunk of keys and a
/// chunk of values in f16 (rows padded by 8), and with warps splitting a head, their
/// parts of S in f32.
fn attention_mma_bytes(d: u32, rows: u32, split: u32, chunk: u32) -> u32 {
    let halves = (16 * rows + 2 * chunk) * (d + 8);
    let parts = if split > 1 {
        rows * split * chunk / 8 * 4 * 32 * 4
    } else {
        0
    };
    halves * 2 + parts
}

#[derive(Clone, Copy)]
struct AttentionMma {
    function: CuFunction,
    shared: u32,
    rows: u32,
    threads: u32,
}

impl Default for AttentionMma {
    fn default() -> Self {
        Self {
            function: std::ptr::null_mut(),
            shared: 0,
            rows: 0,
            threads: 0,
        }
    }
}

struct Kernels {
    matvec: [CuFunction; 7],
    matmul: [CuFunction; 7],
    rms_norm: CuFunction,
    residual_norm: CuFunction,
    qkv_prep: CuFunction,
    attention: CuFunction,
    /// `attention` reading an f16 cache.
    attention_h: CuFunction,
    /// One token's attention split along the context, f32 and f16 caches, and the merge.
    attention_part: [CuFunction; 2],
    /// One token's attention over an f16 cache, a block per key/value head (and set of up
    /// to 4 of its query heads), for head sizes 64, 128, 256 and 512; with how many of its
    /// blocks an SM holds at once.
    attention_part_kv: [(CuFunction, u32); 4],
    attention_combine: CuFunction,
    /// On the tensor cores over an f16 cache, for head sizes 64, 128, 256 and 512:
    /// only on compute capability 8.0 and later.
    attention_mma: Option<[AttentionMma; 4]>,
    /// For head sizes 64, 128, 256 and 512.
    attention_tiled: [CuFunction; 4],
    swiglu: CuFunction,
    geglu: CuFunction,
    add: CuFunction,
    softcap: CuFunction,
    apply_gate: CuFunction,
    conv_silu: CuFunction,
    conv_silu_parallel: CuFunction,
    conv_history: CuFunction,
    delta_net: CuFunction,
    delta_net_wide: CuFunction,
    delta_l2: CuFunction,
    delta_scan: CuFunction,
    /// A prompt's DeltaNet in chunks: with its dynamic shared memory for n = 128.
    delta_chunk_prep: (CuFunction, u32),
    delta_chunk_scan: CuFunction,
    delta_out: CuFunction,
    quantize_q8: CuFunction,
    /// 8-bit activations, in the order of `KINDS_8`.
    matvec8: [CuFunction; 5],
    /// The same for a few tokens at once: by `FEW_SIZES`, then `KINDS_8`.
    matvec8_few: [[CuFunction; 5]; FEW_SIZES.len()],
    /// On the tensor cores: only on compute capability 8.0 and later.
    matmul8: Option<[CuFunction; 5]>,
    /// The same with tiles of 16 and of 32 tokens, for batches of up to that many.
    matmul8_small: Option<[[CuFunction; 5]; 2]>,
}

/// The weight types with 8-bit-activation kernels, in the order of `Kernels::matvec8`.
const KINDS_8: [TensorType; 5] = [
    TensorType::Q4_0,
    TensorType::Q8_0,
    TensorType::Q4_K,
    TensorType::Q5_K,
    TensorType::Q6_K,
];

/// The few-token matrix-vector kernels' sizes (`matvec8_few<T>_*` in vireo.cu): a batch
/// takes the smallest that holds it.
const FEW_SIZES: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];

/// The weight types with kernels, in the order of `Kernels::matvec` and `matmul`.
const KINDS: [TensorType; 7] = [
    TensorType::F32,
    TensorType::F16,
    TensorType::Q4_0,
    TensorType::Q8_0,
    TensorType::Q4_K,
    TensorType::Q5_K,
    TensorType::Q6_K,
];

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
    /// attention_part's partial results: room for MAX_PARTS parts of 512-wide heads, 64
    /// heads. All work goes through one stream, so one is enough for every session.
    attention_parts: Buffer,
    pub info: DeviceInfo,
}

#[cfg(vireo_cuda)]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vireo.ptx"));
/// The same kernels plus the tensor-core ones, for compute capability 8.0 and later.
#[cfg(vireo_cuda)]
const PTX_SM80: &str = include_str!(concat!(env!("OUT_DIR"), "/vireo_sm80.ptx"));

/// The number of threads the matrix-vector kernels use per block: 4 warps, 4 rows.
const ROWS_PER_BLOCK: u32 = 4;
/// The matrix-matrix kernels' tile: rows of weights by tokens.
const TILE_ROWS: u32 = 128;
/// The most parts one token's attention is split into (`attention_part`, f32 cache).
const MAX_PARTS: u32 = 16;
/// The same for `attention_part_kv` (f16 cache), whose blocks each take a key/value head:
/// fewer blocks a part, so more parts.
const MAX_PARTS_KV: u32 = 64;
const TILE_TOKENS: u32 = 64;

/// A parameter for the kernels, which take raw addresses: 0 stands for "none".
fn or_null(buffer: Option<&Buffer>) -> CuDevicePtr {
    buffer.map_or(0, Buffer::ptr)
}

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
            let tensor_cores = info.compute_capability.0 >= 8;
            let ptx = CString::new(if tensor_cores { PTX_SM80 } else { PTX })
                .expect("PTX has no NUL bytes");
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
            let mut stream = std::ptr::null_mut();
            let created = (driver.cuStreamCreate)(&mut stream, STREAM_NON_BLOCKING);
            if created != CUDA_SUCCESS {
                (driver.cuModuleUnload)(module);
                (driver.cuDevicePrimaryCtxRelease_v2)(device);
                return Err(Error::Driver {
                    call: "cuStreamCreate",
                    code: created,
                    name: format!("error {created}"),
                });
            }
            let context = Arc::new(Context {
                driver,
                device,
                module,
                stream,
            });

            let function = |name: &str| -> Result<CuFunction> {
                let c_name = CString::new(name).unwrap();
                let mut f = std::ptr::null_mut();
                let r =
                    (context.driver.cuModuleGetFunction)(&mut f, context.module, c_name.as_ptr());
                context.check("cuModuleGetFunction", r).map(|_| f)
            };
            let per_kind = |prefix: &str| -> Result<[CuFunction; 7]> {
                let mut out = [std::ptr::null_mut(); 7];
                for (slot, kind) in out.iter_mut().zip(KINDS) {
                    *slot = function(&format!("{prefix}_{}", kind.name().to_lowercase()))?;
                }
                Ok(out)
            };
            let kernels = Kernels {
                matvec: per_kind("matvec")?,
                matmul: per_kind("matmul")?,
                rms_norm: function("rms_norm")?,
                residual_norm: function("residual_norm")?,
                qkv_prep: function("qkv_prep")?,
                attention: function("attention")?,
                attention_h: function("attention_h")?,
                attention_part: [
                    function("attention_part_f32")?,
                    function("attention_part_f16")?,
                ],
                attention_part_kv: {
                    let mut out = [(std::ptr::null_mut(), 1); 4];
                    for (slot, hd) in out.iter_mut().zip([64, 128, 256, 512]) {
                        let f = function(&format!("attention_part_kv_{hd}"))?;
                        let mut blocks = 0;
                        let r = (context.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor)(
                            &mut blocks,
                            f,
                            128,
                            0,
                        );
                        context.check("cuOccupancyMaxActiveBlocksPerMultiprocessor", r)?;
                        *slot = (f, blocks.max(1) as u32);
                    }
                    out
                },
                attention_combine: function("attention_combine")?,
                attention_mma: if tensor_cores {
                    let mut out = [AttentionMma::default(); 4];
                    for (slot, &(d, rows, split, chunk)) in out.iter_mut().zip(&ATTENTION_MMA) {
                        let f = function(&format!("attention_mma_{d}"))?;
                        let bytes = attention_mma_bytes(d, rows, split, chunk);
                        let r = (context.driver.cuFuncSetAttribute)(
                            f,
                            FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes as i32,
                        );
                        context.check("cuFuncSetAttribute", r)?;
                        *slot = AttentionMma {
                            function: f,
                            shared: bytes,
                            rows: 16 * rows,
                            threads: 32 * rows * split,
                        };
                    }
                    Some(out)
                } else {
                    None
                },
                attention_tiled: [
                    function("attention_tiled_64")?,
                    function("attention_tiled_128")?,
                    function("attention_tiled_256")?,
                    function("attention_tiled_512")?,
                ],
                swiglu: function("swiglu")?,
                geglu: function("geglu")?,
                add: function("add")?,
                softcap: function("softcap")?,
                apply_gate: function("apply_gate")?,
                conv_silu: function("conv_silu")?,
                conv_silu_parallel: function("conv_silu_parallel")?,
                conv_history: function("conv_history")?,
                delta_net: function("delta_net")?,
                delta_net_wide: function("delta_net_wide")?,
                delta_l2: function("delta_l2")?,
                delta_scan: function("delta_scan")?,
                delta_chunk_prep: {
                    let f = function("delta_chunk_prep")?;
                    let bytes = delta_chunk_prep_bytes(128);
                    let r = (context.driver.cuFuncSetAttribute)(
                        f,
                        FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        bytes as i32,
                    );
                    context.check("cuFuncSetAttribute", r)?;
                    (f, bytes)
                },
                delta_chunk_scan: function("delta_chunk_scan")?,
                delta_out: function("delta_out")?,
                quantize_q8: function("quantize_q8")?,
                matvec8: {
                    let mut out = [std::ptr::null_mut(); 5];
                    for (slot, kind) in out.iter_mut().zip(KINDS_8) {
                        *slot = function(&format!("matvec8_{}", kind.name().to_lowercase()))?;
                    }
                    out
                },
                matvec8_few: {
                    let mut out = [[std::ptr::null_mut(); 5]; FEW_SIZES.len()];
                    for (row, size) in out.iter_mut().zip(FEW_SIZES) {
                        for (slot, kind) in row.iter_mut().zip(KINDS_8) {
                            let name =
                                format!("matvec8_few{size}_{}", kind.name().to_lowercase());
                            *slot = function(&name)?;
                        }
                    }
                    out
                },
                matmul8: if tensor_cores {
                    let mut out = [std::ptr::null_mut(); 5];
                    for (slot, kind) in out.iter_mut().zip(KINDS_8) {
                        *slot = function(&format!("matmul8_{}", kind.name().to_lowercase()))?;
                    }
                    Some(out)
                } else {
                    None
                },
                matmul8_small: if tensor_cores {
                    let mut out = [[std::ptr::null_mut(); 5]; 2];
                    for (row, tile) in out.iter_mut().zip([16, 32]) {
                        for (slot, kind) in row.iter_mut().zip(KINDS_8) {
                            let name = format!("matmul8_t{tile}_{}", kind.name().to_lowercase());
                            *slot = function(&name)?;
                        }
                    }
                    Some(out)
                } else {
                    None
                },
            };
            // Twice what one token's 64 heads of 512 in the most parts need: a few tokens of
            // any model here fit with all their parts.
            let parts_bytes = 2 * 64 * MAX_PARTS.max(MAX_PARTS_KV) as usize * (2 + 512) * 4;
            let mut ptr = 0;
            let r = (context.driver.cuMemAlloc_v2)(&mut ptr, parts_bytes);
            context.check("cuMemAlloc", r)?;
            let attention_parts = Buffer {
                context: Arc::clone(&context),
                ptr,
                bytes: parts_bytes,
            };
            Ok(Self {
                context,
                kernels,
                attention_parts,
                info,
            })
        }
    }

    /// Whether a weight of this type can be multiplied on the GPU.
    pub fn supports(kind: TensorType) -> bool {
        KINDS.contains(&kind)
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
        self.context
            .upload(buffer.ptr, data.as_ptr().cast(), data.len())?;
        Ok(buffer)
    }

    /// A weight matrix of `rows` × `cols` numbers of type `kind` (GGUF's layout) in GPU
    /// memory, in the layout the kernels read: as it is, except Q4_0, whose 18-byte blocks
    /// are split into each row's 4-bit numbers (16-byte aligned) and then its scales, and
    /// Q6_K, whose 210-byte blocks are padded to 224 (16-byte aligned).
    pub fn upload_weights(
        &self,
        kind: TensorType,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Buffer> {
        if kind == TensorType::Q6_K {
            // Q6_K: each 210-byte block padded to 224, so blocks start 16-byte aligned.
            let mut out = vec![0u8; data.len() / 210 * 224];
            for (src, dst) in data
                .as_chunks::<210>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<224>().0)
            {
                dst[..210].copy_from_slice(src);
            }
            return self.upload(&out);
        }
        if kind != TensorType::Q4_0 {
            return self.upload(data);
        }
        let blocks = cols / 32;
        let row_bytes = blocks * 16 + (blocks * 2).next_multiple_of(16);
        let mut out = vec![0u8; rows * row_bytes];
        for (src, dst) in data
            .chunks_exact(blocks * 18)
            .zip(out.chunks_exact_mut(row_bytes))
        {
            let (numbers, scales) = dst.split_at_mut(blocks * 16);
            for (b, block) in src.as_chunks::<18>().0.iter().enumerate() {
                scales[b * 2..b * 2 + 2].copy_from_slice(&block[..2]);
                numbers[b * 16..b * 16 + 16].copy_from_slice(&block[2..]);
            }
        }
        self.upload(&out)
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<Buffer> {
        let buffer = self.alloc(data.len() * 4)?;
        buffer.write_f32(data)?;
        Ok(buffer)
    }

    /// Copies `bytes` within GPU memory, after the work already queued.
    pub fn copy(&self, to: CuDevicePtr, from: CuDevicePtr, bytes: usize) -> Result<()> {
        let c = &self.context;
        let r = unsafe { (c.driver.cuMemcpyDtoDAsync_v2)(to, from, bytes, c.stream) };
        c.check("cuMemcpyDtoDAsync", r)
    }

    /// Waits for all queued GPU work, and reports any error a kernel hit.
    pub fn synchronize(&self) -> Result<()> {
        let r = unsafe { (self.context.driver.cuStreamSynchronize)(self.context.stream) };
        self.context.check("cuStreamSynchronize", r)
    }

    /// From here until [`Gpu::finish_recording`], launches are recorded, not run. Only
    /// launches: copying to or from the GPU meanwhile is an error.
    pub fn start_recording(&self) -> Result<()> {
        let c = &self.context;
        let r = unsafe {
            (c.driver.cuStreamBeginCapture_v2)(c.stream, STREAM_CAPTURE_MODE_THREAD_LOCAL)
        };
        c.check("cuStreamBeginCapture", r)
    }

    /// The launches since [`Gpu::start_recording`], as a [`Graph`]. Always call it after
    /// starting, even when queuing the launches failed, to get the stream back.
    pub fn finish_recording(&self) -> Result<Graph> {
        let c = &self.context;
        let mut graph = std::ptr::null_mut();
        let r = unsafe { (c.driver.cuStreamEndCapture)(c.stream, &mut graph) };
        c.check("cuStreamEndCapture", r)?;
        let mut exec = std::ptr::null_mut();
        let r = unsafe { (c.driver.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) };
        if let Err(e) = c.check("cuGraphInstantiate", r) {
            unsafe { (c.driver.cuGraphDestroy)(graph) };
            return Err(e);
        }
        Ok(Graph {
            context: Arc::clone(c),
            graph,
            exec,
        })
    }

    /// Queues `kernel` on `grid` blocks of `block` threads, with `args` in the kernel's
    /// parameter order. The driver copies the arguments before returning.
    fn run(&self, kernel: CuFunction, grid: (u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        self.run_3d(kernel, (grid.0, grid.1, 1), block, 0, args)
    }

    /// As `run`, with a third grid dimension and `shared` bytes of dynamic shared memory.
    fn run_3d(
        &self,
        kernel: CuFunction,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> Result<()> {
        let mut values = [0u64; 24];
        let mut params = [std::ptr::null_mut::<c_void>(); 24];
        assert!(args.len() <= values.len());
        for (i, a) in args.iter().enumerate() {
            let slot = &mut values[i];
            match *a {
                Arg::Ptr(v) => *slot = v,
                Arg::I32(v) => unsafe { *(slot as *mut u64).cast::<i32>() = v },
                Arg::F32(v) => unsafe { *(slot as *mut u64).cast::<f32>() = v },
            }
            params[i] = (slot as *mut u64).cast();
        }
        let r = unsafe {
            (self.context.driver.cuLaunchKernel)(
                kernel,
                grid.0,
                grid.1,
                grid.2,
                block,
                1,
                1,
                shared,
                self.context.stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.context.check("cuLaunchKernel", r)
    }

    /// One thread per number, 256 to a block.
    fn run_per_number(&self, kernel: CuFunction, n: usize, args: &[Arg]) -> Result<()> {
        self.run(kernel, ((n as u32).div_ceil(256), 1), 256, args)
    }

    fn kind_index(kind: TensorType) -> Result<usize> {
        KINDS
            .iter()
            .position(|&k| k == kind)
            .ok_or(Error::Unsupported(kind))
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
        let kernel = self.kernels.matvec[Self::kind_index(kind)?];
        // The quantized kernels give each warp 2 rows (MV_ROWS in vireo.cu).
        let rows_per_warp = if matches!(kind, TensorType::F32 | TensorType::F16) {
            1
        } else {
            2
        };
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * rows_per_warp);
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
        ];
        self.run(kernel, (grid, 1), ROWS_PER_BLOCK * 32, &args)
    }

    /// `y = W x` for each of `tokens` rows of `x` (`cols` numbers each) into rows of `y`
    /// (`rows` each): one pass over the weights for all of them.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        tokens: usize,
    ) -> Result<()> {
        if tokens == 1 {
            return self.matvec(kind, w, x, y, rows, cols);
        }
        if !cols.is_multiple_of(32) {
            // The tiles take 32 columns at a time; no model here has such a matrix.
            for t in 0..tokens {
                let (xt, yt) = (x + (t * cols * 4) as u64, y + (t * rows * 4) as u64);
                self.matvec(kind, w, xt, yt, rows, cols)?;
            }
            return Ok(());
        }
        let kernel = self.kernels.matmul[Self::kind_index(kind)?];
        let grid = (
            (rows as u32).div_ceil(TILE_ROWS),
            (tokens as u32).div_ceil(TILE_TOKENS),
        );
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::I32(tokens as i32),
        ];
        self.run(kernel, grid, 256, &args)
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
        self.run(self.kernels.rms_norm, (rows as u32, 1), 256, &args)
    }

    /// The residual stream between steps, for `tokens` rows of `n`: `x = (x + p') ×
    /// scale`, where p' is `p`, or RMSNorm(p) × `post` when given (and nothing without
    /// `p`); then `h = RMSNorm(x) × next` when `next` is given.
    #[allow(clippy::too_many_arguments)]
    pub fn residual_norm(
        &self,
        x: CuDevicePtr,
        p: Option<CuDevicePtr>,
        post: Option<&Buffer>,
        scale: f32,
        next: Option<&Buffer>,
        h: CuDevicePtr,
        tokens: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(x),
            Arg::Ptr(p.unwrap_or(0)),
            Arg::Ptr(or_null(post)),
            Arg::F32(scale),
            Arg::Ptr(or_null(next)),
            Arg::Ptr(h),
            Arg::I32(n as i32),
            Arg::F32(eps),
        ];
        self.run(self.kernels.residual_norm, (tokens as u32, 1), 256, &args)
    }

    /// Gets `tokens` tokens' queries, keys and values ready for attention: see
    /// [`QkvPrep`].
    pub fn qkv_prep(&self, p: &QkvPrep, tokens: usize) -> Result<()> {
        assert!(p.head_dim <= 512, "head size {} is over 512", p.head_dim);
        assert_eq!(p.ring, 0, "no rings on CUDA");
        let kv = p.cache.is_some();
        let (k_cache, v_cache) = p.cache.unwrap_or((0, 0));
        let args = [
            Arg::Ptr(p.q),
            Arg::I32(p.q_stride as i32),
            Arg::I32(p.q_head_stride as i32),
            Arg::Ptr(p.q_norm.unwrap_or(0)),
            Arg::Ptr(p.q_out),
            Arg::Ptr(p.k),
            Arg::Ptr(p.k_norm.unwrap_or(0)),
            Arg::Ptr(p.v),
            Arg::I32(p.v_norm as i32),
            Arg::Ptr(k_cache),
            Arg::Ptr(v_cache),
            Arg::I32(p.heads as i32),
            Arg::I32(p.kv_heads as i32),
            Arg::I32(p.head_dim as i32),
            Arg::I32(p.rope_dims as i32),
            Arg::F32(p.rope_base),
            Arg::Ptr(p.rope_factors.unwrap_or(0)),
            Arg::I32(p.neox as i32),
            Arg::Ptr(p.pos),
            Arg::F32(p.eps),
            Arg::I32(p.half_cache as i32),
            Arg::I32(tokens as i32),
        ];
        let slots = p.heads + if kv { 2 * p.kv_heads } else { 0 };
        // A thread per number of a head, up to 256 (Llama's 64-wide heads left 3 in 4
        // idle). The sums come out the same: block_sum counts missing warps as zeros.
        // A block per head and run of 8 tokens (QKV_TOKENS in the kernel).
        let threads = (p.head_dim.div_ceil(32) * 32).min(256) as u32;
        self.run(
            self.kernels.qkv_prep,
            (slots as u32, (tokens as u32).div_ceil(8)),
            threads,
            &args,
        )
    }

    /// [`Gpu::attention`], for the Metal backend's rings, which this one doesn't have.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_ring(
        &self,
        half_cache: bool,
        q: CuDevicePtr,
        keys: CuDevicePtr,
        values: CuDevicePtr,
        out: CuDevicePtr,
        pos: CuDevicePtr,
        window: usize,
        ring: usize,
        heads: usize,
        tokens: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        scale: f32,
    ) -> Result<()> {
        assert_eq!(ring, 0, "no rings on CUDA");
        self.attention(
            half_cache, q, keys, values, out, pos, window, heads, tokens, head_dim, kv_dim, group,
            scale,
        )
    }

    /// Causal attention for `tokens` tokens of `heads` query heads, token t at position
    /// `*pos + t` (read on the GPU), over the cache up to its own position; `window`
    /// positions at most (0 for no limit). `q` and `out` hold a row of heads × head_dim
    /// per token. With `half_cache` the cache holds f16: a prompt then goes to the tensor
    /// cores where there are some.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        half_cache: bool,
        q: CuDevicePtr,
        keys: CuDevicePtr,
        values: CuDevicePtr,
        out: CuDevicePtr,
        pos: CuDevicePtr,
        window: usize,
        heads: usize,
        tokens: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        scale: f32,
    ) -> Result<()> {
        assert!(
            head_dim <= 512 && head_dim.is_multiple_of(4),
            "head size {head_dim}"
        );
        let args = [
            Arg::Ptr(q),
            Arg::Ptr(keys),
            Arg::Ptr(values),
            Arg::Ptr(out),
            Arg::Ptr(pos),
            Arg::I32(window as i32),
            Arg::I32(head_dim as i32),
            Arg::I32(kv_dim as i32),
            Arg::I32(group as i32),
            Arg::F32(scale),
        ];
        let kv_few = half_cache && matches!(head_dim, 64 | 128 | 256 | 512);
        if (tokens == 1 || (kv_few && tokens <= Self::FEW_TOKENS)) && heads <= 64 {
            // One token: split each head's context into parts so every SM has work (about
            // four blocks per SM), then merge them. A few tokens (over an f16 cache): the
            // same, a grid layer each.
            let sms = self.info.multiprocessors.max(1) as u32;
            let kv_kernel = match head_dim {
                64 => Some(0),
                128 => Some(1),
                256 => Some(2),
                512 => Some(3),
                _ => None,
            };
            if let (true, Some(i)) = (half_cache, kv_kernel) {
                // A block per key/value head and set of up to 4 of its query heads, so each
                // key and value is read once a set. As many parts as fill the GPU in one
                // wave (the blocks hold many registers: 2 an SM at head size 512), and no
                // more than a sliding window fills 32 positions of each.
                let (kernel, per_sm) = self.kernels.attention_part_kv[i];
                let blocks = (heads / group * group.div_ceil(4)) as u32;
                let layers = tokens as u32;
                let mut parts = (per_sm * sms / blocks).clamp(1, MAX_PARTS_KV);
                if window > 0 {
                    parts = parts.min((window as u32).div_ceil(32)).max(1);
                }
                // A few tokens split their contexts as one does, so each gets exactly the
                // numbers it would alone (the GPU is as full: there's a token's work each).
                // Fewer parts only if their partial rows wouldn't fit (no model here).
                let row_bytes = (2 + head_dim) * 4;
                let room = self.attention_parts.len_bytes() / (tokens * heads * row_bytes);
                parts = parts.min(room as u32).max(1);
                let part_args = [
                    Arg::Ptr(q),
                    Arg::Ptr(keys),
                    Arg::Ptr(values),
                    Arg::Ptr(self.attention_parts.ptr()),
                    Arg::Ptr(pos),
                    Arg::I32(window as i32),
                    Arg::I32(kv_dim as i32),
                    Arg::I32(group as i32),
                    Arg::F32(scale),
                    Arg::I32(heads as i32),
                ];
                self.run_3d(kernel, (blocks, parts, layers), 128, 0, &part_args)?;
                let combine = [
                    Arg::Ptr(self.attention_parts.ptr()),
                    Arg::Ptr(out),
                    Arg::I32(parts as i32),
                    Arg::I32(head_dim as i32),
                ];
                // A block per token and head: their rows follow one another.
                return self.run(
                    self.kernels.attention_combine,
                    ((heads * tokens) as u32, 1),
                    128,
                    &combine,
                );
            }
            let parts = (4 * sms).div_ceil(heads as u32).clamp(1, MAX_PARTS);
            let part_args = [
                Arg::Ptr(q),
                Arg::Ptr(keys),
                Arg::Ptr(values),
                Arg::Ptr(self.attention_parts.ptr()),
                Arg::Ptr(pos),
                Arg::I32(window as i32),
                Arg::I32(head_dim as i32),
                Arg::I32(kv_dim as i32),
                Arg::I32(group as i32),
                Arg::F32(scale),
            ];
            let kernel = self.kernels.attention_part[usize::from(half_cache)];
            self.run(kernel, (heads as u32, parts), 128, &part_args)?;
            let combine = [
                Arg::Ptr(self.attention_parts.ptr()),
                Arg::Ptr(out),
                Arg::I32(parts as i32),
                Arg::I32(head_dim as i32),
            ];
            return self.run(
                self.kernels.attention_combine,
                (heads as u32, 1),
                128,
                &combine,
            );
        }
        if half_cache {
            let wide = match head_dim {
                64 => Some(0),
                128 => Some(1),
                256 => Some(2),
                512 => Some(3),
                _ => None,
            };
            if let (true, Some(kernels), Some(i)) = (tokens > 1, self.kernels.attention_mma, wide) {
                let k = kernels[i];
                let mut args = args.to_vec();
                args.push(Arg::I32(tokens as i32));
                let grid = (heads as u32, (tokens as u32).div_ceil(k.rows), 1);
                return self.run_3d(k.function, grid, k.threads, k.shared, &args);
            }
            return self.run(
                self.kernels.attention_h,
                (heads as u32, tokens as u32),
                128,
                &args,
            );
        }
        let tiled = match head_dim {
            64 => Some((0, 8)),
            128 => Some((1, 8)),
            256 => Some((2, 16)),
            512 => Some((3, 32)),
            _ => None,
        };
        if let (true, Some((kernel, lanes))) = (tokens > 1, tiled) {
            // A prompt: a block of tokens shares each key and value it reads.
            let mut args = args.to_vec();
            args.push(Arg::I32(tokens as i32));
            let blocks = (tokens as u32).div_ceil(256 / lanes);
            return self.run(
                self.kernels.attention_tiled[kernel],
                (heads as u32, blocks),
                256,
                &args,
            );
        }
        self.run(
            self.kernels.attention,
            (heads as u32, tokens as u32),
            128,
            &args,
        )
    }

    /// `gate = SiLU(gate) × up` over `tokens` rows of `n`; up's rows are `up_stride` apart.
    pub fn swiglu(
        &self,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        n: usize,
        up_stride: usize,
        tokens: usize,
    ) -> Result<()> {
        self.gated(self.kernels.swiglu, gate, up, n, up_stride, tokens)
    }

    /// `gate = GELU(gate) × up` (tanh approximation), laid out as in [`Gpu::swiglu`].
    pub fn geglu(
        &self,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        n: usize,
        up_stride: usize,
        tokens: usize,
    ) -> Result<()> {
        self.gated(self.kernels.geglu, gate, up, n, up_stride, tokens)
    }

    fn gated(
        &self,
        kernel: CuFunction,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        n: usize,
        up_stride: usize,
        tokens: usize,
    ) -> Result<()> {
        let total = n * tokens;
        let args = [
            Arg::Ptr(gate),
            Arg::Ptr(up),
            Arg::I32(n as i32),
            Arg::I32(up_stride as i32),
            Arg::I32(total as i32),
        ];
        self.run_per_number(kernel, total, &args)
    }

    /// `x += y`
    pub fn add(&self, x: CuDevicePtr, y: CuDevicePtr, n: usize) -> Result<()> {
        self.run_per_number(
            self.kernels.add,
            n,
            &[Arg::Ptr(x), Arg::Ptr(y), Arg::I32(n as i32)],
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

    /// `out ×= sigmoid(gate)`, the gates being the second halves of `[query | gate]`,
    /// for `heads` heads (all tokens' heads, one after another).
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

    /// The causal convolution and SiLU of Gated DeltaNet for `tokens` tokens in order,
    /// moving the history along.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_silu(
        &self,
        current: CuDevicePtr,
        history: CuDevicePtr,
        weights: CuDevicePtr,
        out: CuDevicePtr,
        channels: usize,
        kernel: usize,
        tokens: usize,
    ) -> Result<()> {
        assert!((2..=8).contains(&kernel), "convolution of {kernel}");
        if tokens > Self::FEW_TOKENS {
            // A prompt: every token and channel at once, then the history moved along. (A
            // few tokens walk through the one-token kernel: one launch, not two.)
            let conv = [
                Arg::Ptr(current),
                Arg::Ptr(history),
                Arg::Ptr(weights),
                Arg::Ptr(out),
                Arg::I32(channels as i32),
                Arg::I32(kernel as i32),
                Arg::I32(tokens as i32),
            ];
            // A thread per channel and run of 16 tokens (CONV_TOKENS in the kernel).
            let grid = (
                (channels as u32).div_ceil(256),
                (tokens as u32).div_ceil(16),
            );
            self.run(self.kernels.conv_silu_parallel, grid, 256, &conv)?;
            let shift = [
                Arg::Ptr(current),
                Arg::Ptr(history),
                Arg::I32(channels as i32),
                Arg::I32(kernel as i32),
                Arg::I32(tokens as i32),
            ];
            return self.run_per_number(self.kernels.conv_history, channels, &shift);
        }
        let args = [
            Arg::Ptr(current),
            Arg::Ptr(history),
            Arg::Ptr(weights),
            Arg::Ptr(out),
            Arg::I32(channels as i32),
            Arg::I32(kernel as i32),
            Arg::I32(tokens as i32),
        ];
        self.run_per_number(self.kernels.conv_silu, channels, &args)
    }

    /// Gated DeltaNet for `tokens` tokens in order, `v_heads` heads with states of
    /// `n` × `n` (n ≤ 256): see [`DeltaNet`].
    /// Scratch for a prompt's DeltaNet in chunks ([`DeltaNet::scratch`]), in f32s, for up
    /// to `tokens` tokens at a time.
    pub fn delta_scratch_floats(tokens: usize, v_heads: usize, n: usize) -> usize {
        tokens.div_ceil(DELTA_CHUNK)
            * v_heads
            * (4 * DELTA_CHUNK * n + DELTA_CHUNK * DELTA_CHUNK + 4)
    }

    pub fn delta_net(&self, d: &DeltaNet, tokens: usize) -> Result<()> {
        let n = d.n;
        // Only the one-token kernel with the state in registers keeps its steps.
        assert!(
            d.steps == 0 || (tokens <= Self::FEW_TOKENS && n <= 128 && n.is_multiple_of(8)),
            "DeltaNet steps are kept for up to {} tokens of states up to 128",
            Self::FEW_TOKENS
        );
        assert!(
            n <= 256 && n.is_multiple_of(32),
            "state size {n}: the kernel takes multiples of 32 up to 256"
        );
        let args = [
            Arg::Ptr(d.conv),
            Arg::I32(d.channels as i32),
            Arg::I32(d.key_dim as i32),
            Arg::Ptr(d.beta_raw),
            Arg::Ptr(d.alpha_raw),
            Arg::Ptr(d.dt_bias),
            Arg::Ptr(d.a),
            Arg::Ptr(d.state),
            Arg::Ptr(d.z),
            Arg::Ptr(d.norm),
            Arg::Ptr(d.out),
            Arg::I32(n as i32),
            Arg::I32(d.k_heads as i32),
            Arg::F32(d.eps),
            Arg::I32(tokens as i32),
        ];
        // A few tokens step through the one-token kernel (the state in registers
        // throughout): a fraction of the chunked kernels' time, and exactly one token at a
        // time's numbers.
        let prompt = tokens > Self::FEW_TOKENS;
        if prompt && n <= 128 && n.is_multiple_of(32) && d.scratch != 0 {
            // A prompt in chunks of 64 tokens (delta_chunk_prep and delta_chunk_scan in
            // vireo.cu), between the norms for every token at once.
            let rows = (tokens * 2 * d.k_heads) as u32;
            let l2 = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            self.run(self.kernels.delta_l2, (rows, 1), n as u32, &l2)?;
            let prep = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.key_dim as i32),
                Arg::Ptr(d.beta_raw),
                Arg::Ptr(d.alpha_raw),
                Arg::Ptr(d.dt_bias),
                Arg::Ptr(d.a),
                Arg::Ptr(d.scratch),
                Arg::I32(n as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(tokens as i32),
            ];
            let chunks = tokens.div_ceil(DELTA_CHUNK) as u32;
            let (prep_kernel, _) = self.kernels.delta_chunk_prep;
            self.run_3d(
                prep_kernel,
                (d.v_heads as u32, chunks, 1),
                256,
                delta_chunk_prep_bytes(n),
                &prep,
            )?;
            let scan = [
                Arg::Ptr(d.scratch),
                Arg::Ptr(d.state),
                Arg::Ptr(d.out),
                Arg::I32(n as i32),
                Arg::I32(tokens as i32),
            ];
            // 16 columns of the state a block (DSC_COLS in vireo.cu).
            let grid = (d.v_heads as u32, (n / 16) as u32);
            self.run(self.kernels.delta_chunk_scan, grid, 64, &scan)?;
            let out = [
                Arg::Ptr(d.out),
                Arg::Ptr(d.z),
                Arg::Ptr(d.norm),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            let rows = (tokens * d.v_heads) as u32;
            return self.run(self.kernels.delta_out, (rows, 1), n as u32, &out);
        }
        if prompt && n <= 128 && n.is_multiple_of(32) {
            // A prompt: the norms for every token at once, the state update split over
            // blocks of 16 columns, then the output norm for every token at once.
            let rows = (tokens * 2 * d.k_heads) as u32;
            let l2 = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            self.run(self.kernels.delta_l2, (rows, 1), n as u32, &l2)?;
            let scan = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.key_dim as i32),
                Arg::Ptr(d.beta_raw),
                Arg::Ptr(d.alpha_raw),
                Arg::Ptr(d.dt_bias),
                Arg::Ptr(d.a),
                Arg::Ptr(d.state),
                Arg::Ptr(d.out),
                Arg::I32(n as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(tokens as i32),
            ];
            // 16 columns a block (DS_COLS in vireo.cu): 64 threads.
            let grid = (d.v_heads as u32, (n / 16) as u32);
            self.run(self.kernels.delta_scan, grid, 64, &scan)?;
            let out = [
                Arg::Ptr(d.out),
                Arg::Ptr(d.z),
                Arg::Ptr(d.norm),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            let rows = (tokens * d.v_heads) as u32;
            return self.run(self.kernels.delta_out, (rows, 1), n as u32, &out);
        }
        if n <= 128 && n.is_multiple_of(8) {
            // The state in registers: 4 threads to a column.
            let mut with_steps = args.to_vec();
            with_steps.push(Arg::Ptr(d.steps));
            return self.run(
                self.kernels.delta_net,
                (d.v_heads as u32, 1),
                4 * n as u32,
                &with_steps,
            );
        }
        self.run(
            self.kernels.delta_net_wide,
            (d.v_heads as u32, 1),
            n as u32,
            &args,
        )
    }

    /// The most tokens the few-token kernels take at once.
    pub const FEW_TOKENS: usize = FEW_SIZES[FEW_SIZES.len() - 1];

    /// Whether a weight of this type has kernels for 8-bit activations.
    pub fn supports_8bit(kind: TensorType) -> bool {
        KINDS_8.contains(&kind)
    }

    /// Room for `tokens` rows of up to `cols` numbers rounded to 8 bits.
    pub fn alloc_q8(&self, tokens: usize, cols: usize) -> Result<Q8> {
        Ok(Q8 {
            q: self.alloc(tokens * cols)?,
            scales: self.alloc(tokens * cols / 32 * 4)?,
            sums: self.alloc(tokens * cols / 16 * 4)?,
        })
    }

    /// Rounds `tokens` rows of `cols` numbers (a multiple of 32) at `x` to 8 bits, into `q8`.
    pub fn quantize(&self, x: CuDevicePtr, q8: &Q8, tokens: usize, cols: usize) -> Result<()> {
        assert!(
            cols.is_multiple_of(32),
            "rows of {cols} aren't whole blocks of 32"
        );
        assert!(
            tokens * cols <= q8.q.len_bytes(),
            "the 8-bit buffer is too small"
        );
        let blocks = tokens * cols / 32;
        let args = [
            Arg::Ptr(x),
            Arg::Ptr(q8.q.ptr()),
            Arg::Ptr(q8.scales.ptr()),
            Arg::Ptr(q8.sums.ptr()),
            Arg::I32(blocks as i32),
        ];
        // A warp per block of 32, 8 to a thread block.
        self.run(
            self.kernels.quantize_q8,
            ((blocks as u32).div_ceil(8), 1),
            256,
            &args,
        )
    }

    /// `y = W x` for each of `tokens` rows of `x`, already rounded to 8 bits into `q8`
    /// (see [`Gpu::quantize`]). Only for the types [`Gpu::supports_8bit`] names.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_8bit(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        q8: &Q8,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        tokens: usize,
    ) -> Result<()> {
        let index = KINDS_8
            .iter()
            .position(|&k| k == kind)
            .ok_or(Error::Unsupported(kind))?;
        if tokens == 1 {
            let args = [
                Arg::Ptr(w),
                Arg::Ptr(q8.q.ptr()),
                Arg::Ptr(q8.scales.ptr()),
                Arg::Ptr(y),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
            ];
            let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * 2);
            return self.run(
                self.kernels.matvec8[index],
                (grid, 1),
                ROWS_PER_BLOCK * 32,
                &args,
            );
        }
        if let Some(size) = FEW_SIZES.iter().position(|&s| s >= tokens) {
            // A few tokens: the weights read once, as for one, and multiplied by each.
            let args = [
                Arg::Ptr(w),
                Arg::Ptr(q8.q.ptr()),
                Arg::Ptr(q8.scales.ptr()),
                Arg::Ptr(y),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                Arg::I32(tokens as i32),
            ];
            let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * 2);
            return self.run(
                self.kernels.matvec8_few[size][index],
                (grid, 1),
                ROWS_PER_BLOCK * 32,
                &args,
            );
        }
        match self.kernels.matmul8 {
            Some(kernels) => {
                let args = [
                    Arg::Ptr(w),
                    Arg::Ptr(q8.q.ptr()),
                    Arg::Ptr(q8.scales.ptr()),
                    Arg::Ptr(q8.sums.ptr()),
                    Arg::Ptr(y),
                    Arg::I32(rows as i32),
                    Arg::I32(cols as i32),
                    Arg::I32(tokens as i32),
                ];
                // A short batch takes a tile of 16 or 32 tokens: a tile of 64 costs the
                // same for 9 tokens as for 64.
                let (kernel, tile) = match (self.kernels.matmul8_small, tokens) {
                    (Some(small), 0..=16) => (small[0][index], 16),
                    (Some(small), 17..=32) => (small[1][index], 32),
                    _ => (kernels[index], TILE_TOKENS),
                };
                let grid = (
                    (rows as u32).div_ceil(TILE_ROWS),
                    (tokens as u32).div_ceil(tile),
                );
                self.run(kernel, grid, 256, &args)
            }
            None => {
                // No tensor cores: the 8-bit matrix-vector kernel for each token.
                for t in 0..tokens {
                    let args = [
                        Arg::Ptr(w),
                        Arg::Ptr(q8.q.ptr() + (t * cols) as u64),
                        Arg::Ptr(q8.scales.ptr() + (t * cols / 32 * 4) as u64),
                        Arg::Ptr(y + (t * rows * 4) as u64),
                        Arg::I32(rows as i32),
                        Arg::I32(cols as i32),
                    ];
                    let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * 2);
                    self.run(
                        self.kernels.matvec8[index],
                        (grid, 1),
                        ROWS_PER_BLOCK * 32,
                        &args,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Fills a buffer with zeros.
    pub fn zero(&self, buffer: &Buffer) -> Result<()> {
        let zeros = vec![0f32; buffer.len_bytes() / 4];
        buffer.write_f32(&zeros)
    }
}

/// What [`Gpu::qkv_prep`] works on. Token t's query head h is at `q + (t × q_stride +
/// h × q_head_stride)` floats; its keys and values at `k`/`v + t × kv_heads × head_dim`.
/// Queries go to `q_out` (a row of heads × head_dim per token); keys and values into
/// `cache` at their positions (`*pos + t`), when there's a cache: layers that share an
/// earlier layer's have none.
pub struct QkvPrep {
    pub q: CuDevicePtr,
    pub q_stride: usize,
    pub q_head_stride: usize,
    pub q_norm: Option<CuDevicePtr>,
    pub q_out: CuDevicePtr,
    pub k: CuDevicePtr,
    pub k_norm: Option<CuDevicePtr>,
    pub v: CuDevicePtr,
    /// RMSNorm (with no weight) on the values: Gemma 4.
    pub v_norm: bool,
    pub cache: Option<(CuDevicePtr, CuDevicePtr)>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_base: f32,
    pub rope_factors: Option<CuDevicePtr>,
    /// RoPE on halves (Qwen, Gemma) instead of adjacent pairs (Llama).
    pub neox: bool,
    pub pos: CuDevicePtr,
    pub eps: f32,
    /// The cache holds f16 (8-bit mode), not f32.
    pub half_cache: bool,
    /// Rings (see the Metal backend's `attention_ring`) are Metal's only: 0 here.
    pub ring: usize,
}

/// What [`Gpu::delta_net`] works on: token t's convolved `[q | k | v]` is row t of
/// `conv` (`channels` wide, q and k `key_dim` each); `beta_raw` and `alpha_raw` have a
/// row of `v_heads` per token, `z` and `out` a row of `v_heads × n`.
pub struct DeltaNet {
    pub conv: CuDevicePtr,
    pub channels: usize,
    pub key_dim: usize,
    pub beta_raw: CuDevicePtr,
    pub alpha_raw: CuDevicePtr,
    pub dt_bias: CuDevicePtr,
    pub a: CuDevicePtr,
    pub state: CuDevicePtr,
    pub z: CuDevicePtr,
    pub norm: CuDevicePtr,
    pub out: CuDevicePtr,
    pub v_heads: usize,
    pub n: usize,
    pub k_heads: usize,
    pub eps: f32,
    /// Room for a prompt in chunks ([`Gpu::delta_scratch_floats`]), or 0 to run it token
    /// by token.
    pub scratch: CuDevicePtr,
    /// Room for the state before a batch of up to [`Gpu::FEW_TOKENS`] tokens and after each
    /// of them but the last (a whole state each), for undoing a draft's guesses, or 0.
    pub steps: CuDevicePtr,
}

/// Tokens per chunk of a prompt's DeltaNet (DC in vireo.cu).
const DELTA_CHUNK: usize = 64;

/// delta_chunk_prep's shared memory: the chunk's k (rows of n + 4), which A (rows of
/// DELTA_CHUNK + 1) then takes the place of, the cumulative log decays and β.
fn delta_chunk_prep_bytes(n: usize) -> u32 {
    let c = DELTA_CHUNK;
    ((c * (n + 4)).max(c * (c + 1)) * 4 + 2 * c * 4) as u32
}

/// Activations rounded to 8 bits ([`Gpu::quantize`]): per row of `cols`, `cols` signed
/// bytes, a scale per 32 and the scaled sum of each 16.
pub struct Q8 {
    q: Buffer,
    scales: Buffer,
    sums: Buffer,
}

/// A kernel argument, by type.
#[derive(Clone, Copy)]
enum Arg {
    Ptr(CuDevicePtr),
    I32(i32),
    F32(f32),
}
