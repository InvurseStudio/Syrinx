//! The Metal backend: Vireo's own kernels (`kernels/vireo.metal`) on the Mac's GPU.
//!
//! On macOS this module stands in for `cuda` (see `lib.rs`): it has the same types and
//! methods, so the models' GPU forward passes (`models::*_cuda`), the CLI, the server and
//! the tests run on it unchanged. The kernels are a port of `kernels/vireo.cu`, compiled
//! from source when the GPU is opened (a fraction of a second), with Metal's fast math off
//! so they match the CPU reference as the CUDA ones do.
//!
//! How the CUDA ideas map:
//!
//! - **Addresses.** A [`CuDevicePtr`] is a GPU address, as on CUDA: `MTLBuffer.gpuAddress`
//!   plus an offset, which the models do arithmetic on. Kernels take them as raw pointers,
//!   so every buffer is kept resident through one residency set, and Metal's hazard
//!   tracking is off (the work runs in order, as on one CUDA stream).
//! - **The stream.** Work is encoded, in order, into one compute encoder that runs its
//!   dispatches one after another; it's committed and waited for when results are read
//!   ([`Gpu::synchronize`], [`Buffer::read_f32`] and the like).
//! - **Copies.** Memory is shared with the CPU. A write when the GPU has nothing queued is
//!   a plain copy; with work queued it goes in line behind it, through a staging buffer and
//!   a copy kernel, as CUDA's asynchronous copies do.
//! - **Graphs.** A recording is the list of dispatches with their arguments; replaying it
//!   encodes them again, which costs a few microseconds per dispatch.
//! - **The default mode's matrix products.** Apple GPUs have no dp4a, so there are no 8-bit
//!   activations. On GPUs with neural accelerators (M5, A19 and later; Metal 4's tensor
//!   operations, `kernels/vireo_tensor.metal`), the default mode's prompts go there
//!   instead, with the weights rounded to f16: [`Gpu::supports_8bit`] says so, the
//!   "8-bit" input is the f32 activations themselves, and exact mode keeps f32
//!   throughout. Without them the default mode is exact mode's kernels.
//!
//! All GPU work goes, in order, through one queue. The types are Send and Sync, as the CUDA
//! ones are, and like them meant to be used from one thread at a time.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use crate::tensor_type::TensorType;

// MTLCreateSystemDefaultDevice lives in CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// A GPU address: a buffer's `gpuAddress` plus an offset in bytes.
pub type CuDevicePtr = u64;

#[derive(Debug)]
pub enum Error {
    /// No usable GPU.
    Unavailable(String),
    /// A Metal call failed.
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
            Error::Driver { call, code, name } => write!(f, "Metal's {call} failed: {name} ({code})"),
            Error::Unsupported(kind) => write!(f, "{kind} has no GPU kernel yet"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

type Device = ProtocolObject<dyn MTLDevice>;
type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Shared memory, untracked: the work is ordered by the encoder, not by Metal.
fn buffer_options() -> MTLResourceOptions {
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked
}

/// One kernel dispatch: what a recording holds, and what's encoded.
#[derive(Clone)]
struct Dispatch {
    pipeline: Pso,
    args: [u64; 24],
    count: usize,
    grid: (u32, u32, u32),
    threads: u32,
    shared: u32,
}

/// The command buffer being filled, and its encoder.
struct Open {
    commands: CommandBuffer,
    encoder: Encoder,
}

#[derive(Default)]
struct State {
    open: Option<Open>,
    /// Dispatches being recorded instead of run.
    recording: Option<Vec<Dispatch>>,
    /// Staging buffers for writes queued behind other work, kept until it's done.
    staging: Vec<MtlBuffer>,
    /// Buffers were added to or taken from the residency set since its last commit.
    residency_dirty: bool,
    /// With VIREO_METAL_PROFILE set: each dispatch runs on its own, and its GPU time is
    /// added up by kernel (seconds, dispatches), printed when the GPU is closed.
    profile: Option<HashMap<String, (f64, usize)>>,
}

/// The device, its queue, the kernels and the in-order state all work goes through.
/// Shared by everything that holds GPU memory, so it's released last.
struct Context {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    copy: Pso,
    state: Mutex<State>,
    /// Each pipeline's kernel name, by its address, for the profile.
    names: HashMap<usize, String>,
}

// Metal's objects may be used from any thread; the engine uses each GPU from one at a time
// (see the module docs), and the state is behind a mutex.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

fn driver_error(call: &'static str, e: &objc2_foundation::NSError) -> Error {
    Error::Driver {
        call,
        code: e.code() as i32,
        name: e.localizedDescription().to_string(),
    }
}

impl Context {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A new buffer of `bytes` (with `data` copied in, when given), resident from the next
    /// commit on.
    fn new_buffer(&self, bytes: usize, data: Option<&[u8]>) -> Result<MtlBuffer> {
        let length = bytes.max(16);
        let buffer = match data {
            Some(d) if !d.is_empty() => unsafe {
                self.device.newBufferWithBytes_length_options(
                    NonNull::new(d.as_ptr() as *mut c_void).unwrap(),
                    d.len(),
                    buffer_options(),
                )
            },
            _ => self.device.newBufferWithLength_options(length, buffer_options()),
        };
        let buffer = buffer.ok_or(Error::Driver {
            call: "newBuffer",
            code: 0,
            name: format!("couldn't allocate {bytes} bytes"),
        })?;
        Ok(buffer)
    }

    fn track(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        self.residency.addAllocation(ProtocolObject::from_ref(buffer));
        self.lock().residency_dirty = true;
    }

    /// The encoder for the work being queued, starting a command buffer if needed.
    fn encoder<'s>(&self, state: &'s mut State) -> Result<&'s Encoder> {
        if state.open.is_none() {
            let commands = self.queue.commandBuffer().ok_or(Error::Driver {
                call: "commandBuffer",
                code: 0,
                name: "no command buffer".into(),
            })?;
            let encoder = commands
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .ok_or(Error::Driver {
                    call: "computeCommandEncoder",
                    code: 0,
                    name: "no encoder".into(),
                })?;
            state.open = Some(Open { commands, encoder });
        }
        Ok(&state.open.as_ref().unwrap().encoder)
    }

    fn encode(&self, state: &mut State, d: &Dispatch) -> Result<()> {
        let encoder = self.encoder(state)?;
        encoder.setComputePipelineState(&d.pipeline);
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(d.args.as_ptr() as *mut c_void).unwrap(),
                d.count.max(1) * 8,
                0,
            );
            if d.shared > 0 {
                encoder.setThreadgroupMemoryLength_atIndex(d.shared as usize, 0);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: d.grid.0 as usize,
                height: d.grid.1 as usize,
                depth: d.grid.2 as usize,
            },
            MTLSize {
                width: d.threads as usize,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// With profiling on: runs what's queued (just `d`) and adds its GPU time to the profile.
    fn profile_step(&self, state: &mut State, d: &Dispatch) -> Result<()> {
        if state.profile.is_none() {
            return Ok(());
        }
        let commands = state.open.as_ref().unwrap().commands.clone();
        self.finish(state)?;
        let seconds = commands.GPUEndTime() - commands.GPUStartTime();
        let name = self
            .names
            .get(&(Retained::as_ptr(&d.pipeline) as *const c_void as usize))
            .cloned()
            .unwrap_or_else(|| "?".into());
        let entry = state.profile.as_mut().unwrap().entry(name).or_default();
        entry.0 += seconds;
        entry.1 += 1;
        Ok(())
    }

    /// Runs `d` after the work already queued, or records it.
    fn dispatch(&self, d: Dispatch) -> Result<()> {
        if d.grid.0 == 0 || d.grid.1 == 0 || d.grid.2 == 0 {
            return Ok(());
        }
        debug_assert!(
            d.threads as usize <= d.pipeline.maxTotalThreadsPerThreadgroup(),
            "{} threads is over the kernel's {}",
            d.threads,
            d.pipeline.maxTotalThreadsPerThreadgroup()
        );
        let mut state = self.lock();
        if let Some(recording) = state.recording.as_mut() {
            recording.push(d);
            return Ok(());
        }
        self.encode(&mut state, &d)?;
        self.profile_step(&mut state, &d)
    }

    /// Commits the queued work and waits for it.
    fn finish(&self, state: &mut State) -> Result<()> {
        let Some(open) = state.open.take() else {
            return Ok(());
        };
        open.encoder.endEncoding();
        if state.residency_dirty {
            self.residency.commit();
            state.residency_dirty = false;
        }
        open.commands.commit();
        open.commands.waitUntilCompleted();
        state.staging.clear();
        if open.commands.status() == MTLCommandBufferStatus::Error {
            return Err(match open.commands.error() {
                Some(e) => driver_error("running the queued work", &e),
                None => Error::Driver {
                    call: "running the queued work",
                    code: 0,
                    name: "unknown error".into(),
                },
            });
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        let mut state = self.lock();
        self.finish(&mut state)
    }

    /// Copies `data` to `offset` bytes into `to`, after the work already queued.
    fn write(&self, to: &Buffer, offset: usize, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut state = self.lock();
        assert!(
            state.recording.is_none(),
            "copying to the GPU while recording"
        );
        if state.open.is_none() {
            // Nothing queued: the GPU is done with the buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    to.buffer.contents().as_ptr().cast::<u8>().add(offset),
                    data.len(),
                );
            }
            return Ok(());
        }
        let staged = self.new_buffer(data.len(), Some(data))?;
        let encoder = self.encoder(&mut state)?;
        encoder.useResource_usage(ProtocolObject::from_ref(&*staged), MTLResourceUsage::Read);
        let d = copy_dispatch(&self.copy, to.ptr + offset as u64, staged.gpuAddress(), data.len());
        self.encode(&mut state, &d)?;
        state.staging.push(staged);
        Ok(())
    }

    /// Copies `out.len()` bytes from `offset` bytes into `from` once the queued work is done.
    fn read(&self, from: &Buffer, offset: usize, out: &mut [u8]) -> Result<()> {
        let mut state = self.lock();
        assert!(
            state.recording.is_none(),
            "copying from the GPU while recording"
        );
        self.finish(&mut state)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                from.buffer.contents().as_ptr().cast::<u8>().add(offset),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }
}

fn copy_dispatch(copy: &Pso, to: CuDevicePtr, from: CuDevicePtr, bytes: usize) -> Dispatch {
    let aligned = (to | from | bytes as u64) & 15 == 0;
    let items = if aligned { bytes / 16 } else { bytes };
    let mut args = [0u64; 24];
    args[0] = to;
    args[1] = from;
    args[2] = bytes as u64;
    Dispatch {
        pipeline: copy.clone(),
        args,
        count: 3,
        grid: ((items as u32).div_ceil(256).max(1), 1, 1),
        threads: 256,
        shared: 0,
    }
}

/// GPU memory, shared with the CPU. Freed when dropped.
pub struct Buffer {
    context: Arc<Context>,
    buffer: MtlBuffer,
    ptr: CuDevicePtr,
    bytes: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

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
        self.context.write(self, index * 4, as_bytes(data))
    }

    /// Copies `data` to the start of the buffer.
    pub fn write_i32(&self, data: &[i32]) -> Result<()> {
        assert!(data.len() * 4 <= self.bytes);
        self.context.write(self, 0, as_bytes(data))
    }

    /// Copies the start of the buffer into `out`, waiting for queued GPU work first.
    pub fn read_f32(&self, out: &mut [f32]) -> Result<()> {
        self.read_f32_at(0, out)
    }

    /// Copies `out.len()` bytes from byte `offset` on into `out`, waiting for queued GPU
    /// work first.
    pub fn read_bytes_at(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        assert!(offset + out.len() <= self.bytes);
        self.context.read(self, offset, out)
    }

    /// Copies `data` to the buffer from byte `offset` on, after the GPU work already
    /// queued.
    pub fn write_bytes_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        assert!(offset + data.len() <= self.bytes);
        self.context.write(self, offset, data)
    }

    /// Copies from f32 number `index` on into `out`, waiting for queued GPU work first.
    pub fn read_f32_at(&self, index: usize, out: &mut [f32]) -> Result<()> {
        assert!((index + out.len()) * 4 <= self.bytes);
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.context.read(self, index * 4, bytes)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Queued work may still use it: let it finish first (cuMemFree waits too).
        let mut state = self.context.lock();
        if state.recording.is_none() {
            let _ = self.context.finish(&mut state);
        }
        self.context
            .residency
            .removeAllocation(ProtocolObject::from_ref(&*self.buffer));
        state.residency_dirty = true;
    }
}

fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

/// A recorded run of dispatches, replayed with [`Graph::launch`]. It holds the GPU
/// addresses it was recorded with: the buffers they belong to must outlive it.
pub struct Graph {
    context: Arc<Context>,
    dispatches: Vec<Dispatch>,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    /// Queues the whole recorded run.
    pub fn launch(&self) -> Result<()> {
        let mut state = self.context.lock();
        assert!(state.recording.is_none(), "launching a graph while recording");
        for d in &self.dispatches {
            self.context.encode(&mut state, d)?;
            self.context.profile_step(&mut state, d)?;
        }
        Ok(())
    }
}

struct Kernels {
    matvec: [Pso; 7],
    /// The matrix-matrix products on the SIMD-group matrix units, in f32.
    matmul_sg: [Pso; 7],
    /// On the neural accelerators (f16 weights), where the GPU has them.
    matmul_tc: Option<[Pso; 7]>,
    rms_norm: Pso,
    residual_norm: Pso,
    qkv_prep: Pso,
    attention: Pso,
    attention_h: Pso,
    attention_part: [Pso; 2],
    attention_combine: Pso,
    /// For head sizes 64, 128, 256 and 512, over f32 and f16 caches.
    attention_tiled: [Pso; 4],
    attention_tiled_h: [Pso; 4],
    /// A prompt over an f16 cache on the SIMD-group matrix units, for head sizes 64, 128,
    /// 256 and 512 (tokens per threadgroup in `FLASH_TOKENS`).
    attention_flash: [Pso; 4],
    swiglu: Pso,
    geglu: Pso,
    add: Pso,
    softcap: Pso,
    apply_gate: Pso,
    conv_silu: Pso,
    conv_silu_parallel: Pso,
    conv_history: Pso,
    delta_net: Pso,
    delta_net_wide: Pso,
    delta_l2: Pso,
    delta_scan: Pso,
    delta_out: Pso,
    /// A prompt's DeltaNet in chunks of DELTA_CHUNK tokens.
    delta_chunk_prep: Pso,
    delta_chunk_scan: Pso,
}

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

/// What Metal reports about the GPU. The fields are the CUDA backend's: `multiprocessors`
/// is the number of GPU cores, and there's no compute capability or driver version.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
    /// The most memory the GPU should use (Metal's recommended working set).
    pub memory_bytes: usize,
    pub driver_version: i32,
}

/// The Mac's GPU with Vireo's kernels loaded.
pub struct Gpu {
    context: Arc<Context>,
    kernels: Kernels,
    /// attention_part's partial results: room for MAX_PARTS parts of 512-wide heads, 64
    /// heads. All work goes through one queue, so one is enough for every session.
    attention_parts: Buffer,
    pub info: DeviceInfo,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

const SOURCE: &str = include_str!("../../kernels/vireo.metal");
/// The kernels on the neural accelerators: compiled after SOURCE, where Metal has them.
const TENSOR_SOURCE: &str = include_str!("../../kernels/vireo_tensor.metal");

/// Whether the GPU opened has neural accelerators (the tensor kernels compiled).
static TENSOR_OPS: AtomicBool = AtomicBool::new(false);

/// The number of threads the matrix-vector kernels use per threadgroup: 4 SIMD groups.
const ROWS_PER_BLOCK: u32 = 4;
/// The matrix-matrix kernel's tile (matmul_sg): rows of weights by tokens.
const TILE_ROWS: u32 = 64;
const TILE_TOKENS: u32 = 32;
/// The tensor kernel's tile (matmul_tc in vireo_tensor.metal).
const TC_ROWS: u32 = 64;
const TC_TOKENS: u32 = 128;
/// attention_flash's tokens per threadgroup, for head sizes 64, 128, 256 and 512.
const FLASH_TOKENS: [u32; 4] = [32, 32, 16, 8];
/// The most parts one token's attention is split into.
const MAX_PARTS: u32 = 16;

/// A parameter for the kernels, which take raw addresses: 0 stands for "none".
fn or_null(buffer: Option<&Buffer>) -> CuDevicePtr {
    buffer.map_or(0, Buffer::ptr)
}

impl Gpu {
    /// Opens the Mac's GPU.
    pub fn open() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::Unavailable("Metal found no GPU".into()))?;
        let options = MTLCompileOptions::new();
        // No fast-math rewrites, and the precise exp, tanh and the like: the fast tanh
        // overflows to NaN for large inputs (GeGLU and Gemma's softcap).
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        // With the tensor kernels where Metal compiles them (it can't before the M5 and A19).
        let with_tensors = format!("{SOURCE}\n{TENSOR_SOURCE}");
        let (library, tensor_ops) = match device
            .newLibraryWithSource_options_error(&NSString::from_str(&with_tensors), Some(&options))
        {
            Ok(library) => (library, true),
            Err(_) => (
                device
                    .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
                    .map_err(|e| driver_error("compiling the kernels", &e))?,
                false,
            ),
        };
        let names = std::cell::RefCell::new(HashMap::new());
        let pipeline = |name: &str| -> Result<Pso> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Driver {
                    call: "newFunctionWithName",
                    code: 0,
                    name: format!("no kernel {name}"),
                })?;
            let pso = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| driver_error("newComputePipelineState", &e))?;
            names
                .borrow_mut()
                .insert(Retained::as_ptr(&pso) as *const c_void as usize, name.to_string());
            Ok(pso)
        };
        let per_kind = |prefix: &str| -> Result<[Pso; 7]> {
            let v: Vec<Pso> = KINDS
                .iter()
                .map(|k| pipeline(&format!("{prefix}_{}", k.name().to_lowercase())))
                .collect::<Result<_>>()?;
            Ok(v.try_into().unwrap_or_else(|_| unreachable!()))
        };
        let kernels = Kernels {
            matvec: per_kind("matvec")?,
            matmul_sg: per_kind("matmul_sg")?,
            matmul_tc: if tensor_ops {
                Some(per_kind("matmul_tc")?)
            } else {
                None
            },
            rms_norm: pipeline("rms_norm")?,
            residual_norm: pipeline("residual_norm")?,
            qkv_prep: pipeline("qkv_prep")?,
            attention: pipeline("attention")?,
            attention_h: pipeline("attention_h")?,
            attention_part: [pipeline("attention_part_f32")?, pipeline("attention_part_f16")?],
            attention_combine: pipeline("attention_combine")?,
            attention_tiled: [
                pipeline("attention_tiled_64")?,
                pipeline("attention_tiled_128")?,
                pipeline("attention_tiled_256")?,
                pipeline("attention_tiled_512")?,
            ],
            attention_flash: [
                pipeline("attention_flash_64")?,
                pipeline("attention_flash_128")?,
                pipeline("attention_flash_256")?,
                pipeline("attention_flash_512")?,
            ],
            attention_tiled_h: [
                pipeline("attention_tiled_h_64")?,
                pipeline("attention_tiled_h_128")?,
                pipeline("attention_tiled_h_256")?,
                pipeline("attention_tiled_h_512")?,
            ],
            swiglu: pipeline("swiglu")?,
            geglu: pipeline("geglu")?,
            add: pipeline("add")?,
            softcap: pipeline("softcap")?,
            apply_gate: pipeline("apply_gate")?,
            conv_silu: pipeline("conv_silu")?,
            conv_silu_parallel: pipeline("conv_silu_parallel")?,
            conv_history: pipeline("conv_history")?,
            delta_net: pipeline("delta_net")?,
            delta_net_wide: pipeline("delta_net_wide")?,
            delta_l2: pipeline("delta_l2")?,
            delta_scan: pipeline("delta_scan")?,
            delta_out: pipeline("delta_out")?,
            delta_chunk_prep: pipeline("delta_chunk_prep")?,
            delta_chunk_scan: pipeline("delta_chunk_scan")?,
        };
        let copy = pipeline("copy_bytes")?;
        TENSOR_OPS.store(tensor_ops, Ordering::Relaxed);

        let queue = device.newCommandQueue().ok_or(Error::Driver {
            call: "newCommandQueue",
            code: 0,
            name: "no queue".into(),
        })?;
        let descriptor = MTLResidencySetDescriptor::new();
        let residency = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|e| driver_error("newResidencySet", &e))?;
        queue.addResidencySet(&residency);

        let info = DeviceInfo {
            name: device.name().to_string(),
            compute_capability: (0, 0),
            multiprocessors: gpu_core_count().unwrap_or(16),
            memory_bytes: device.recommendedMaxWorkingSetSize() as usize,
            driver_version: 0,
        };
        let context = Arc::new(Context {
            device,
            queue,
            residency,
            copy,
            state: Mutex::new(State {
                profile: std::env::var_os("VIREO_METAL_PROFILE").map(|_| HashMap::new()),
                ..State::default()
            }),
            names: names.into_inner(),
        });
        let parts_bytes = 64 * MAX_PARTS as usize * (2 + 512) * 4;
        let attention_parts = Self::alloc_in(&context, parts_bytes)?;
        Ok(Self {
            context,
            kernels,
            attention_parts,
            info,
        })
    }

    fn alloc_in(context: &Arc<Context>, bytes: usize) -> Result<Buffer> {
        let buffer = context.new_buffer(bytes, None)?;
        Self::wrap(context, buffer, bytes)
    }

    fn wrap(context: &Arc<Context>, buffer: MtlBuffer, bytes: usize) -> Result<Buffer> {
        context.track(&buffer);
        let ptr = buffer.gpuAddress();
        Ok(Buffer {
            context: Arc::clone(context),
            buffer,
            ptr,
            bytes,
        })
    }

    /// Whether a weight of this type can be multiplied on the GPU.
    pub fn supports(kind: TensorType) -> bool {
        KINDS.contains(&kind)
    }

    /// Free GPU memory, in bytes: what's left of Metal's recommended working set.
    pub fn free_memory(&self) -> usize {
        let d = &self.context.device;
        (d.recommendedMaxWorkingSetSize() as usize).saturating_sub(d.currentAllocatedSize())
    }

    /// `bytes` of uninitialized GPU memory.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        Self::alloc_in(&self.context, bytes)
    }

    /// A copy of `data` in GPU memory.
    pub fn upload(&self, data: &[u8]) -> Result<Buffer> {
        let buffer = self.context.new_buffer(data.len(), Some(data))?;
        Self::wrap(&self.context, buffer, data.len())
    }

    /// A weight matrix of `rows` × `cols` numbers of type `kind` (GGUF's layout) in GPU
    /// memory, in the layout the kernels read: as `cuda` lays it out (Q4_0's numbers then
    /// scales per row, Q6_K's blocks padded to 224 bytes).
    pub fn upload_weights(
        &self,
        kind: TensorType,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Buffer> {
        if kind == TensorType::Q6_K {
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
        self.upload(as_bytes(data))
    }

    /// Copies `bytes` within GPU memory, after the work already queued.
    pub fn copy(&self, to: CuDevicePtr, from: CuDevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.context
            .dispatch(copy_dispatch(&self.context.copy, to, from, bytes))
    }

    /// Waits for all queued GPU work, and reports any error a kernel hit.
    pub fn synchronize(&self) -> Result<()> {
        self.context.synchronize()
    }

    /// From here until [`Gpu::finish_recording`], dispatches are recorded, not run. Only
    /// dispatches: copying to or from the GPU meanwhile is an error.
    pub fn start_recording(&self) -> Result<()> {
        let mut state = self.context.lock();
        assert!(state.recording.is_none(), "already recording");
        state.recording = Some(Vec::new());
        Ok(())
    }

    /// The dispatches since [`Gpu::start_recording`], as a [`Graph`].
    pub fn finish_recording(&self) -> Result<Graph> {
        let dispatches = self.context.lock().recording.take().unwrap_or_default();
        Ok(Graph {
            context: Arc::clone(&self.context),
            dispatches,
        })
    }

    fn run(&self, kernel: &Pso, grid: (u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        self.run_3d(kernel, (grid.0, grid.1, 1), block, 0, args)
    }

    fn run_3d(
        &self,
        kernel: &Pso,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> Result<()> {
        let mut values = [0u64; 24];
        assert!(args.len() <= values.len());
        for (slot, a) in values.iter_mut().zip(args) {
            *slot = match *a {
                Arg::Ptr(v) => v,
                Arg::I32(v) => v as u32 as u64,
                Arg::F32(v) => v.to_bits() as u64,
            };
        }
        self.context.dispatch(Dispatch {
            pipeline: kernel.clone(),
            args: values,
            count: args.len(),
            grid,
            threads: block,
            shared,
        })
    }

    /// One thread per number, 256 to a threadgroup.
    fn run_per_number(&self, kernel: &Pso, n: usize, args: &[Arg]) -> Result<()> {
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
        let kernel = &self.kernels.matvec[Self::kind_index(kind)?];
        // The quantized kernels give each SIMD group 2 rows (MV_ROWS in vireo.metal).
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
            for t in 0..tokens {
                let (xt, yt) = (x + (t * cols * 4) as u64, y + (t * rows * 4) as u64);
                self.matvec(kind, w, xt, yt, rows, cols)?;
            }
            return Ok(());
        }
        let kernel = &self.kernels.matmul_sg[Self::kind_index(kind)?];
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
        self.run(kernel, grid, 128, &args)
    }

    /// RMSNorm of each of `rows` rows of `n` numbers; `weight` is the same for every row,
    /// or 0 for none. In place is fine.
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
        self.run(&self.kernels.rms_norm, (rows as u32, 1), 256, &args)
    }

    /// The residual stream between steps (see the CUDA backend's `residual_norm`).
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
        self.run(&self.kernels.residual_norm, (tokens as u32, 1), 256, &args)
    }

    /// Gets `tokens` tokens' queries, keys and values ready for attention: see
    /// [`QkvPrep`].
    pub fn qkv_prep(&self, p: &QkvPrep, tokens: usize) -> Result<()> {
        assert!(p.head_dim <= 512, "head size {} is over 512", p.head_dim);
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
        let threads = (p.head_dim.div_ceil(32) * 32).min(256) as u32;
        self.run(
            &self.kernels.qkv_prep,
            (slots as u32, (tokens as u32).div_ceil(8)),
            threads,
            &args,
        )
    }

    /// Causal attention for `tokens` tokens of `heads` query heads (see the CUDA backend's
    /// `attention`). With `half_cache` the cache holds f16.
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
        if tokens == 1 && heads <= 64 {
            // One token: split each head's context into parts so every GPU core has work.
            let cores = self.info.multiprocessors.max(1) as u32;
            let parts = (4 * cores).div_ceil(heads as u32).clamp(1, MAX_PARTS);
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
            let kernel = &self.kernels.attention_part[usize::from(half_cache)];
            self.run(kernel, (heads as u32, parts), 128, &part_args)?;
            let combine = [
                Arg::Ptr(self.attention_parts.ptr()),
                Arg::Ptr(out),
                Arg::I32(parts as i32),
                Arg::I32(head_dim as i32),
            ];
            return self.run(
                &self.kernels.attention_combine,
                (heads as u32, 1),
                128,
                &combine,
            );
        }
        let tiled = match head_dim {
            64 => Some((0, 8)),
            128 => Some((1, 8)),
            256 => Some((2, 16)),
            512 => Some((3, 32)),
            _ => None,
        };
        if let (true, true, Some((kernel, _))) = (tokens > 1, half_cache, tiled) {
            let mut args = args.to_vec();
            args.push(Arg::I32(tokens as i32));
            let blocks = (tokens as u32).div_ceil(FLASH_TOKENS[kernel]);
            return self.run(
                &self.kernels.attention_flash[kernel],
                (heads as u32, blocks),
                128,
                &args,
            );
        }
        if let (true, Some((kernel, lanes))) = (tokens > 1, tiled) {
            // A prompt: a threadgroup of tokens shares each key and value it reads.
            let mut args = args.to_vec();
            args.push(Arg::I32(tokens as i32));
            let blocks = (tokens as u32).div_ceil(256 / lanes);
            let kernels = if half_cache {
                &self.kernels.attention_tiled_h
            } else {
                &self.kernels.attention_tiled
            };
            return self.run(&kernels[kernel], (heads as u32, blocks), 256, &args);
        }
        let one = if half_cache {
            &self.kernels.attention_h
        } else {
            &self.kernels.attention
        };
        self.run(
            one,
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
        self.gated(&self.kernels.swiglu, gate, up, n, up_stride, tokens)
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
        self.gated(&self.kernels.geglu, gate, up, n, up_stride, tokens)
    }

    fn gated(
        &self,
        kernel: &Pso,
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
            &self.kernels.add,
            n,
            &[Arg::Ptr(x), Arg::Ptr(y), Arg::I32(n as i32)],
        )
    }

    /// `x = cap × tanh(x / cap)`
    pub fn softcap(&self, x: CuDevicePtr, cap: f32, n: usize) -> Result<()> {
        self.run_per_number(
            &self.kernels.softcap,
            n,
            &[Arg::Ptr(x), Arg::F32(cap), Arg::I32(n as i32)],
        )
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
        self.run_per_number(&self.kernels.apply_gate, heads * head_dim, &args)
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
        let conv = [
            Arg::Ptr(current),
            Arg::Ptr(history),
            Arg::Ptr(weights),
            Arg::Ptr(out),
            Arg::I32(channels as i32),
            Arg::I32(kernel as i32),
            Arg::I32(tokens as i32),
        ];
        if tokens > 1 {
            let grid = (
                (channels as u32).div_ceil(256),
                (tokens as u32).div_ceil(16),
            );
            self.run(&self.kernels.conv_silu_parallel, grid, 256, &conv)?;
            let shift = [
                Arg::Ptr(current),
                Arg::Ptr(history),
                Arg::I32(channels as i32),
                Arg::I32(kernel as i32),
                Arg::I32(tokens as i32),
            ];
            return self.run_per_number(&self.kernels.conv_history, channels, &shift);
        }
        self.run_per_number(&self.kernels.conv_silu, channels, &conv)
    }

    /// Scratch for a prompt's DeltaNet in chunks ([`DeltaNet::scratch`]), in f32s, for up
    /// to `tokens` tokens at a time.
    pub fn delta_scratch_floats(tokens: usize, v_heads: usize, n: usize) -> usize {
        tokens.div_ceil(DELTA_CHUNK)
            * v_heads
            * (4 * DELTA_CHUNK * n + DELTA_CHUNK * DELTA_CHUNK + 4)
    }

    /// Gated DeltaNet for `tokens` tokens in order, `v_heads` heads with states of
    /// `n` × `n` (n ≤ 256): see [`DeltaNet`].
    pub fn delta_net(&self, d: &DeltaNet, tokens: usize) -> Result<()> {
        let n = d.n;
        assert!(
            n <= 256 && n.is_multiple_of(32),
            "state size {n}: the kernel takes multiples of 32 up to 256"
        );
        if tokens > 1 && n <= 128 && d.scratch != 0 {
            // A prompt in chunks of DELTA_CHUNK tokens (delta_chunk_prep and
            // delta_chunk_scan in vireo.metal), between the norms for every token at once.
            let rows = (tokens * 2 * d.k_heads) as u32;
            let l2 = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            self.run(&self.kernels.delta_l2, (rows, 1), n as u32, &l2)?;
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
            self.run(&self.kernels.delta_chunk_prep, (d.v_heads as u32, chunks), 256, &prep)?;
            let scan = [
                Arg::Ptr(d.scratch),
                Arg::Ptr(d.state),
                Arg::Ptr(d.out),
                Arg::I32(n as i32),
                Arg::I32(tokens as i32),
            ];
            // 16 columns of the state a threadgroup (DSC_COLS in vireo.metal).
            let grid = (d.v_heads as u32, (n / 16) as u32);
            self.run(&self.kernels.delta_chunk_scan, grid, 64, &scan)?;
            let out = [
                Arg::Ptr(d.out),
                Arg::Ptr(d.z),
                Arg::Ptr(d.norm),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            let rows = (tokens * d.v_heads) as u32;
            return self.run(&self.kernels.delta_out, (rows, 1), n as u32, &out);
        }
        if tokens > 1 && n <= 128 {
            // A prompt: the norms for every token at once, the state update split over
            // threadgroups of 16 columns, then the output norm for every token at once.
            let rows = (tokens * 2 * d.k_heads) as u32;
            let l2 = [
                Arg::Ptr(d.conv),
                Arg::I32(d.channels as i32),
                Arg::I32(d.k_heads as i32),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            self.run(&self.kernels.delta_l2, (rows, 1), n as u32, &l2)?;
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
            let grid = (d.v_heads as u32, (n / 16) as u32);
            self.run(&self.kernels.delta_scan, grid, 64, &scan)?;
            let out = [
                Arg::Ptr(d.out),
                Arg::Ptr(d.z),
                Arg::Ptr(d.norm),
                Arg::I32(n as i32),
                Arg::F32(d.eps),
            ];
            let rows = (tokens * d.v_heads) as u32;
            return self.run(&self.kernels.delta_out, (rows, 1), n as u32, &out);
        }
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
        if n <= 128 && n.is_multiple_of(8) {
            return self.run(
                &self.kernels.delta_net,
                (d.v_heads as u32, 1),
                4 * n as u32,
                &args,
            );
        }
        self.run(
            &self.kernels.delta_net_wide,
            (d.v_heads as u32, 1),
            n as u32,
            &args,
        )
    }

    /// Whether a weight of this type has the default mode's own kernels (on the neural
    /// accelerators): only where the GPU has them.
    pub fn supports_8bit(kind: TensorType) -> bool {
        TENSOR_OPS.load(Ordering::Relaxed) && KINDS.contains(&kind)
    }

    /// The default mode's input to the matrix products. On Metal it's the f32 activations
    /// themselves (the kernels round the weights instead), so this takes no memory.
    pub fn alloc_q8(&self, _tokens: usize, _cols: usize) -> Result<Q8> {
        Ok(Q8 {
            x: AtomicU64::new(0),
        })
    }

    /// Readies `x` as the input of [`Gpu::matmul_8bit`]: it's read as it is (see
    /// [`Q8`]), so this only notes where.
    pub fn quantize(&self, x: CuDevicePtr, q8: &Q8, _tokens: usize, cols: usize) -> Result<()> {
        assert!(
            cols.is_multiple_of(32),
            "rows of {cols} aren't whole blocks of 32"
        );
        q8.x.store(x, Ordering::Relaxed);
        Ok(())
    }

    /// `y = W x` for each of `tokens` rows of the activations [`Gpu::quantize`] noted: a
    /// prompt on the neural accelerators, one token as in exact mode.
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
        let x = q8.x.load(Ordering::Relaxed);
        let kernels = match &self.kernels.matmul_tc {
            Some(k) if tokens > 1 && cols.is_multiple_of(32) => k,
            _ => return self.matmul(kind, w, x, y, rows, cols, tokens),
        };
        let grid = (
            (rows as u32).div_ceil(TC_ROWS),
            (tokens as u32).div_ceil(TC_TOKENS),
        );
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::I32(tokens as i32),
        ];
        self.run(&kernels[Self::kind_index(kind)?], grid, 128, &args)
    }

    /// Fills a buffer with zeros.
    pub fn zero(&self, buffer: &Buffer) -> Result<()> {
        let zeros = vec![0u8; buffer.len_bytes()];
        buffer.write_bytes_at(0, &zeros)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let state = self.context.lock();
        let Some(profile) = state.profile.as_ref() else {
            return;
        };
        let total: f64 = profile.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = profile.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        eprintln!("GPU time by kernel ({:.1} ms in all, each dispatch on its own):", total * 1e3);
        for (name, (seconds, count)) in rows {
            eprintln!(
                "  {:>8.2} ms {:>5.1}%  {:>7} × {:>7.1} µs  {name}",
                seconds * 1e3,
                100.0 * seconds / total,
                count,
                seconds * 1e6 / *count as f64
            );
        }
    }
}

/// What [`Gpu::qkv_prep`] works on (see the CUDA backend's `QkvPrep`).
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
}

/// What [`Gpu::delta_net`] works on (see the CUDA backend's `DeltaNet`).
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
}

/// Tokens per chunk of a prompt's DeltaNet (DC in vireo.metal: 32, where CUDA's is 64, so
/// the prep's threadgroup memory fits Metal's 32 KB).
const DELTA_CHUNK: usize = 32;

/// The default mode's input to the matrix products ([`Gpu::quantize`]). The tensor
/// kernels read the f32 activations as they are, so it holds only their address: noted
/// when the products' input is ready, read when they're queued, as CUDA's rounded copy
/// would be.
pub struct Q8 {
    x: AtomicU64,
}

/// A kernel argument, by type.
#[derive(Clone, Copy)]
enum Arg {
    Ptr(CuDevicePtr),
    I32(i32),
    F32(f32),
}

/// The number of GPU cores, from the IO registry (Metal doesn't say).
fn gpu_core_count() -> Option<i32> {
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: *const c_void,
            allocator: *const c_void,
            options: u32,
        ) -> *const c_void;
        fn IOObjectRelease(object: u32) -> i32;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: *const c_void,
            s: *const c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFNumberGetValue(number: *const c_void, kind: isize, out: *mut c_void) -> bool;
        fn CFRelease(object: *const c_void);
    }
    const UTF8: u32 = 0x0800_0100;
    const SINT32: isize = 3;
    unsafe {
        let class = CString::new("AGXAccelerator").unwrap();
        let service = IOServiceGetMatchingService(0, IOServiceMatching(class.as_ptr()));
        if service == 0 {
            return None;
        }
        let key_name = CString::new("gpu-core-count").unwrap();
        let key = CFStringCreateWithCString(std::ptr::null(), key_name.as_ptr(), UTF8);
        let value = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
        CFRelease(key);
        IOObjectRelease(service);
        if value.is_null() {
            return None;
        }
        let mut count = 0i32;
        let ok = CFNumberGetValue(value, SINT32, (&mut count as *mut i32).cast());
        CFRelease(value);
        (ok && count > 0).then_some(count)
    }
}
