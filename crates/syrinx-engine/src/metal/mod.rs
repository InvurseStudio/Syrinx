//! The Metal backend: Vireo's own kernels (`kernels/syrinx.metal`) on the Mac's GPU.
//!
//! On macOS this module stands in for `cuda` (see `lib.rs`): it has the same types and
//! methods, so the models' GPU forward passes (`models::*_cuda`), the CLI, the server and
//! the tests run on it unchanged. The kernels are a port of `kernels/syrinx.cu`, compiled
//! from source when the GPU is opened (a fraction of a second), with Metal's fast math off
//! so they match the CPU reference as the CUDA ones do.
//!
//! How the CUDA ideas map:
//!
//! - **Addresses.** A [`CuDevicePtr`] is a GPU address, as on CUDA: `MTLBuffer.gpuAddress`
//!   plus an offset, which the models do arithmetic on. Kernels take them as raw pointers,
//!   so every buffer is kept resident through one residency set, and Metal's hazard
//!   tracking is off (the work runs in order, as on one CUDA stream).
//! - **The stream.** Work is encoded, in order, into a compute encoder, committed every
//!   128 dispatches so the GPU starts while the rest is encoded (a fence orders each
//!   encoder after the last), and waited for when results are read ([`Gpu::synchronize`],
//!   [`Buffer::read_f32`] and the like). The encoder lets dispatches overlap, as a CUDA stream doesn't, so the
//!   backend puts a memory barrier before each dispatch that touches a buffer an earlier
//!   one since the last barrier wrote (or writes one it read): it knows which of each
//!   kernel's arguments it writes ([`written_args`]) and which buffer each address falls
//!   in. Independent steps (a layer's q, k and v, its gate and up) then run together.
//! - **Copies.** Memory is shared with the CPU. A write when the GPU has nothing queued is
//!   a plain copy; with work queued it goes in line behind it, through a staging buffer and
//!   a copy kernel, as CUDA's asynchronous copies do.
//! - **Graphs.** A recording is the list of dispatches with their arguments; replaying it
//!   encodes them again, which costs a few microseconds per dispatch.
//! - **The default mode's matrix products.** Apple GPUs have no dp4a, so there are no 8-bit
//!   activations. On GPUs with neural accelerators (M5, A19 and later; Metal 4's tensor
//!   operations, `kernels/syrinx_tensor.metal`), the default mode's prompts go there
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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
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
            Error::Driver { call, code, name } => {
                write!(f, "Metal's {call} failed: {name} ({code})")
            }
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
    /// The buffers it reads or writes (by start address; true when it writes), or None
    /// when that isn't known: then it's kept apart from everything around it.
    access: Option<Vec<(u64, bool)>>,
}

/// Which of a kernel's arguments (by index) it writes; the other addresses it's given it
/// only reads. None for a kernel without a list, which is kept apart from all others.
fn written_args(name: &str) -> Option<&'static [usize]> {
    Some(match name {
        n if n.starts_with("matvec") || n.starts_with("matmul") => &[2],
        "rms_norm" => &[2],
        "moe_pick" => &[2, 3],
        n if n.starts_with("moe_matvec") || n.starts_with("moe_matmul") => &[2],
        "moe_group" => &[1, 2, 7],
        "moe_geglu" | "moe_sum" => &[0],
        "moe_gather_half" | "moe_geglu_half" => &[1],
        "residual_norm" => &[0, 5],
        "qkv_prep" => &[4, 9, 10, 23],
        "dequant_kv_q8_to_half" => &[1],
        n if n.starts_with("attention_part") => &[3],
        n if n.starts_with("attention_decode") => &[3, 10, 11],
        "attention_tcg_merge" => &[1],
        n if n.starts_with("attention_tcg") => &[3, 11],
        "attention_combine" => &[1],
        n if n.starts_with("attention") => &[3],
        "swiglu" | "geglu" | "add" | "softcap" | "apply_gate" => &[0],
        "swiglu_half" | "geglu_half" => &[2],
        "swiglu_i8" | "geglu_i8" => &[2, 3],
        "conv_silu" => &[1, 3],
        "conv_silu_parallel" => &[3],
        "conv_history" => &[1],
        "delta_net" | "delta_net_8" | "delta_net_wide" => &[7, 10],
        "delta_l2" => &[0],
        "delta_scan" => &[7, 8],
        "delta_out" => &[0],
        "delta_chunk_prep" => &[7],
        n if n.starts_with("delta_chunk_scan") => &[1, 2],
        "copy_bytes" => &[0],
        "argmax_part" => &[1, 2],
        "argmax_final" => &[2],
        "to_half" | "q_group_half" => &[1],
        "to_i8" => &[1, 2],
        _ => return None,
    })
}

/// The command buffer being filled, its encoder, and how many dispatches are in it.
struct Open {
    commands: CommandBuffer,
    encoder: Encoder,
    count: usize,
}

#[derive(Default)]
struct State {
    open: Option<Open>,
    /// Command buffers committed while more work was being encoded, not yet waited for.
    inflight: Vec<CommandBuffer>,
    /// What the dispatches since the encoder's last barrier read and wrote (see
    /// [`Dispatch::access`]); `blocked` when one of them is unknown.
    pending: Vec<(u64, bool)>,
    blocked: bool,
    /// Dispatches being recorded instead of run.
    recording: Option<Vec<Dispatch>>,
    /// Staging buffers for writes queued behind other work, kept until it's done.
    staging: Vec<MtlBuffer>,
    /// Buffers were added to or taken from the residency set since its last commit.
    residency_dirty: bool,
    /// With VIREO_METAL_PROFILE set: each dispatch runs on its own, and its GPU time is
    /// added up by kernel (seconds, dispatches), printed when the GPU is closed.
    profile: Option<HashMap<String, (f64, usize)>>,
    /// With VIREO_METAL_TIMING set: the command buffers' GPU time and wall time from
    /// commit to done (seconds), and how many, printed when the GPU is closed.
    timing: Option<(f64, f64, usize)>,
}

/// The device, its queue, the kernels and the in-order state all work goes through.
/// Shared by everything that holds GPU memory, so it's released last.
struct Context {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    copy: Pso,
    /// Orders each encoder after the one before (the buffers are untracked): updated at
    /// the end of each, waited for at the start of the next.
    fence: Retained<ProtocolObject<dyn objc2_metal::MTLFence>>,
    /// Commit the command buffer being filled every this many dispatches (0: only when
    /// results are read), so the GPU starts on them while the rest is encoded.
    commit_every: usize,
    state: Mutex<State>,
    /// Each pipeline's kernel name, by its address, for the profile.
    names: HashMap<usize, String>,
    /// Each pipeline's written arguments ([`written_args`]), by its address.
    writes: HashMap<usize, &'static [usize]>,
    /// The live buffers: start address to end.
    allocations: Mutex<std::collections::BTreeMap<u64, u64>>,
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
        // Slack past the end of the memory asked for (zeros, as Metal hands out new
        // buffers): attention_tc reads whole chunks of positions, up to 63 past a cache's
        // last, with weights of 0, and they mustn't fall off the buffer.
        let length = bytes.max(16) + BUFFER_SLACK;
        let buffer = match data {
            Some(d) if !d.is_empty() => unsafe {
                self.device.newBufferWithBytes_length_options(
                    NonNull::new(d.as_ptr() as *mut c_void).unwrap(),
                    d.len(),
                    buffer_options(),
                )
            },
            _ => self
                .device
                .newBufferWithLength_options(length, buffer_options()),
        };
        let buffer = buffer.ok_or(Error::Driver {
            call: "newBuffer",
            code: 0,
            name: format!("couldn't allocate {bytes} bytes"),
        })?;
        Ok(buffer)
    }

    fn track(&self, buffer: &ProtocolObject<dyn MTLBuffer>, bytes: usize) {
        self.residency
            .addAllocation(ProtocolObject::from_ref(buffer));
        self.lock().residency_dirty = true;
        let start = buffer.gpuAddress();
        self.allocations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(start, start + bytes.max(1) as u64);
    }

    /// The bytes from `address` to the end of the live buffer it falls in (0 if in none).
    fn bytes_from(&self, address: u64) -> u64 {
        let allocations = self.allocations.lock().unwrap_or_else(|e| e.into_inner());
        match allocations.range(..=address).next_back() {
            Some((_, &end)) if address < end => end - address,
            _ => 0,
        }
    }

    /// The buffers `d` reads and writes, from its kernel's [`written_args`] and the
    /// live buffers its addresses fall in.
    fn access_of(&self, d: &Dispatch) -> Option<Vec<(u64, bool)>> {
        let key = Retained::as_ptr(&d.pipeline) as *const c_void as usize;
        let writes = self.writes.get(&key)?;
        let allocations = self.allocations.lock().unwrap_or_else(|e| e.into_inner());
        let mut access: Vec<(u64, bool)> = Vec::with_capacity(d.count);
        for (i, &a) in d.args[..d.count].iter().enumerate() {
            if a == 0 {
                continue;
            }
            let Some((&start, &end)) = allocations.range(..=a).next_back() else {
                continue;
            };
            if a >= end {
                continue;
            }
            let write = writes.contains(&i);
            match access.iter_mut().find(|(b, _)| *b == start) {
                Some(entry) => entry.1 |= write,
                None => access.push((start, write)),
            }
        }
        Some(access)
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
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .ok_or(Error::Driver {
                    call: "computeCommandEncoder",
                    code: 0,
                    name: "no encoder".into(),
                })?;
            encoder.waitForFence(&self.fence);
            state.open = Some(Open {
                commands,
                encoder,
                count: 0,
            });
            state.pending.clear();
            state.blocked = false;
        }
        Ok(&state.open.as_ref().unwrap().encoder)
    }

    fn encode(&self, state: &mut State, d: &Dispatch) -> Result<()> {
        self.encoder(state)?;
        // A barrier first when it touches what the dispatches since the last one wrote, or
        // writes what they read.
        let hazard = hazard(&state.pending, state.blocked, d.access.as_deref());
        self.submit_dispatch(state, d, hazard)?;
        match &d.access {
            Some(access) => state.pending.extend_from_slice(access),
            None => state.blocked = true,
        }
        Ok(())
    }

    /// Encodes `d` with a precomputed barrier flag (used by [`Graph::launch`] to skip
    /// per-dispatch hazard scanning on replay).
    fn encode_precomputed(&self, state: &mut State, d: &Dispatch, barrier: bool) -> Result<()> {
        self.encoder(state)?;
        let open_count = state.open.as_ref().unwrap().count;
        self.submit_dispatch(state, d, barrier && open_count > 0)
    }

    fn submit_dispatch(&self, state: &mut State, d: &Dispatch, hazard: bool) -> Result<()> {
        let encoder = &state.open.as_ref().unwrap().encoder;
        if hazard {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            state.pending.clear();
            state.blocked = false;
        }
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
        let open = state.open.as_mut().unwrap();
        open.count += 1;
        // With nothing in flight the GPU is idle: hand it the first few dispatches at once
        // (a token generated ~0.5-1% faster on the M5 Max, Gemma 4 E2B and Qwen 3.5 4B,
        // with 8 to 32 than with 128).
        let every = if state.inflight.is_empty() {
            FIRST_COMMIT
        } else {
            self.commit_every
        };
        if self.commit_every > 0 && open.count >= every {
            self.commit_open(state);
        }
        Ok(())
    }

    /// Ends and commits the command buffer being filled, without waiting for it.
    fn commit_open(&self, state: &mut State) {
        let Some(open) = state.open.take() else {
            return;
        };
        open.encoder.updateFence(&self.fence);
        open.encoder.endEncoding();
        if state.residency_dirty {
            self.residency.commit();
            state.residency_dirty = false;
        }
        open.commands.commit();
        state.inflight.push(open.commands);
    }

    /// With profiling on: runs what's queued (just `d`) and adds its GPU time to the profile.
    fn profile_step(&self, state: &mut State, d: &Dispatch) -> Result<()> {
        if state.profile.is_none() {
            return Ok(());
        }
        let commands = match &state.open {
            Some(open) => open.commands.clone(),
            None => match state.inflight.last() {
                Some(c) => c.clone(),
                None => return Ok(()),
            },
        };
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
    fn dispatch(&self, mut d: Dispatch) -> Result<()> {
        if d.grid.0 == 0 || d.grid.1 == 0 || d.grid.2 == 0 {
            return Ok(());
        }
        debug_assert!(
            d.threads as usize <= d.pipeline.maxTotalThreadsPerThreadgroup(),
            "{} threads is over the kernel's {}",
            d.threads,
            d.pipeline.maxTotalThreadsPerThreadgroup()
        );
        d.access = self.access_of(&d);
        let mut state = self.lock();
        if let Some(recording) = state.recording.as_mut() {
            recording.push(d);
            return Ok(());
        }
        self.encode(&mut state, &d)?;
        self.profile_step(&mut state, &d)
    }

    /// Commits the queued work and waits for it all.
    fn finish(&self, state: &mut State) -> Result<()> {
        let committed = std::time::Instant::now();
        self.commit_open(state);
        let mut result = Ok(());
        for commands in std::mem::take(&mut state.inflight) {
            commands.waitUntilCompleted();
            if let Some(t) = state.timing.as_mut() {
                t.0 += commands.GPUEndTime() - commands.GPUStartTime();
                t.2 += 1;
                // VIREO_METAL_TIMING=each: every command buffer's time in the driver, its
                // wait for the GPU and its time on it.
                if std::env::var("VIREO_METAL_TIMING").is_ok_and(|v| v == "each") {
                    eprintln!(
                        "  command buffer: {:.1} ms in the driver, {:.1} ms waiting, {:.1} ms on the GPU",
                        (commands.kernelEndTime() - commands.kernelStartTime()) * 1e3,
                        (commands.GPUStartTime() - commands.kernelEndTime()) * 1e3,
                        (commands.GPUEndTime() - commands.GPUStartTime()) * 1e3,
                    );
                }
            }
            if commands.status() == MTLCommandBufferStatus::Error && result.is_ok() {
                result = Err(match commands.error() {
                    Some(e) => driver_error("running the queued work", &e),
                    None => Error::Driver {
                        call: "running the queued work",
                        code: 0,
                        name: "unknown error".into(),
                    },
                });
            }
        }
        if let Some(t) = state.timing.as_mut() {
            t.1 += committed.elapsed().as_secs_f64();
        }
        state.staging.clear();
        result
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
        if state.open.is_none() && state.inflight.is_empty() {
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
        let mut d = copy_dispatch(
            &self.copy,
            to.ptr + offset as u64,
            staged.gpuAddress(),
            data.len(),
        );
        d.access = self.access_of(&d);
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
        access: None,
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
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
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
        self.context
            .allocations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.ptr);
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
    /// Precomputed per-dispatch barrier flags within the graph, so `launch` skips the
    /// `state.pending` hazard scan on every replay.
    barriers: Vec<bool>,
    /// What the dispatches after the graph's last barrier read and wrote, and whether one
    /// of them is unknown: the encoder's state once the graph is queued.
    tail: Vec<(u64, bool)>,
    tail_blocked: bool,
}

/// Whether a dispatch that reads and writes `access` (None: unknown) needs a barrier
/// first: it touches what the dispatches since the last one (`pending`; `blocked` when one
/// of them is unknown) wrote, or writes what they read.
fn hazard(pending: &[(u64, bool)], blocked: bool, access: Option<&[(u64, bool)]>) -> bool {
    blocked
        || match access {
            None => !pending.is_empty(),
            Some(access) => access
                .iter()
                .any(|&(b, w)| pending.iter().any(|&(pb, pw)| pb == b && (w || pw))),
        }
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    /// Queues the whole recorded run.
    pub fn launch(&self) -> Result<()> {
        let mut state = self.context.lock();
        assert!(
            state.recording.is_none(),
            "launching a graph while recording"
        );
        if self.dispatches.is_empty() {
            return Ok(());
        }
        // First dispatch checks against any prior work already in the open encoder.
        let first_barrier = state.blocked || !state.pending.is_empty();
        self.context
            .encode_precomputed(&mut state, &self.dispatches[0], first_barrier)?;
        self.context.profile_step(&mut state, &self.dispatches[0])?;
        for (d, &barrier) in self.dispatches[1..].iter().zip(&self.barriers[1..]) {
            self.context.encode_precomputed(&mut state, d, barrier)?;
            self.context.profile_step(&mut state, d)?;
        }
        // As after queuing them one by one. (A barrier skipped along the way, the encoder
        // being new, leaves this more than is pending: a barrier too many at worst.)
        state.pending.clear();
        state.pending.extend_from_slice(&self.tail);
        state.blocked = self.tail_blocked;
        Ok(())
    }
}

struct Kernels {
    matvec: [Pso; 7],
    /// The quantized ones with each row pair split over the threadgroup's 4 SIMD groups
    /// (none for F32 and F16).
    matvec_split: [Option<Pso>; 7],
    /// Q6_K's own kernels for 2 and 3 tokens at once (see [`Gpu::matvec_multi`]).
    matvec_q6_k_multi: Vec<Pso>,
    /// One token's gate and up projections with the activation (Gpu::matvec_glu).
    matvec_glu: [Option<Pso>; 7],
    /// The split kernels with tanh-GELU times another vector after (Gpu::matvec_gelu).
    matvec_gelu: [Option<Pso>; 7],
    /// A few tokens' products (Gpu::matvec_few), for up to 2, 3 and 4 tokens.
    matvec_few: [[Pso; 7]; 3],

    /// The matrix-matrix products on the SIMD-group matrix units, in f32.
    matmul_sg: [Pso; 7],
    /// On the neural accelerators (f16 weights and activations), where the GPU has them,
    /// with the kernel that rounds the activations.
    matmul_tc: Option<([Pso; 7], Pso)>,
    /// Adds up matmul_tc's splits (Gpu::matmul_8bit).
    sum_parts: Option<Pso>,
    /// A long batch's products in 8-bit whole numbers on the neural accelerators
    /// (matmul_i8 in syrinx_tensor.metal): for Q6_K and Q8_0 weights (128- and 256-token
    /// tiles), and the kernels that round the activations.
    matmul_i8: Option<MatmulI8>,
    /// The same with the wide tile, for long batches (TC_WIDE).
    matmul_tc_wide: Option<[Pso; 7]>,
    /// The same with 8- to 192-token tiles, for short batches (SMALL_TILES).
    matmul_tc_small: Option<[[Pso; 7]; 6]>,
    rms_norm: Pso,
    /// A mixture of experts' kernels: its pick, its products for Q8_0 and Q6_K matrices,
    /// its activation and its sum.
    moe_pick: Pso,
    moe_matvec_q8_0: Pso,
    moe_matvec_q6_k: Pso,
    moe_geglu: Pso,
    moe_geglu_half: Pso,
    moe_sum: Pso,
    /// A batch's slots put in order of their expert, and the product a tile of them.
    moe_group: Pso,
    moe_matmul: [Pso; 7],
    moe_gather_half: Pso,
    /// `moe_matmul` on the neural accelerators, where the GPU has them.
    moe_matmul_tc: Option<[Pso; 7]>,
    moe_matmul_tc_small: Option<[Pso; 7]>,
    residual_norm: Pso,
    qkv_prep: Pso,
    dequant_kv_q8_to_half: Pso,
    sigmoid_mul: Pso,
    attention: Pso,
    attention_h: Pso,
    attention_part: [Pso; 2],
    attention_combine: Pso,
    /// One token's attention in one dispatch, over f32 and f16 caches, for head sizes
    /// 64, 128, 256 and 512, by how many query heads they take at once (built for 1; the
    /// kernel takes more, see syrinx.metal).
    attention_decode: [[[Option<Pso>; 4]; 4]; 2],
    /// The same for heads of 128, 256 and 512 with a SIMD group a position (lanes across
    /// the head), over f32, f16, and q8 caches.
    attention_decode_wide: [[Pso; 3]; 3],
    /// The same with each key and value row read once for all the query heads that share
    /// it, over f32, f16, and q8 caches, for heads of 256 and 512, by how many heads a SIMD
    /// group holds (index = that number; see [`GROUP_HEADS`]).
    attention_decode_group: [[[Option<Pso>; 5]; 2]; 3],
    /// For head sizes 64, 128, 256 and 512, over f32 and f16 caches.
    attention_tiled: [Pso; 4],
    attention_tiled_h: [Pso; 4],
    /// A prompt over an f16 cache on the SIMD-group matrix units, for head sizes 64, 128,
    /// 256 and 512 (tokens per threadgroup in `FLASH_TOKENS`).
    attention_flash: [Pso; 4],
    /// The same with bigger query tiles (FLASH2_TOKENS, FLASH2_THREADS).
    attention_flash2: [Pso; 4],
    /// On the neural accelerators, where the GPU has them (ATTENTION_TC_TOKENS).
    attention_tc: Option<[Pso; 4]>,
    /// The same for query heads sharing a key/value head (attention_tcg), by head size
    /// (128, 256, 512) and tile (TCG_TILES), and q_group_half for their queries.
    attention_tcg: Option<([Pso; 3], Pso, Pso)>,
    swiglu: Pso,
    geglu: Pso,
    swiglu_half: Pso,
    geglu_half: Pso,
    add: Pso,
    softcap: Pso,
    apply_gate: Pso,
    conv_silu: Pso,
    conv_silu_parallel: Pso,
    conv_history: Pso,
    delta_net: Pso,
    /// delta_net with 8 threads to a column (delta_net_p<8>).
    delta_net_8: Pso,
    delta_net_wide: Pso,
    delta_l2: Pso,
    delta_scan: Pso,
    delta_out: Pso,
    /// A prompt's DeltaNet in chunks of DELTA_CHUNK tokens.
    delta_chunk_prep: Pso,
    delta_chunk_scan: Pso,
    /// delta_chunk_scan on SIMD-group matrices, 8 columns a threadgroup.
    delta_chunk_scan_sg: [Pso; 1],
    /// delta_chunk_scan on Metal 4 neural accelerators, 16 columns a threadgroup (n = 128).
    delta_chunk_scan_tc: Option<Pso>,
    argmax_part: Pso,
    argmax_final: Pso,
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
    /// matmul_tc's splits' sums (MATMUL_PARTS_BYTES).
    matmul_parts: Buffer,
    /// attention_decode's count of finished parts, one per head (left at 0).
    attention_counters: Buffer,
    /// attention_tcg's queries in f16 (q_group_half), grown to the largest batch seen.
    attention_qh: Mutex<Option<Buffer>>,
    /// attention_tcg's parts for attention_tcg_merge, grown the same way.
    attention_tcg_parts: Mutex<Option<Buffer>>,
    /// Buffers those two outgrew: kept, as a recorded graph may still name them.
    retired: Mutex<Vec<Buffer>>,
    /// The queries the last [`Gpu::qkv_prep`] also wrote there, until an attention call
    /// takes them: (q, tokens, heads, head size, group).
    qh_ready: Mutex<Option<ReadyQueries>>,
    /// Scratch buffers for GPU-side [`Gpu::argmax`] (up to 256 partials + 4-byte result).
    argmax_vals: Buffer,
    argmax_idxs: Buffer,
    argmax_out: Buffer,
    /// Reusable f16 KV dequantization scratch buffers for multi-token prompt attention
    /// when `VIREO_KV_Q8=1` stores KV caches in Q8_0.
    kv_half_k: Mutex<Option<Buffer>>,
    kv_half_v: Mutex<Option<Buffer>>,
    pub info: DeviceInfo,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

const SOURCE: &str = include_str!("../../kernels/syrinx.metal");
/// The kernels on the neural accelerators: compiled after SOURCE, where Metal has them.
const TENSOR_SOURCE: &str = include_str!("../../kernels/syrinx_tensor.metal");

/// Whether the GPU opened has neural accelerators (the tensor kernels compiled).
static TENSOR_OPS: AtomicBool = AtomicBool::new(false);

/// Dispatches in the first command buffer committed while the GPU is idle.
const FIRST_COMMIT: usize = 16;

/// The number of threads the matrix-vector kernels use per threadgroup: 4 SIMD groups.
const ROWS_PER_BLOCK: u32 = 4;
/// The slots in a tile of `moe_matmul` (SG_TOKENS in syrinx.metal).
const MOE_TILE_SLOTS: usize = 32;
/// The slots in a tile of `moe_matmul_tc` (MOE_TC_TOKENS in syrinx_tensor.metal), and
/// its smaller tile for multi-chat decode / short batches where each expert gets few slots.
const MOE_TC_SLOTS: usize = 64;
const MOE_TC_SMALL_SLOTS: usize = 16;
/// The matrix-matrix kernel's tile (matmul_sg): rows of weights by tokens.
const TILE_ROWS: u32 = 64;
const TILE_TOKENS: u32 = 32;
/// The tensor kernel's tile (matmul_tc in syrinx_tensor.metal): weight rows, tokens and
/// SIMD groups. Of the shapes tried on the M5 Max (16–128 rows, 64–256 tokens, 2–8
/// groups), 32 × 128 × 4 read prompts fastest on the small test models at first; with
/// it, 128 columns a step (TC_K) was ~3% faster for Gemma 4 than 64, and level for Qwen
/// 3.5. Long batches take a wider one (TC_WIDE): of the shapes tried on Gemma 4 31B's
/// 6,144-token prompt, 32 × 256 × 4 678 tokens/s, 64 × 128 × 4 655, 64 × 256 × 8 652,
/// 32 × 256 × 8 600.
fn tensor_tile() -> (u32, u32, u32) {
    (32, 128, 4)
}

/// The wide tile's tokens (TC_WIDE), for a prompt's whole batches of at least
/// [`WIDE_TOKENS`]: on Gemma 4 31B, a 6,144-token prompt 533 → 678 tokens/s, E2B 22.4K →
/// 23.9K and E4B 7.3K → 8.0K, Qwen 3.5 4B level. Not for shorter batches: the eval's
/// prompts and tool results, mostly under 1,024 tokens, read 10–20% slower with it.
const TC_WIDE: u32 = 256;
const WIDE_TOKENS: usize = 1024;
/// The short batches' tiles (matmul_tc_t32, matmul_tc_t64): a batch of up to 32 or 64
/// tokens (a tool result, a new turn, a scoring question) gets a tile of its size, not
/// 128, so the accelerators don't multiply padding. On the M5 Max, Gemma 4 31B's Q6_K
/// ffn_up at 16 / 42 / 64 tokens: 0.59 → 0.38-0.42 / 0.60 → 0.46 / 0.62 → 0.51 ms, the
/// same sums to the bit (each tile's arithmetic is the 128-token one's). 9-64-token reads
/// in the eval: 31B 15.5 → 10.5 s over nine cases, E2B, E4B and Qwen 3.5 4B 15-25% less.
/// The product alone (no decoding) takes 0.22 of the 0.42 ms at 32 tokens, so a tile
/// shape can't do much more: tried 16-128 rows, 4 and 8 SIMD groups, two weight tiles
/// in turn and the next step's bytes loaded ahead, all level or slower. 96 for 65-96
/// tokens (ffn_up at 70 / 80 / 96: 0.61 → 0.55, 0.64 → 0.56, 0.64 → 0.58 ms) and 192 for
/// 129-192 (two tiles of 128 before; ffn_up at 150 / 180: 1.13 → 0.90, 1.13 → 0.93) too;
/// 97-128 keep the 128-token one. 8 and 16 for up to 8 and 16 tokens, taking 256 columns
/// a step (the 128 of the others: half as many products, each the accelerators' smallest
/// in tokens): ffn_up at 4 / 8 / 16 tokens 0.33 → 0.25, 0.33 → 0.25, 0.35 → 0.27 ms,
/// ffn_down 0.32 → 0.26, 0.32 → 0.26, 0.33 → 0.27 (one token: 0.19). 256 columns a step
/// did nothing for whole batches (ffn_up at 1,024 tokens 4.69 → 4.63 ms, ffn_down 5.43 →
/// 5.50) and little at 32 (0.35 → 0.32).
///
/// The 256-column tiles hold one tile of decoded weights, not the two the others decode
/// in turn (BUFS in syrinx_tensor.metal): two of 32 rows × 256 columns are the whole 32 KB
/// of a threadgroup's memory, and with them (15-17 November) the 8- and 16-token tiles
/// ran slower than the 32-token one. A batch of 4 / 8 / 16 tokens 4,000 tokens into a
/// chat, two → one (`tests/check_speed.rs`): Gemma 4 31B 102.8 / 107.1 / 113.5 → 67.5 /
/// 70.1 / 75.0 ms (one token 51), Qwen 3.5 9B 24.9 / 27.2 / 29.4 → 18.6 / 20.4 / 22.2
/// (13.7), Gemma 4 E4B 15.0 / 16.0 / 17.4 → 12.5 / 13.5 / 14.5 (9.8). The 32- and
/// 64-token tiles take 256 columns and one tile too since then: 31B at 32 / 64 tokens
/// 102 → 98 / 141 → 137 ms, Qwen 3.5 9B 26.8 → 25.6 / 35.0 → 33.1, E4B 16.1 → 15.7 /
/// 18.8 → 18.3; their sums differ from the 128-token tile's by f32 rounding (under 1e-5
/// of the largest, `tests/big_matmul.rs`).
const SMALL_TILES: [usize; 6] = [8, 16, 32, 64, 96, 192];
/// A short batch's product split along the columns (matmul_tc's `splits`, added up by
/// sum_parts) until there are about SPLIT_TARGET threadgroups: batches of up to
/// SPLIT_TOKENS (256: ffn_down at 150 tokens 1.19 → 0.89 ms with the 192-token tile), at most MAX_SPLITS parts of at least SPLIT_COLS columns, their sums in a
/// buffer of MATMUL_PARTS_BYTES. A matrix of few rows and long ones (Gemma 4 31B's
/// ffn_down, 5,376 × 21,504, and attn_output) otherwise has 168 threadgroups for 40
/// cores, each walking 168 steps: at 20 / 64 / 100 tokens ffn_down 0.77 → 0.33 / 0.91 →
/// 0.45 / 1.02 → 0.63 ms, now level with ffn_up (Q6_K, M5 Max); targets of 320 and 640
/// gave less, 2,560 no more. The sums are added in another order, so the numbers move by
/// f32 rounding (31B, nine eval cases: one reply worded differently, the same passes);
/// the same batch on the same build still gives the same bits.
const SPLIT_TARGET: u32 = 1280;
const SPLIT_TOKENS: usize = 256;
const SPLIT_COLS: usize = 1024;
const MAX_SPLITS: usize = 8;
const MATMUL_PARTS_BYTES: usize = 32 << 20;
/// attention_flash's tokens per threadgroup, for head sizes 64, 128, 256 and 512.
const FLASH_TOKENS: [u32; 4] = [32, 32, 16, 8];
/// attention_flash2's tokens and threads per threadgroup, for the same head sizes.
const FLASH2_TOKENS: [u32; 4] = [64, 64, 32, 16];
const FLASH2_THREADS: [u32; 4] = [128, 256, 256, 256];
/// attention_tc's tokens per threadgroup (256 threads), for the same head sizes.
const ATTENTION_TC_TOKENS: [u32; 4] = [32, 32, 32, 16];
/// Bytes past the end of every buffer [`Gpu::alloc`] makes (see `Context::new_buffer`).
const BUFFER_SLACK: usize = 128 * 1024;
/// Queries [`Gpu::qkv_prep`] left in f16 for attention_tcg: (q, tokens, heads, head size,
/// group).
type ReadyQueries = (CuDevicePtr, usize, usize, usize, usize);
/// attention_tcg's tiles for heads of 128, 256 and 512: query rows, positions a step and
/// threads (see syrinx_tensor.metal).
const TCG_TILES: [(usize, usize, u32); 3] = [(32, 128, 128), (32, 128, 256), (16, 256, 256)];
/// The most parts attention_tcg splits a short batch's context into (attention_tcg_merge's
/// weights).
const TCG_MAX_PARTS: u32 = 64;
/// The experimental 8-bit products (matmul_i8 in syrinx_tensor.metal, which says what they
/// cost in accuracy): the fewest tokens that go to them (the tile is I8_TOKENS tokens by
/// I8_ROWS rows, a step I8_K columns), and whether any do: only with VIREO_I8=on.
struct MatmulI8 {
    q6_k: Pso,
    q8_0: Pso,
    wide_q6_k: Pso,
    wide_q8_0: Pso,
    to_i8: Pso,
    swiglu_i8: Pso,
    geglu_i8: Pso,
}

const I8_FROM: usize = 128;
const I8_TOKENS: u32 = 128;
const I8_WIDE_TOKENS: u32 = 256;
const I8_ROWS: u32 = 32;
const I8_K: usize = 256;
/// 0 = off, 1 = middle-layer FFN only (`VIREO_I8=ffn`), 2 = all (`VIREO_I8=on`).
static I8_MODE: AtomicU8 = AtomicU8::new(0);
static I8_FFN_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Set to keep attention_tc for full attention too (VIREO_TCG=off, or
/// [`Gpu::group_prompt_attention`]), to compare the two.
static TCG_OFF: AtomicBool = AtomicBool::new(false);
fn tcg_off() -> bool {
    TCG_OFF.load(Ordering::Relaxed)
}
/// Whether 1 to 8 tokens' attention may go to attention_tcg too ([`Gpu::grouped_few`]):
/// on unless VIREO_TCG_FEW=off.
static TCG_FEW: AtomicBool = AtomicBool::new(true);
fn tcg_few() -> bool {
    TCG_FEW.load(Ordering::Relaxed)
}
/// The most parts one token's attention is split into.
const MAX_PARTS: u32 = 16;
/// attention_decode_group's builds: for heads of 256 and 512, how many heads a SIMD group
/// may hold (their queries and outputs in registers, so fewer for wider heads).
const GROUP_HEADS: [(usize, &[usize]); 2] = [(256, &[1, 2, 3, 4]), (512, &[1, 2])];
/// attention_decode_group's threadgroup scratch, in floats (GROUP_SCRATCH in syrinx.metal).
const GROUP_SCRATCH: usize = 7936;
/// The most parts attention_decode splits a context into.
const MAX_DECODE_PARTS: u32 = 64;

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
        let (tc_rows, tc_tokens, tc_sg) = tensor_tile();
        let with_tensors = format!(
            "{SOURCE}\n#define TC_ROWS {tc_rows}\n#define TC_TOKENS {tc_tokens}\n#define TC_WIDE {TC_WIDE}\n#define TC_SG {tc_sg}\n#define MOE_TC_TOKENS {MOE_TC_SLOTS}\n{TENSOR_SOURCE}"
        );
        let (library, tensor_ops) = match device
            .newLibraryWithSource_options_error(&NSString::from_str(&with_tensors), Some(&options))
        {
            Ok(library) => (library, true),
            Err(e) => {
                // Expected before the M5 and A19; on a Mac that has the accelerators it
                // means the tensor kernels no longer compile (VIREO_METAL_ERRORS shows why).
                if std::env::var_os("VIREO_METAL_ERRORS").is_some() {
                    eprintln!(
                        "The tensor kernels didn't compile: {}",
                        e.localizedDescription()
                    );
                }
                (
                    device
                        .newLibraryWithSource_options_error(
                            &NSString::from_str(SOURCE),
                            Some(&options),
                        )
                        .map_err(|e| driver_error("compiling the kernels", &e))?,
                    false,
                )
            }
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
            names.borrow_mut().insert(
                Retained::as_ptr(&pso) as *const c_void as usize,
                name.to_string(),
            );
            Ok(pso)
        };
        let per_kind = |prefix: &str| -> Result<[Pso; 7]> {
            let v: Vec<Pso> = KINDS
                .iter()
                .map(|k| pipeline(&format!("{prefix}_{}", k.name().to_lowercase())))
                .collect::<Result<_>>()?;
            Ok(v.try_into().unwrap_or_else(|_| unreachable!()))
        };
        // The quantized types' kernels of a family (none for F32 and F16).
        let quantized = |prefix: &str| -> Result<[Option<Pso>; 7]> {
            let mut v: [Option<Pso>; 7] = Default::default();
            for (slot, kind) in v.iter_mut().zip(KINDS) {
                if !matches!(kind, TensorType::F32 | TensorType::F16) {
                    *slot = Some(pipeline(&format!(
                        "{prefix}_{}",
                        kind.name().to_lowercase()
                    ))?);
                }
            }
            Ok(v)
        };
        let kernels = Kernels {
            matvec: per_kind("matvec")?,
            matvec_split: quantized("matvec_split")?,
            matvec_q6_k_multi: (2..=3)
                .map(|n| pipeline(&format!("matvec_multi_q6_k_{n}")))
                .collect::<Result<_>>()?,
            matvec_glu: quantized("matvec_glu")?,
            matvec_gelu: quantized("matvec_split_gelu")?,
            matvec_few: [
                per_kind("matvec_few2")?,
                per_kind("matvec_few3")?,
                per_kind("matvec_few4")?,
            ],

            matmul_sg: per_kind("matmul_sg")?,
            matmul_tc: if tensor_ops {
                Some((per_kind("matmul_tc")?, pipeline("to_half")?))
            } else {
                None
            },
            matmul_i8: if tensor_ops {
                Some(MatmulI8 {
                    q6_k: pipeline("matmul_i8_q6_k")?,
                    q8_0: pipeline("matmul_i8_q8_0")?,
                    wide_q6_k: pipeline("matmul_i8_wide_q6_k")?,
                    wide_q8_0: pipeline("matmul_i8_wide_q8_0")?,
                    to_i8: pipeline("to_i8")?,
                    swiglu_i8: pipeline("swiglu_i8")?,
                    geglu_i8: pipeline("geglu_i8")?,
                })
            } else {
                None
            },
            matmul_tc_wide: if tensor_ops {
                Some(per_kind("matmul_tc_wide")?)
            } else {
                None
            },
            sum_parts: if tensor_ops {
                Some(pipeline("sum_parts")?)
            } else {
                None
            },
            matmul_tc_small: if tensor_ops {
                Some([
                    per_kind("matmul_tc_t8")?,
                    per_kind("matmul_tc_t16")?,
                    per_kind("matmul_tc_t32")?,
                    per_kind("matmul_tc_t64")?,
                    per_kind("matmul_tc_t96")?,
                    per_kind("matmul_tc_t192")?,
                ])
            } else {
                None
            },
            rms_norm: pipeline("rms_norm")?,
            moe_pick: pipeline("moe_pick")?,
            moe_matvec_q8_0: pipeline("moe_matvec_q8_0")?,
            moe_matvec_q6_k: pipeline("moe_matvec_q6_k")?,
            moe_geglu: pipeline("moe_geglu")?,
            moe_geglu_half: pipeline("moe_geglu_half")?,
            moe_sum: pipeline("moe_sum")?,
            moe_group: pipeline("moe_group")?,
            moe_matmul: per_kind("moe_matmul")?,
            moe_gather_half: pipeline("moe_gather_half")?,
            moe_matmul_tc: if tensor_ops {
                Some(per_kind("moe_matmul_tc")?)
            } else {
                None
            },
            moe_matmul_tc_small: if tensor_ops {
                Some(per_kind("moe_matmul_tc_t16")?)
            } else {
                None
            },
            residual_norm: pipeline("residual_norm")?,
            qkv_prep: pipeline("qkv_prep")?,
            dequant_kv_q8_to_half: pipeline("dequant_kv_q8_to_half")?,
            sigmoid_mul: pipeline("sigmoid_mul")?,
            attention: pipeline("attention")?,
            attention_h: pipeline("attention_h")?,
            attention_part: [
                pipeline("attention_part_f32")?,
                pipeline("attention_part_f16")?,
            ],
            attention_combine: pipeline("attention_combine")?,
            attention_decode_wide: [
                [
                    pipeline("attention_decode_wide_f32_128")?,
                    pipeline("attention_decode_wide_f32_256")?,
                    pipeline("attention_decode_wide_f32_512")?,
                ],
                [
                    pipeline("attention_decode_wide_f16_128")?,
                    pipeline("attention_decode_wide_f16_256")?,
                    pipeline("attention_decode_wide_f16_512")?,
                ],
                [
                    pipeline("attention_decode_wide_q8_128")?,
                    pipeline("attention_decode_wide_q8_256")?,
                    pipeline("attention_decode_wide_q8_512")?,
                ],
            ],
            attention_decode: {
                let mut v: [[[Option<Pso>; 4]; 4]; 2] = Default::default();
                for (c, cache) in ["f32", "f16"].iter().enumerate() {
                    for (i, d) in [64, 128, 256, 512].iter().enumerate() {
                        for (j, hg) in [1].iter().enumerate() {
                            if hg * d / 32 <= 32 {
                                v[c][i][j] =
                                    Some(pipeline(&format!("attention_decode_{cache}_{d}_{hg}"))?);
                            }
                        }
                    }
                }
                v
            },
            attention_decode_group: {
                let mut v: [[[Option<Pso>; 5]; 2]; 3] = Default::default();
                for (c, cache) in ["f32", "f16", "q8"].iter().enumerate() {
                    for (i, (d, hs)) in GROUP_HEADS.iter().enumerate() {
                        for &h in *hs {
                            v[c][i][h] = Some(pipeline(&format!(
                                "attention_decode_group_{cache}_{d}_{h}"
                            ))?);
                        }
                    }
                }
                v
            },
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
            attention_flash2: [
                pipeline("attention_flash2_64")?,
                pipeline("attention_flash2_128")?,
                pipeline("attention_flash2_256")?,
                pipeline("attention_flash2_512")?,
            ],
            attention_tc: if tensor_ops {
                Some([
                    pipeline("attention_tc_64")?,
                    pipeline("attention_tc_128")?,
                    pipeline("attention_tc_256")?,
                    pipeline("attention_tc_512")?,
                ])
            } else {
                None
            },
            attention_tcg: if tensor_ops {
                Some((
                    [
                        pipeline("attention_tcg_128")?,
                        pipeline("attention_tcg_256")?,
                        pipeline("attention_tcg_512")?,
                    ],
                    pipeline("q_group_half")?,
                    pipeline("attention_tcg_merge")?,
                ))
            } else {
                None
            },
            attention_tiled_h: [
                pipeline("attention_tiled_h_64")?,
                pipeline("attention_tiled_h_128")?,
                pipeline("attention_tiled_h_256")?,
                pipeline("attention_tiled_h_512")?,
            ],
            swiglu: pipeline("swiglu")?,
            geglu: pipeline("geglu")?,
            swiglu_half: pipeline("swiglu_half")?,
            geglu_half: pipeline("geglu_half")?,
            add: pipeline("add")?,
            softcap: pipeline("softcap")?,
            apply_gate: pipeline("apply_gate")?,
            conv_silu: pipeline("conv_silu")?,
            conv_silu_parallel: pipeline("conv_silu_parallel")?,
            conv_history: pipeline("conv_history")?,
            delta_net: pipeline("delta_net")?,
            delta_net_8: pipeline("delta_net_8")?,
            delta_net_wide: pipeline("delta_net_wide")?,
            delta_l2: pipeline("delta_l2")?,
            delta_scan: pipeline("delta_scan")?,
            delta_out: pipeline("delta_out")?,
            delta_chunk_prep: pipeline("delta_chunk_prep")?,
            delta_chunk_scan: pipeline("delta_chunk_scan")?,
            delta_chunk_scan_sg: [pipeline("delta_chunk_scan_sg_8")?],
            delta_chunk_scan_tc: if tensor_ops {
                Some(pipeline("delta_chunk_scan_tc_32")?)
            } else {
                None
            },
            argmax_part: pipeline("argmax_part")?,
            argmax_final: pipeline("argmax_final")?,
        };
        let copy = pipeline("copy_bytes")?;
        TENSOR_OPS.store(tensor_ops, Ordering::Relaxed);
        match std::env::var("VIREO_I8").as_deref() {
            Ok("on" | "all") => I8_MODE.store(2, Ordering::Relaxed),
            Ok("ffn") => I8_MODE.store(1, Ordering::Relaxed),
            _ => {}
        }
        if std::env::var("VIREO_TCG").is_ok_and(|v| v == "off") {
            TCG_OFF.store(true, Ordering::Relaxed);
        }
        if std::env::var("VIREO_TCG_FEW").is_ok_and(|v| v == "off") {
            TCG_FEW.store(false, Ordering::Relaxed);
        }
        let fence = device.newFence().ok_or(Error::Driver {
            call: "newFence",
            code: 0,
            name: "no fence".into(),
        })?;
        let names = names.into_inner();
        let writes = names
            .iter()
            .filter_map(|(&key, name)| written_args(name).map(|w| (key, w)))
            .collect();

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
            fence,
            // Every 128 dispatches (a few layers): on the M5 Max prompts were read 6–8%
            // faster and tokens generated 2–3% faster than committing only when results
            // are read, the GPU starting while the CPU encodes the rest (64 and 256 were
            // a little slower).
            commit_every: 128,
            state: Mutex::new(State {
                profile: std::env::var_os("VIREO_METAL_PROFILE").map(|_| HashMap::new()),
                timing: std::env::var_os("VIREO_METAL_TIMING").map(|_| (0.0, 0.0, 0)),
                ..State::default()
            }),
            writes,
            names,
            allocations: Mutex::new(std::collections::BTreeMap::new()),
        });
        let parts_bytes = 64 * MAX_DECODE_PARTS as usize * (2 + 512) * 4;
        let attention_parts = Self::alloc_in(&context, parts_bytes)?;
        // A counter a threadgroup column (up to 64 heads) and token (up to 8).
        let attention_counters = Self::alloc_in(&context, 64 * 64 * 4)?;
        attention_counters.write_bytes_at(0, &[0u8; 64 * 64 * 4])?;
        let matmul_parts = Self::alloc_in(&context, MATMUL_PARTS_BYTES)?;
        let argmax_vals = Self::alloc_in(&context, 256 * 4)?;
        let argmax_idxs = Self::alloc_in(&context, 256 * 4)?;
        let argmax_out = Self::alloc_in(&context, 4)?;
        Ok(Self {
            context,
            kernels,
            attention_parts,
            matmul_parts,
            attention_counters,
            attention_qh: Mutex::new(None),
            attention_tcg_parts: Mutex::new(None),
            retired: Mutex::new(Vec::new()),
            qh_ready: Mutex::new(None),
            argmax_vals,
            argmax_idxs,
            argmax_out,
            kv_half_k: Mutex::new(None),
            kv_half_v: Mutex::new(None),
            info,
        })
    }

    fn alloc_in(context: &Arc<Context>, bytes: usize) -> Result<Buffer> {
        let buffer = context.new_buffer(bytes, None)?;
        Self::wrap(context, buffer, bytes)
    }

    fn wrap(context: &Arc<Context>, buffer: MtlBuffer, bytes: usize) -> Result<Buffer> {
        context.track(&buffer, bytes);
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
    /// scales per row, Q6_K's blocks padded to 212 bytes, whole 4-byte words).
    pub fn upload_weights(
        &self,
        kind: TensorType,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Buffer> {
        if kind == TensorType::Q6_K {
            let mut out = vec![0u8; data.len() / 210 * 212];
            for (src, dst) in data
                .as_chunks::<210>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<212>().0)
            {
                dst[..210].copy_from_slice(src);
                // The block's largest scale, for matmul_i8's rounding.
                dst[210] = src[192..208]
                    .iter()
                    .map(|&s| (s as i8).unsigned_abs())
                    .max()
                    .unwrap_or(0);
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

    /// Index of the largest f32 in `x[0..n]` (smallest index on ties), computed on the
    /// GPU so callers read back 4 bytes instead of the full vocabulary's logits.
    pub fn argmax(&self, x: CuDevicePtr, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        let blocks = n.div_ceil(1024).clamp(1, 256) as u32;
        self.run(
            &self.kernels.argmax_part,
            (blocks, 1),
            256,
            &[
                Arg::Ptr(x),
                Arg::Ptr(self.argmax_vals.ptr()),
                Arg::Ptr(self.argmax_idxs.ptr()),
                Arg::I32(n as i32),
            ],
        )?;
        self.run(
            &self.kernels.argmax_final,
            (1, 1),
            256,
            &[
                Arg::Ptr(self.argmax_vals.ptr()),
                Arg::Ptr(self.argmax_idxs.ptr()),
                Arg::Ptr(self.argmax_out.ptr()),
                Arg::I32(blocks as i32),
            ],
        )?;
        let mut bytes = [0u8; 4];
        self.argmax_out.read_bytes_at(0, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Has the buffers made so far (a model's weights, just uploaded) made resident now.
    /// Left to itself Metal does it when the first work is queued: the first prompt after
    /// a load then waited on it (135 ms for Gemma 4 26B-A4B's 23 GB on the M5 Max).
    pub fn settle(&self) {
        let mut state = self.context.lock();
        if state.residency_dirty {
            self.context.residency.commit();
            state.residency_dirty = false;
        }
        self.context.residency.requestResidency();
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
        let mut barriers = Vec::with_capacity(dispatches.len());
        let mut pending: Vec<(u64, bool)> = Vec::new();
        let mut blocked = false;
        for d in &dispatches {
            let hazard = hazard(&pending, blocked, d.access.as_deref());
            barriers.push(hazard);
            if hazard {
                pending.clear();
                blocked = false;
            }
            match &d.access {
                Some(access) => pending.extend_from_slice(access),
                None => blocked = true,
            }
        }
        Ok(Graph {
            context: Arc::clone(&self.context),
            dispatches,
            barriers,
            tail: pending,
            tail_blocked: blocked,
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
            access: None,
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
        let index = Self::kind_index(kind)?;
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
        ];
        if let Some(split) = &self.kernels.matvec_split[index] {
            // A SIMD group of the plain kernel takes 512 columns a step (Q8_0 128), one
            // after another. Long rows (4 steps or more) and small matrices are quicker
            // with each row pair split over 4 SIMD groups; many short rows aren't (on the
            // M5 Max: 2,048 × 8,192 Q6_K 21.8 → 19.2 µs, 512 × 2,048 Q4_K 5.7 → 4.2, but
            // 6,144 × 1,024 Q5_K 11.1 → 11.9).
            let per_step = if kind == TensorType::Q8_0 { 128 } else { 512 };
            if cols / per_step >= 4 || rows <= 1024 {
                return self.run(split, ((rows as u32).div_ceil(2), 1), 128, &args);
            }
        }
        let kernel = &self.kernels.matvec[index];
        // The quantized kernels give each SIMD group 2 rows (MV_ROWS in syrinx.metal).
        let rows_per_warp = if matches!(kind, TensorType::F32 | TensorType::F16) {
            1
        } else {
            2
        };
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * rows_per_warp);
        self.run(kernel, (grid, 1), ROWS_PER_BLOCK * 32, &args)
    }

    /// Whether `tokens` (2 to [`Gpu::MULTI_TOKENS`]) of a `rows` × `cols` matrix of `kind`
    /// are quicker through the short batches' 8-token tile ([`Gpu::matmul_8bit`]) than
    /// through [`Gpu::matvec_multi`]. The tile costs the same for 2 to 8 tokens, about 1.3
    /// times one token's product on a large matrix; matvec_multi costs more with each
    /// token, and reads the weights twice from 5. On the M5 Max, µs at 3 / 4 / 5 / 8
    /// tokens, matvec_multi against the tile: Gemma 4 31B's Q6_K ffn_up (116M weights)
    /// 221 / 296 / 437 / 583 against 250-260, its attn_k (22M) 62 / 71 / 102 / 124 against
    /// 61; Qwen 3.8 27B's Q8_0 ffn_up (89M) 184 / 180 / 355 / 366 against 255; Qwen 3.5
    /// 4B's Q4_K ffn_gate (24M) 59 / 63 / 92 / 114 against 60, its Q5_K ssm_out (10M)
    /// 39 / 58 / 74 / 82 against 43; Gemma 4 E4B's Q4_0 ffn_gate (26M) 50 / 56 / 79 / 117
    /// against 57. So from 4 tokens for the K-quants and 5 for the others (their kernels
    /// take 4 as cheaply as 2), but for small matrices (1-3M: level or slower).
    /// `tests/big_matmul.rs` (`big_few_or_tile`) measures it.
    pub fn tile_beats_few(kind: TensorType, rows: usize, cols: usize, tokens: usize) -> bool {
        let from = match kind {
            TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => 4,
            _ => 5,
        };
        tokens >= from && rows * cols >= 4 << 20
    }

    /// The most tokens [`Gpu::matvec_multi`] takes at once (MT in syrinx.metal).
    pub const MULTI_TOKENS: usize = 8;

    /// `y[t] = W x[t]` for 2 to [`Gpu::MULTI_TOKENS`] tokens, `x` and `y` a row of f32 a
    /// token, reading W once (or twice, above 4) as for one token: a draft's guesses
    /// checked in one pass, forced text, a short tool result. False, having done
    /// nothing, for another count or columns not a multiple of 8.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_multi(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        tokens: usize,
    ) -> Result<bool> {
        if !(2..=Self::MULTI_TOKENS).contains(&tokens) || !cols.is_multiple_of(8) {
            return Ok(false);
        }
        // Q6_K's own kernel is faster for 2 and 3 tokens (2,560 × 9,216: 35 µs against
        // 53 for 2), slower from 4 on (73 against 61; 570 against ~130 for 8).
        if kind != TensorType::Q6_K || tokens > 3 {
            self.matvec_few(kind, w, x, y, rows, cols, tokens)?;
            return Ok(true);
        }
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
        ];
        // 4 rows a threadgroup: MULTI_ROWS in syrinx.metal for 2 and 3 tokens.
        self.run(
            &self.kernels.matvec_q6_k_multi[tokens - 2],
            ((rows as u32).div_ceil(4), 1),
            128,
            &args,
        )?;
        Ok(true)
    }

    /// One token's gate and up projections and the activation in one dispatch: `y =
    /// act(W_gate x) × (W_up x)`, act SiLU (`gelu` false) or tanh-GELU, both matrices
    /// `rows` × `cols` of type `kind`. What two `matvec`s and `swiglu`/`geglu` do, with
    /// one dependent step fewer. Ok(false), having done nothing, for a type without the
    /// kernel (F32, F16).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_glu(
        &self,
        kind: TensorType,
        w_gate: CuDevicePtr,
        w_up: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        gelu: bool,
    ) -> Result<bool> {
        let Some(kernel) = &self.kernels.matvec_glu[Self::kind_index(kind)?] else {
            return Ok(false);
        };
        let args = [
            Arg::Ptr(w_gate),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::Ptr(w_up),
            Arg::I32(i32::from(gelu)),
        ];
        self.run(kernel, ((rows as u32).div_ceil(2), 1), 128, &args)?;
        Ok(true)
    }

    /// `y[t] = W x[t]` for 1 to [`Gpu::MULTI_TOKENS`] tokens (`x` and `y` a row of f32 a
    /// token), each weight decoded once for up to 4 of them: for a handful of tokens the
    /// prompt kernels' 128-token tiles do far more work than needed. Any type. 5 to 8
    /// tokens take two dispatches of up to 4, which run side by side (on the M5 Max
    /// faster than the prompt kernels for every matrix of the test models, and than one
    /// dispatch of 8).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_few(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        tokens: usize,
    ) -> Result<()> {
        assert!((1..=Self::MULTI_TOKENS).contains(&tokens) && cols.is_multiple_of(8));
        if tokens > 4 {
            let first = tokens / 2;
            self.matvec_few(kind, w, x, y, rows, cols, first)?;
            let (x, y) = (x + (first * cols * 4) as u64, y + (first * rows * 4) as u64);
            return self.matvec_few(kind, w, x, y, rows, cols, tokens - first);
        }
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::I32(tokens as i32),
        ];
        let kernel = &self.kernels.matvec_few[tokens.max(2) - 2][Self::kind_index(kind)?];
        // FEW_ROWS (2) rows a threadgroup.
        self.run(kernel, ((rows as u32).div_ceil(2), 1), 128, &args)
    }

    /// One token's `y = gelu(W x) × u` in one dispatch (tanh-GELU, as [`Gpu::geglu`]):
    /// what `matvec` and `geglu` do, with one dependent step fewer. Ok(false), having
    /// done nothing, for a type without the kernel (F32, F16).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_gelu(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        u: CuDevicePtr,
    ) -> Result<bool> {
        let Some(kernel) = &self.kernels.matvec_gelu[Self::kind_index(kind)?] else {
            return Ok(false);
        };
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::Ptr(u),
        ];
        self.run(kernel, ((rows as u32).div_ceil(2), 1), 128, &args)?;
        Ok(true)
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
    /// Whether a mixture of experts' matrices of this type have a kernel (`moe_matvec`).
    pub fn supports_experts(kind: TensorType) -> bool {
        matches!(kind, TensorType::Q8_0 | TensorType::Q6_K)
    }

    /// For each of `tokens` rows of `experts` scores: the `used` highest into `idx`, and
    /// their shares (a softmax over them, times each expert's `scale`) into `share`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_pick(
        &self,
        scores: CuDevicePtr,
        scale: CuDevicePtr,
        idx: CuDevicePtr,
        share: CuDevicePtr,
        experts: usize,
        used: usize,
        tokens: usize,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(scores),
            Arg::Ptr(scale),
            Arg::Ptr(idx),
            Arg::Ptr(share),
            Arg::I32(experts as i32),
            Arg::I32(used as i32),
        ];
        assert!(used <= 32, "moe_pick holds up to 32 picks (MOE_MAX_USED)");
        self.run(&self.kernels.moe_pick, (tokens as u32, 1), 128, &args)
    }

    /// `y = W_e x` for each of `slots` slots (a token's turn with one of its `used`
    /// experts): `w` holds every expert's `rows` × `cols` matrix one after another, and
    /// `idx` each slot's expert. `x` is the token's (or, with `x_per_slot`, the slot's
    /// own), `x_stride` numbers apart; `y` the slot's, `y_stride` apart.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_matvec(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        idx: CuDevicePtr,
        used: usize,
        x_stride: usize,
        x_per_slot: bool,
        y_stride: usize,
        slots: usize,
    ) -> Result<()> {
        let kernel = match kind {
            TensorType::Q8_0 => &self.kernels.moe_matvec_q8_0,
            TensorType::Q6_K => &self.kernels.moe_matvec_q6_k,
            other => return Err(Error::Unsupported(other)),
        };
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::Ptr(idx),
            Arg::I32(used as i32),
            Arg::I32(x_stride as i32),
            Arg::I32(i32::from(x_per_slot)),
            Arg::I32(y_stride as i32),
        ];
        // A SIMD group takes 2 rows (MV_ROWS), ROWS_PER_BLOCK groups a threadgroup.
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK * 2);
        self.run(kernel, (grid, slots as u32), ROWS_PER_BLOCK * 32, &args)
    }

    /// How many tiles `moe_group` may cut `slots` slots of `experts` experts into: the
    /// full ones, and a part-filled one an expert. (Counted for the smallest tile so the
    /// buffer fits every tile size.)
    pub fn moe_tiles(slots: usize, experts: usize) -> usize {
        slots / MOE_TC_SMALL_SLOTS + experts
    }

    /// Whether a batch's experts can run on the neural accelerators (`moe_matmul_fast`),
    /// for weights of this type.
    pub fn moe_fast(&self, kind: TensorType) -> bool {
        self.kernels.moe_matmul_tc.is_some() && KINDS.contains(&kind)
    }

    /// The slots in one of `moe_group`'s tiles: for `moe_matmul_fast` (16 for short/sparse
    /// batches where each expert averages ≤ 8 slots, 64 for dense prompt batches) or
    /// `moe_matmul`.
    fn moe_tile_slots(fast: bool, slots: usize, experts: usize) -> usize {
        if !fast {
            MOE_TILE_SLOTS
        } else if slots <= experts * 8 {
            MOE_TC_SMALL_SLOTS
        } else {
            MOE_TC_SLOTS
        }
    }

    /// A batch's `slots` slots sorted by their expert into `order` (an i32 each, and
    /// where each slot went into `place`), and cut into `tiles` (three i32s each,
    /// `moe_tiles` of them) for `moe_matmul`, or for `moe_matmul_fast` (`fast`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_group(
        &self,
        idx: CuDevicePtr,
        order: CuDevicePtr,
        tiles: CuDevicePtr,
        place: CuDevicePtr,
        slots: usize,
        experts: usize,
        fast: bool,
    ) -> Result<()> {
        assert!(experts <= 512, "moe_group holds up to 512 experts");
        let tile = Self::moe_tile_slots(fast, slots, experts);
        let args = [
            Arg::Ptr(idx),
            Arg::Ptr(order),
            Arg::Ptr(tiles),
            Arg::I32(slots as i32),
            Arg::I32(experts as i32),
            Arg::I32((slots / tile + experts) as i32),
            Arg::I32(tile as i32),
            Arg::Ptr(place),
        ];
        self.run(&self.kernels.moe_group, (1, 1), 1024, &args)
    }

    /// The inputs of `moe_matmul_fast` into `out`, `slots` rows of `cols` f16s in
    /// `moe_group`'s order: each slot's token's row of `x` (`by_token`, a token's `used`
    /// slots sharing it), or the rows of `x` as they lie (already in that order);
    /// `x_stride` numbers apart.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gather_half(
        &self,
        x: CuDevicePtr,
        out: CuDevicePtr,
        order: CuDevicePtr,
        used: usize,
        cols: usize,
        x_stride: usize,
        by_token: bool,
        slots: usize,
    ) -> Result<()> {
        assert!(cols.is_multiple_of(4), "rows of {cols} aren't whole fours");
        let total = slots * cols / 4;
        self.run_per_number(
            &self.kernels.moe_gather_half,
            total,
            &[
                Arg::Ptr(x),
                Arg::Ptr(out),
                Arg::Ptr(order),
                Arg::I32(used as i32),
                Arg::I32(cols as i32),
                Arg::I32(x_stride as i32),
                Arg::I32(i32::from(by_token)),
                Arg::I32(total as i32),
            ],
        )
    }

    /// `moe_matmul` on the neural accelerators (see [`Gpu::moe_fast`]), with the slots'
    /// inputs and products in `moe_group`'s order: `x` from `moe_gather_half` (rows of
    /// `cols` f16s), `y` rows of `y_stride` f32s, row for row with `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_matmul_fast(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        tiles: CuDevicePtr,
        y_stride: usize,
        slots: usize,
        experts: usize,
    ) -> Result<()> {
        let tile_slots = Self::moe_tile_slots(true, slots, experts);
        let kernels = if tile_slots == MOE_TC_SMALL_SLOTS {
            self.kernels
                .moe_matmul_tc_small
                .as_ref()
                .expect("checked with moe_fast")
        } else {
            self.kernels
                .moe_matmul_tc
                .as_ref()
                .expect("checked with moe_fast")
        };
        let (tc_rows, _, tc_sg) = tensor_tile();
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::Ptr(tiles),
            Arg::I32(cols as i32),
            Arg::I32(y_stride as i32),
        ];
        let grid = (
            (rows as u32).div_ceil(tc_rows),
            (slots / tile_slots + experts) as u32,
        );
        self.run(&kernels[Self::kind_index(kind)?], grid, 32 * tc_sg, &args)
    }

    /// `moe_matvec` for a batch grouped by `moe_group`: each expert's matrix is read once
    /// for all the slots that chose it. Any weight type the prompt kernels read.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_matmul(
        &self,
        kind: TensorType,
        w: CuDevicePtr,
        x: CuDevicePtr,
        y: CuDevicePtr,
        rows: usize,
        cols: usize,
        order: CuDevicePtr,
        tiles: CuDevicePtr,
        used: usize,
        x_stride: usize,
        x_per_slot: bool,
        y_stride: usize,
        slots: usize,
        experts: usize,
    ) -> Result<()> {
        let kernel = &self.kernels.moe_matmul[Self::kind_index(kind)?];
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(x),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::Ptr(order),
            Arg::Ptr(tiles),
            Arg::I32(used as i32),
            Arg::I32(x_stride as i32),
            Arg::I32(i32::from(x_per_slot)),
            Arg::I32(y_stride as i32),
        ];
        let grid = (
            (rows as u32).div_ceil(TILE_ROWS),
            Self::moe_tiles(slots, experts) as u32,
        );
        self.run(kernel, grid, 128, &args)
    }

    /// Each slot's 2 × `width` numbers (its gate products, then its up products) into its
    /// activation, GELU(gate) × up, in the first `width`.
    pub fn moe_geglu(&self, y: CuDevicePtr, width: usize, slots: usize) -> Result<()> {
        let total = width * slots;
        self.run_per_number(
            &self.kernels.moe_geglu,
            total,
            &[Arg::Ptr(y), Arg::I32(width as i32), Arg::I32(total as i32)],
        )
    }

    /// Fused `moe_geglu` + f32->f16 conversion directly into `out` for sorted MoE slots.
    pub fn moe_geglu_half(
        &self,
        gate_up: CuDevicePtr,
        out: CuDevicePtr,
        width: usize,
        slots: usize,
    ) -> Result<()> {
        debug_assert!(width.is_multiple_of(4));
        let total4 = slots * (width / 4);
        self.run_per_number(
            &self.kernels.moe_geglu_half,
            total4,
            &[
                Arg::Ptr(gate_up),
                Arg::Ptr(out),
                Arg::I32(width as i32),
                Arg::I32(total4 as i32),
            ],
        )
    }

    /// `out` = each token's slots' `n` numbers, added up with their shares: `parts` in
    /// slot order, or with `place` in `moe_group`'s order (each slot's at its place).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_sum(
        &self,
        out: CuDevicePtr,
        parts: CuDevicePtr,
        share: CuDevicePtr,
        used: usize,
        n: usize,
        tokens: usize,
        place: Option<CuDevicePtr>,
    ) -> Result<()> {
        let total = n * tokens;
        self.run_per_number(
            &self.kernels.moe_sum,
            total,
            &[
                Arg::Ptr(out),
                Arg::Ptr(parts),
                Arg::Ptr(share),
                Arg::I32(used as i32),
                Arg::I32(n as i32),
                Arg::I32(total as i32),
                // Not read without `place`: any address a buffer starts at.
                Arg::Ptr(place.unwrap_or(share)),
                Arg::I32(i32::from(place.is_some())),
            ],
        )
    }

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
        // 512 threads: one token's norm is a step of its own on one threadgroup, and its
        // latency halved from 256 (Gemma 4 generating ~4.5% faster on the M5 Max).
        self.run(&self.kernels.residual_norm, (tokens as u32, 1), 512, &args)
    }

    /// Gets `tokens` tokens' queries, keys and values ready for attention: see
    /// [`QkvPrep`].
    pub fn qkv_prep(&self, p: &QkvPrep, tokens: usize) -> Result<()> {
        assert!(p.head_dim <= 512, "head size {} is over 512", p.head_dim);
        let kv = p.cache.is_some();
        let (k_cache, v_cache) = p.cache.unwrap_or((0, 0));
        // A prompt batch whose attention may run on attention_tcg: its queries in f16 too,
        // regrouped, saving q_group_half's pass.
        let group = p.heads / p.kv_heads.max(1);
        let ready = self.tcg_shape(p.half_cache, tokens, p.heads, p.head_dim, group);
        let q_half = match ready {
            Some(_) => self.qh_buffer(tokens * p.heads * p.head_dim * 2)?,
            None => 0,
        };
        *self.qh_ready.lock().unwrap_or_else(|e| e.into_inner()) =
            ready.map(|_| (p.q_out, tokens, p.heads, p.head_dim, group));
        let cache_mode = if Self::kv_q8(p.half_cache, p.head_dim) {
            2i32
        } else {
            p.half_cache as i32
        };
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
            Arg::I32(cache_mode),
            Arg::I32(tokens as i32),
            Arg::I32(p.ring as i32),
            Arg::Ptr(q_half),
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

    /// [`QkvPrep`] for a batch of independent chats in one dispatch: `table` points to
    /// `chats` [`BatchChat`] entries in GPU memory (each chat's own KV cache addresses,
    /// position and ring). `own` is whether this layer writes its own keys and values.
    pub fn qkv_prep_batch(
        &self,
        p: &QkvPrep,
        table: CuDevicePtr,
        own: bool,
        chats: usize,
    ) -> Result<()> {
        assert!(p.head_dim <= 512, "head size {} is over 512", p.head_dim);
        *self.qh_ready.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let cache_mode = if Self::kv_q8(p.half_cache, p.head_dim) {
            2i32
        } else {
            p.half_cache as i32
        };
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
            Arg::Ptr(table),
            Arg::Ptr(0),
            Arg::I32(p.heads as i32),
            Arg::I32(p.kv_heads as i32),
            Arg::I32(p.head_dim as i32),
            Arg::I32(p.rope_dims as i32),
            Arg::F32(p.rope_base),
            Arg::Ptr(p.rope_factors.unwrap_or(0)),
            Arg::I32(p.neox as i32),
            Arg::Ptr(0), // null pos -> read per-chat pos/keys/values/ring from `table`
            Arg::F32(p.eps),
            Arg::I32(cache_mode),
            Arg::I32(chats as i32),
            Arg::I32(0),
            Arg::Ptr(0),
        ];
        let slots = p.heads + if own { 2 * p.kv_heads } else { 0 };
        let threads = (p.head_dim.div_ceil(32) * 32).min(256) as u32;
        self.run(
            &self.kernels.qkv_prep,
            (slots as u32, (chats as u32).div_ceil(8)),
            threads,
            &args,
        )
    }

    /// One-token decode attention for `chats` independent conversations in one dispatch:
    /// `table` points to `chats` [`BatchChat`] entries (each chat's own KV cache, position
    /// and ring). Returns `false` when the shape isn't handled by the batched decode kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_batch(
        &self,
        half_cache: bool,
        q: CuDevicePtr,
        table: CuDevicePtr,
        out: CuDevicePtr,
        window: usize,
        heads: usize,
        chats: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        scale: f32,
    ) -> Result<bool> {
        let wide_idx = match head_dim {
            128 => 0,
            256 => 1,
            512 => 2,
            _ => return Ok(false),
        };
        if chats == 0 || chats > 64 || heads > 64 {
            return Ok(false);
        }
        let use_q8 = Self::kv_q8(half_cache, head_dim);
        let cache_idx = if use_q8 { 2 } else { usize::from(half_cache) };
        let base_args = [
            Arg::Ptr(q),
            Arg::Ptr(table),
            Arg::Ptr(0),
            Arg::Ptr(out),
            Arg::Ptr(0), // null pos -> read per-chat keys/values/pos/ring from `table`
            Arg::I32(window as i32),
            Arg::I32(head_dim as i32),
            Arg::I32(kv_dim as i32),
            Arg::I32(group as i32),
            Arg::F32(scale),
        ];
        let cores = self.info.multiprocessors.max(1) as u32;
        let cap_parts = ((64 * MAX_DECODE_PARTS) / (heads as u32 * chats as u32)).max(1);
        if let Some(kernel) = self.group_kernel(cache_idx, head_dim, group, heads, window) {
            let kv_heads = (heads / group) as u32;
            let parts = (3 * cores)
                .div_ceil(kv_heads * chats as u32)
                .clamp(1, cap_parts)
                .min(MAX_DECODE_PARTS);
            let mut args = base_args.to_vec();
            args.push(Arg::Ptr(self.attention_parts.ptr()));
            args.push(Arg::Ptr(self.attention_counters.ptr()));
            args.push(Arg::I32(64));
            self.run_3d(kernel, (kv_heads, parts, chats as u32), 256, 0, &args)?;
            return Ok(true);
        }
        let spread: u32 = if head_dim <= 128 { 4 } else { 2 };
        let threadgroups = heads as u32;
        let parts = (spread * cores)
            .div_ceil(threadgroups * chats as u32)
            .clamp(1, cap_parts)
            .min(MAX_DECODE_PARTS);
        let mut args = base_args.to_vec();
        args.push(Arg::Ptr(self.attention_parts.ptr()));
        args.push(Arg::Ptr(self.attention_counters.ptr()));
        let kernel = &self.kernels.attention_decode_wide[cache_idx][wide_idx];
        assert!(threadgroups * parts * chats as u32 <= 64 * MAX_DECODE_PARTS);
        self.run_3d(kernel, (threadgroups, parts, chats as u32), 256, 0, &args)?;
        Ok(true)
    }

    /// attention_tcg's kernel and tile (rows, positions a step, threads) for a batch of
    /// this shape, if it's one the kernel takes: more than 8 tokens over an f16 cache,
    /// heads of 128, 256 or 512, and a group that fits the tile's rows.
    fn tcg_shape(
        &self,
        half_cache: bool,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        group: usize,
    ) -> Option<(&Pso, (usize, usize, u32))> {
        let (tcg, _, _) = self.kernels.attention_tcg.as_ref()?;
        let i = [128, 256, 512].iter().position(|&d| d == head_dim)?;
        let tile = TCG_TILES[i];
        let fits = half_cache
            && !tcg_off()
            && (tokens > 8 || tcg_few())
            && heads <= 64
            && group >= 1
            && heads.is_multiple_of(group)
            && group <= tile.0;
        fits.then_some((&tcg[i], tile))
    }

    /// One of attention_tcg's scratch buffers, grown to hold `bytes`. One it outgrows is
    /// kept (a recorded graph may name it; they double, so they add up to less than the
    /// last).
    fn grown(&self, slot: &Mutex<Option<Buffer>>, bytes: usize) -> Result<CuDevicePtr> {
        let mut held = slot.lock().unwrap_or_else(|e| e.into_inner());
        if held.as_ref().is_none_or(|b| b.len_bytes() < bytes) {
            let new = self.alloc(bytes.next_power_of_two())?;
            if let Some(old) = held.replace(new) {
                self.retired
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(old);
            }
        }
        Ok(held.as_ref().unwrap().ptr())
    }

    /// attention_tcg's f16 queries (q_group_half, or qkv_prep itself).
    fn qh_buffer(&self, bytes: usize) -> Result<CuDevicePtr> {
        self.grown(&self.attention_qh, bytes)
    }

    /// Whether a prompt's attention takes a key/value head's query heads together
    /// (attention_tcg; on unless VIREO_TCG=off). For tests and timing.
    pub fn group_prompt_attention(&self, on: bool) {
        TCG_OFF.store(!on, Ordering::Relaxed);
    }

    /// The same for 1 to 8 tokens (on unless VIREO_TCG_FEW=off). For tests and timing.
    pub fn group_few_attention(&self, on: bool) {
        TCG_FEW.store(on, Ordering::Relaxed);
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
        self.attention_ring(
            half_cache, q, keys, values, out, pos, window, 0, heads, tokens, head_dim, kv_dim,
            group, scale,
        )
    }

    /// Whether `VIREO_KV_Q8=1` is set: stores KV caches (for head_dim >= 128 in Int8 mode)
    /// as Q8_0 (17/16 bytes per element, 1.88× smaller than f16).
    pub fn kv_q8_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("VIREO_KV_Q8")
                .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true"))
        })
    }

    /// Whether a cache of heads of `head_dim` is kept in Q8_0: asked for, in 8-bit mode,
    /// and the heads whole blocks of 32 (the layout and the kernels have no part blocks).
    fn kv_q8(half_cache: bool, head_dim: usize) -> bool {
        half_cache && Self::kv_q8_enabled() && head_dim >= 128 && head_dim.is_multiple_of(32)
    }

    /// Bytes per head-position for a KV cache of head dimension `head_dim`.
    pub fn kv_head_bytes(half_cache: bool, head_dim: usize) -> usize {
        if Self::kv_q8(half_cache, head_dim) {
            head_dim * 17 / 16
        } else if half_cache {
            head_dim * 2
        } else {
            head_dim * 4
        }
    }

    /// [`Gpu::attention`] over a cache that keeps a sliding-window layer's last `ring`
    /// positions only, position p at row p mod `ring` (0: every position, at its own row).
    /// The ring holds the window and a batch's new positions, and is a multiple of 64 (the
    /// tensor kernel's chunk), so no chunk runs over its end; attention_tcg, with longer
    /// chunks, takes a ring that is a multiple of its own (the sessions round to 256).
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
        assert!(
            head_dim <= 512 && head_dim.is_multiple_of(4),
            "head size {head_dim}"
        );
        // The kernels take the window in the argument's low 16 bits and the ring in its high.
        assert!(
            window < 1 << 16 && ring < 1 << 15 && ring.is_multiple_of(64),
            "window {window}, ring {ring}"
        );
        assert!(
            ring == 0 || ring >= window + tokens,
            "a ring of {ring} for {tokens} tokens"
        );
        let use_q8 = Self::kv_q8(half_cache, head_dim);
        let cache_idx = if use_q8 { 2 } else { usize::from(half_cache) };
        let wide = match head_dim {
            64 => Some(0),
            128 => Some(1),
            256 => Some(2),
            512 => Some(3),
            _ => None,
        };
        // For multi-token prompt attention on a Q8_0 cache, dequantize this layer's K and V
        // into reusable f16 scratch buffers so attention_tcg / attention_tc run at full speed.
        let (keys, values) = if use_q8 && !(tokens <= 8 && heads <= 64 && wide.is_some()) {
            let kv_heads = kv_dim / head_dim;
            let q8_row_bytes = kv_heads * (head_dim * 17 / 16);
            let max_rows = (self.context.bytes_from(keys) as usize / q8_row_bytes)
                .max(ring)
                .max(tokens);
            let half_bytes = max_rows * kv_dim * 2;
            let hk = self.grown(&self.kv_half_k, half_bytes)?;
            let hv = self.grown(&self.kv_half_v, half_bytes)?;
            let stride_blocks = (max_rows as u32).min(256).max(1);
            let grid = (kv_heads as u32, stride_blocks);
            for (src, dst) in [(keys, hk), (values, hv)] {
                let d_args = [
                    Arg::Ptr(src),
                    Arg::Ptr(dst),
                    Arg::Ptr(pos),
                    Arg::I32(tokens as i32),
                    Arg::I32(ring as i32),
                    Arg::I32(kv_heads as i32),
                    Arg::I32(head_dim as i32),
                    Arg::I32(window as i32),
                ];
                self.run(&self.kernels.dequant_kv_q8_to_half, grid, 256, &d_args)?;
            }
            (hk, hv)
        } else {
            (keys, values)
        };
        let args = [
            Arg::Ptr(q),
            Arg::Ptr(keys),
            Arg::Ptr(values),
            Arg::Ptr(out),
            Arg::Ptr(pos),
            Arg::I32((window | ring << 16) as i32),
            Arg::I32(head_dim as i32),
            Arg::I32(kv_dim as i32),
            Arg::I32(group as i32),
            Arg::F32(scale),
        ];
        // Up to 8 tokens (a draft's guesses, a short batch) the same way, a token a grid
        // layer: the prompt kernels' tiles cost several times as much for a few (Gemma 4
        // E2B, 2 tokens at 500: ~71 µs a layer against ~13 for one).
        if let (1..=8, true, Some(i)) = (tokens, heads <= 64, wide) {
            if !use_q8 && tcg_few() && Self::grouped_few(tokens, heads, head_dim, group, window) {
                let mut tc_args = args.to_vec();
                tc_args.push(Arg::I32(tokens as i32));
                if self.attention_grouped(
                    half_cache, q, keys, values, out, ring, heads, tokens, head_dim, kv_dim, group,
                    &tc_args,
                )? {
                    return Ok(());
                }
            }
            if let Some(kernel) = self.group_kernel(cache_idx, head_dim, group, heads, window) {
                let kv_heads = (heads / group) as u32;
                let cores = self.info.multiprocessors.max(1) as u32;
                let parts = (3 * cores)
                    .div_ceil(kv_heads * tokens as u32)
                    .clamp(1, (64 * MAX_DECODE_PARTS) / (heads as u32 * tokens as u32))
                    .min(MAX_DECODE_PARTS);
                let mut args = args.to_vec();
                args.push(Arg::Ptr(self.attention_parts.ptr()));
                args.push(Arg::Ptr(self.attention_counters.ptr()));
                // At least 64 positions a part: a short context uses fewer of them.
                args.push(Arg::I32(64));
                return self.run_3d(kernel, (kv_heads, parts, tokens as u32), 256, 0, &args);
            }
            let (j, hg) = (0, 1);
            let spread: u32 = if head_dim <= 128 { 4 } else { 2 };
            let threadgroups = (heads / hg) as u32;
            let cores = self.info.multiprocessors.max(1) as u32;
            let parts = (spread * cores)
                .div_ceil(threadgroups * tokens as u32)
                .clamp(1, MAX_DECODE_PARTS);
            let mut args = args.to_vec();
            args.push(Arg::Ptr(self.attention_parts.ptr()));
            args.push(Arg::Ptr(self.attention_counters.ptr()));
            // Heads of 128 or more: a SIMD group a position (Gemma 4 generating 5–6% faster
            // than with a lane a position on the M5 Max).
            let kernel = if i >= 1 {
                &self.kernels.attention_decode_wide[cache_idx][i - 1]
            } else {
                self.kernels.attention_decode[usize::from(half_cache)][i][j]
                    .as_ref()
                    .expect("built for one query head at a time")
            };
            // The scratch: a part's maximum, sum and output for every head and token.
            assert!(threadgroups * parts * tokens as u32 <= 64 * MAX_DECODE_PARTS);
            return self.run_3d(kernel, (threadgroups, parts, tokens as u32), 256, 0, &args);
        }
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
            if self.attention_grouped(
                half_cache, q, keys, values, out, ring, heads, tokens, head_dim, kv_dim, group,
                &args,
            )? {
                return Ok(());
            }
            if let Some(tc) = &self.kernels.attention_tc {
                let blocks = (tokens as u32).div_ceil(ATTENTION_TC_TOKENS[kernel]);
                return self.run(&tc[kernel], (heads as u32, blocks), 256, &args);
            }
            // The bigger tiles pay off only for 512-wide heads (Gemma 4's global layers);
            // for 64 and 256 the smaller ones were quicker on the M5 Max.
            if head_dim != 512 {
                let blocks = (tokens as u32).div_ceil(FLASH_TOKENS[kernel]);
                return self.run(
                    &self.kernels.attention_flash[kernel],
                    (heads as u32, blocks),
                    128,
                    &args,
                );
            }
            let blocks = (tokens as u32).div_ceil(FLASH2_TOKENS[kernel]);
            return self.run(
                &self.kernels.attention_flash2[kernel],
                (heads as u32, blocks),
                FLASH2_THREADS[kernel],
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
        self.run(one, (heads as u32, tokens as u32), 128, &args)
    }

    /// Whether 1 to 8 tokens' attention is quicker on attention_tcg than on the one-token
    /// kernels, which walk the context once a token and query head: attention_tcg and its
    /// merge cost 30-50 µs a layer whatever the count, and read the cache once. Measured
    /// on the M5 Max (µs a layer, one-token kernels → attention_tcg). Full attention from
    /// 2 tokens: Gemma 4 31B's global layers at 30K positions, 2 / 4 / 8 tokens 1,090 →
    /// 580, 2,290 → 620, 4,980 → 800; Qwen 3.8 27B 460 → 300, 1,080 → 310, 2,200 → 430
    /// (level at 2,300 positions, 10-20 slower at 200). One token only for narrow heads
    /// in a large group (Muse 30B's global layers: 260 → 80 at 30K, 90 → 50 at 8K, level
    /// at 2,300, 10 → 30 at 200); Gemma's and Qwen's read their cache at memory speed
    /// already (480 → 590, 240 → 290). A sliding window bounds the old kernels' work, so
    /// there only when tokens × heads × window × head size is large: Gemma 31B from 3
    /// tokens (80 → 50), Muse from 4 (90 → 30), not E4B's (10-20 → 40).
    fn grouped_few(
        tokens: usize,
        heads: usize,
        head_dim: usize,
        group: usize,
        window: usize,
    ) -> bool {
        if window > 0 {
            return tokens * heads * window * head_dim >= 20 << 20;
        }
        tokens >= 2 || (head_dim == 128 && group >= 8)
    }

    /// A batch's attention with a key/value head's query heads together, in longer chunks
    /// (attention_tcg), where the shape is one it takes ([`Gpu::tcg_shape`]) and the cache
    /// holds a chunk of positions (a ring whole chunks). `args` are the attention kernels'
    /// first eleven. False, having done nothing, otherwise.
    #[allow(clippy::too_many_arguments)]
    fn attention_grouped(
        &self,
        half_cache: bool,
        q: CuDevicePtr,
        keys: CuDevicePtr,
        values: CuDevicePtr,
        out: CuDevicePtr,
        ring: usize,
        heads: usize,
        tokens: usize,
        head_dim: usize,
        kv_dim: usize,
        group: usize,
        args: &[Arg],
    ) -> Result<bool> {
        let ready = self
            .qh_ready
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some((tcg, (rows, chunk, threads))) =
            self.tcg_shape(half_cache, tokens, heads, head_dim, group)
        {
            let chunk_bytes = (chunk * kv_dim * 2) as u64;
            if ring.is_multiple_of(chunk)
                && self.context.bytes_from(keys) >= chunk_bytes
                && self.context.bytes_from(values) >= chunk_bytes
            {
                let qh = self.qh_buffer(tokens * heads * head_dim * 2)?;
                if ready != Some((q, tokens, heads, head_dim, group)) {
                    let (_, prep, _) = self.kernels.attention_tcg.as_ref().unwrap();
                    let n4 = (tokens * heads * head_dim / 4) as u32;
                    self.run(
                        prep,
                        (n4.div_ceil(256), 1),
                        256,
                        &[
                            Arg::Ptr(q),
                            Arg::Ptr(qh),
                            Arg::I32(heads as i32),
                            Arg::I32(group as i32),
                            Arg::I32(head_dim as i32),
                            Arg::I32(tokens as i32),
                        ],
                    )?;
                }
                let mut args = args.to_vec();
                args[0] = Arg::Ptr(qh);
                let kv_heads = (heads / group) as u32;
                let blocks = (tokens as u32).div_ceil((rows / group) as u32);
                // Few threadgroups (a short batch, few key/value heads): the context
                // in parts, about three threadgroups a GPU core in all.
                let cores = self.info.multiprocessors.max(1) as u32;
                let parts = if kv_heads * blocks >= 2 * cores {
                    1
                } else {
                    (3 * cores).div_ceil(kv_heads * blocks).min(TCG_MAX_PARTS)
                };
                if parts == 1 {
                    args.push(Arg::Ptr(0));
                    self.run(tcg, (kv_heads, blocks), threads, &args)?;
                    return Ok(true);
                }
                let floats = (kv_heads * blocks * parts) as usize * rows * (2 + head_dim);
                let partial = self.grown(&self.attention_tcg_parts, floats * 4)?;
                args.push(Arg::Ptr(partial));
                self.run_3d(tcg, (kv_heads, blocks, parts), threads, 0, &args)?;
                let (_, _, merge) = self.kernels.attention_tcg.as_ref().unwrap();
                self.run(
                    merge,
                    // One block: only the rows its tokens fill.
                    (
                        kv_heads * blocks,
                        if blocks == 1 {
                            (tokens * group).min(rows) as u32
                        } else {
                            rows as u32
                        },
                    ),
                    (head_dim as u32).min(128),
                    &[
                        Arg::Ptr(partial),
                        Arg::Ptr(out),
                        Arg::I32(tokens as i32),
                        Arg::I32(heads as i32),
                        Arg::I32(group as i32),
                        Arg::I32(head_dim as i32),
                        Arg::I32(rows as i32),
                        Arg::I32(parts as i32),
                    ],
                )?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// attention_decode_group for this layer, if it's one the kernel is quicker for: full
    /// attention (no window), heads of 256 or 512, and at least 4 query heads to a key/value
    /// head. A SIMD group holds as many of a group's heads as divide it, up to 4 for heads of
    /// 256 (Qwen 3.8 27B: 3) and 2 for 512; more spilled registers. Measured on the M5 Max,
    /// one token at 30K positions: Qwen 3.8 27B 762 → 235 µs a layer, Gemma 4 31B's global
    /// layers 1287 → 483 µs, both quicker from ~2K positions and ~10 µs slower at 64. The
    /// sliding layers (Gemma's 2 heads a key/value head, 1024 positions at most) were slower
    /// with it, and so was Muse 30B (2 key/value heads, 128 wide: its cache stays in the
    /// GPU's caches anyway).
    fn group_kernel(
        &self,
        cache_idx: usize,
        head_dim: usize,
        group: usize,
        heads: usize,
        window: usize,
    ) -> Option<&Pso> {
        if window > 0 || group < 4 || !heads.is_multiple_of(group) {
            return None;
        }
        let i = GROUP_HEADS.iter().position(|(d, _)| *d == head_dim)?;
        let h = GROUP_HEADS[i].1.iter().rev().copied().find(|&h| {
            let slices = group / h;
            group.is_multiple_of(h)
                && slices <= 8
                && (group + 8 / slices) * head_dim <= GROUP_SCRATCH
        })?;
        self.kernels.attention_decode_group[cache_idx][i][h].as_ref()
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
    /// `out[i] *= sigmoid(gate[i])` for `n` numbers (Muse Glimmer's attention gate).
    pub fn sigmoid_mul(&self, out: CuDevicePtr, gate: CuDevicePtr, n: usize) -> Result<()> {
        let args = [Arg::Ptr(out), Arg::Ptr(gate), Arg::I32(n as i32)];
        self.run_per_number(&self.kernels.sigmoid_mul, n, &args)
    }

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
        // Up to 8 tokens (a short batch, a draft's guesses): the one-token kernel, which
        // steps through them, beats the chunked prompt kernels (Qwen 3.5 4B, 2 tokens:
        // ~13 µs a layer against ~77).
        if tokens > Self::MULTI_TOKENS && n <= 128 && d.scratch != 0 {
            // A prompt in chunks of DELTA_CHUNK tokens (delta_chunk_prep and
            // delta_chunk_scan in syrinx.metal), between the norms for every token at once.
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
            self.run(
                &self.kernels.delta_chunk_prep,
                (d.v_heads as u32, chunks),
                256,
                &prep,
            )?;
            let scan = [
                Arg::Ptr(d.scratch),
                Arg::Ptr(d.state),
                Arg::Ptr(d.out),
                Arg::I32(n as i32),
                Arg::I32(tokens as i32),
            ];
            if let Some(tc) = &self.kernels.delta_chunk_scan_tc
                && n == 128
            {
                let grid = (d.v_heads as u32, (n / 32) as u32);
                self.run(tc, grid, 128, &scan)?;
            } else if n.is_multiple_of(8) {
                // On SIMD-group matrices, 8 columns of the state a threadgroup: on the M5
                // Max, Qwen 3.5 read prompts 15% faster than with delta_chunk_scan (16
                // columns a threadgroup was 1% faster than that, 32 slower).
                let grid = (d.v_heads as u32, (n / 8) as u32);
                self.run(&self.kernels.delta_chunk_scan_sg[0], grid, 128, &scan)?;
            } else {
                // 16 columns of the state a threadgroup (DSC_COLS in syrinx.metal).
                let grid = (d.v_heads as u32, (n / 16) as u32);
                self.run(&self.kernels.delta_chunk_scan, grid, 64, &scan)?;
            }
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
        if tokens > Self::MULTI_TOKENS && n <= 128 {
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
            Arg::Ptr(d.steps),
        ];
        assert!(
            d.steps == 0 || (n <= 128 && n.is_multiple_of(8)),
            "state steps are kept by delta_net_8 only"
        );
        if n <= 128 && n.is_multiple_of(8) {
            // 8 threads to a column of the state: Qwen 3.5 generating ~1.3% faster on the
            // M5 Max than with 4 (delta_net).
            return self.run(
                &self.kernels.delta_net_8,
                (d.v_heads as u32, 1),
                8 * n as u32,
                &args,
            );
        }
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

    /// Room for the default mode's input to the matrix products: `tokens` rows of up to
    /// `cols` activations in f16 (see [`Q8`]).
    pub fn alloc_q8(&self, tokens: usize, cols: usize) -> Result<Q8> {
        let half = if self.kernels.matmul_tc.is_some() && tokens > 1 {
            Some(self.alloc(tokens * cols * 2)?)
        } else {
            None
        };
        let i8 = if self.kernels.matmul_i8.is_some() && tokens >= I8_FROM {
            let n = tokens * cols.next_multiple_of(I8_K);
            Some((self.alloc(n)?, self.alloc(n / I8_K * 4)?))
        } else {
            None
        };
        Ok(Q8 {
            x: AtomicU64::new(0),
            half,
            i8,
            half_ready: AtomicBool::new(false),
            i8_ready: AtomicBool::new(false),
        })
    }

    /// Marks whether the current matrix products are in a middle-layer FFN (for
    /// `VIREO_I8=ffn`, which keeps attention and first/last layers in f16).
    pub fn set_i8_ffn(active: bool) {
        I8_FFN_ACTIVE.store(active, Ordering::Relaxed);
    }

    /// Whether a batch's products with weights of this kind go to the 8-bit kernel
    /// (`matmul_i8`): its sums are within the rounding of both sides of the f16 kernel's,
    /// not equal to them.
    pub fn whole_number_products(kind: TensorType, cols: usize, tokens: usize) -> bool {
        let on = match I8_MODE.load(Ordering::Relaxed) {
            2 => true,
            1 => I8_FFN_ACTIVE.load(Ordering::Relaxed),
            _ => false,
        };
        TENSOR_OPS.load(Ordering::Relaxed)
            && on
            && tokens >= I8_FROM
            && cols.is_multiple_of(I8_K)
            && matches!(kind, TensorType::Q6_K | TensorType::Q8_0)
    }

    /// Readies `tokens` rows of `cols` activations at `x` as the input of
    /// [`Gpu::matmul_8bit`]: a prompt's rounded to f16 for the neural accelerators, after
    /// the work already queued; one token's used as it is.
    pub fn quantize(&self, x: CuDevicePtr, q8: &Q8, tokens: usize, cols: usize) -> Result<()> {
        assert!(
            cols.is_multiple_of(32),
            "rows of {cols} aren't whole blocks of 32"
        );
        let _ = tokens;
        // The rounded copies are made by the first product that reads each.
        q8.x.store(x, Ordering::Relaxed);
        q8.half_ready.store(false, Ordering::Relaxed);
        q8.i8_ready.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Fused `act(gate) × up` directly into `q8`'s f16 buffer for `matmul_tc`, leaving
    /// `gate` untouched and skipping both the f32 store and `to_half`. Returns `true` when
    /// it ran (a prompt batch using f16 tensor products), `false` otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_quantize(
        &self,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        cols: usize,
        q8: &Q8,
        tokens: usize,
        gelu: bool,
    ) -> Result<bool> {
        let Some(half) = &q8.half else {
            return Ok(false);
        };
        if tokens <= 1 || !cols.is_multiple_of(32) || self.kernels.matmul_tc.is_none() {
            return Ok(false);
        }
        let n = tokens * cols;
        assert!(n * 2 <= half.len_bytes(), "the f16 buffer is too small");
        let n4 = n / 4;
        let kernel = if gelu {
            &self.kernels.geglu_half
        } else {
            &self.kernels.swiglu_half
        };
        let args = [
            Arg::Ptr(gate),
            Arg::Ptr(up),
            Arg::Ptr(half.ptr()),
            Arg::I32(n4 as i32),
        ];
        self.run(kernel, ((n4 as u32).div_ceil(256), 1), 256, &args)?;
        // No f32 copy of the activations exists (`gate` still holds the gate's product):
        // 0 has matmul_8bit stop rather than read one.
        q8.x.store(0, Ordering::Relaxed);
        q8.half_ready.store(true, Ordering::Relaxed);
        q8.i8_ready.store(false, Ordering::Relaxed);
        Ok(true)
    }

    /// Fused `act(gate) × up` directly into `q8`'s i8 buffer for `matmul_i8`, leaving
    /// `gate` untouched and skipping both the f32 store and `to_i8`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_quantize_i8(
        &self,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        cols: usize,
        q8: &Q8,
        tokens: usize,
        gelu: bool,
    ) -> Result<bool> {
        let (Some(i8k), Some((bytes, scales))) = (&self.kernels.matmul_i8, &q8.i8) else {
            return Ok(false);
        };
        if tokens < I8_FROM || !cols.is_multiple_of(I8_K) {
            return Ok(false);
        }
        let n = tokens * cols;
        assert!(n <= bytes.len_bytes(), "the 8-bit buffer is too small");
        let kernel = if gelu { &i8k.geglu_i8 } else { &i8k.swiglu_i8 };
        let args = [
            Arg::Ptr(gate),
            Arg::Ptr(up),
            Arg::Ptr(bytes.ptr()),
            Arg::Ptr(scales.ptr()),
        ];
        self.run(kernel, ((n / I8_K) as u32, 1), I8_K as u32, &args)?;
        q8.x.store(0, Ordering::Relaxed);
        q8.half_ready.store(false, Ordering::Relaxed);
        q8.i8_ready.store(true, Ordering::Relaxed);
        Ok(true)
    }

    /// What [`Gpu::matmul_8bit`] says when asked for a product that reads the f32
    /// activations after [`Gpu::gated_quantize`], which leaves only the f16 copy.
    const NO_F32: &'static str =
        "a product that reads f32 activations after gated_quantize, which made only f16";

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
        if let (Some(i8k), Some((bytes, scales))) = (&self.kernels.matmul_i8, &q8.i8)
            && Self::whole_number_products(kind, cols, tokens)
        {
            let n = tokens * cols;
            if !q8.i8_ready.swap(true, Ordering::Relaxed) {
                assert!(x != 0, "{}", Self::NO_F32);
                assert!(n <= bytes.len_bytes(), "the 8-bit buffer is too small");
                let args = [Arg::Ptr(x), Arg::Ptr(bytes.ptr()), Arg::Ptr(scales.ptr())];
                self.run(&i8k.to_i8, ((n / I8_K) as u32, 1), I8_K as u32, &args)?;
            }
            let wide = tokens >= WIDE_TOKENS;
            let (kernel, tok_tile, threads) = match (kind == TensorType::Q6_K, wide) {
                (true, true) => (&i8k.wide_q6_k, I8_WIDE_TOKENS, 256),
                (false, true) => (&i8k.wide_q8_0, I8_WIDE_TOKENS, 256),
                (true, false) => (&i8k.q6_k, I8_TOKENS, 128),
                (false, false) => (&i8k.q8_0, I8_TOKENS, 128),
            };
            let grid = (
                (rows as u32).div_ceil(I8_ROWS),
                (tokens as u32).div_ceil(tok_tile),
            );
            let args = [
                Arg::Ptr(w),
                Arg::Ptr(bytes.ptr()),
                Arg::Ptr(y),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                Arg::I32(tokens as i32),
                Arg::Ptr(scales.ptr()),
            ];
            return self.run(kernel, grid, threads, &args);
        }
        let (kernels, half) = match (&self.kernels.matmul_tc, &q8.half) {
            (Some((k, _)), Some(half)) if tokens > 1 && cols.is_multiple_of(32) => (k, half),
            _ => {
                assert!(x != 0, "{}", Self::NO_F32);
                return self.matmul(kind, w, x, y, rows, cols, tokens);
            }
        };
        if !q8.half_ready.swap(true, Ordering::Relaxed)
            && let Some((_, to_half)) = &self.kernels.matmul_tc
        {
            assert!(x != 0, "{}", Self::NO_F32);
            let n = tokens * cols;
            assert!(n * 2 <= half.len_bytes(), "the f16 buffer is too small");
            let args = [Arg::Ptr(x), Arg::Ptr(half.ptr()), Arg::I32(n as i32)];
            self.run(to_half, ((n as u32 / 4).div_ceil(256), 1), 256, &args)?;
        }
        let (tc_rows, tc_tokens, tc_sg) = tensor_tile();
        let (kernels, tc_tokens) =
            match (&self.kernels.matmul_tc_wide, &self.kernels.matmul_tc_small) {
                (Some(wide), _) if tokens >= WIDE_TOKENS => (wide, TC_WIDE),
                // 129-192: one tile of 192, not two of 128; 97-128 keep the 128-token one.
                (_, Some(small))
                    if tokens <= SMALL_TILES[4] || (tokens > 128 && tokens <= SMALL_TILES[5]) =>
                {
                    let i = SMALL_TILES.iter().position(|&t| tokens <= t).unwrap();
                    (&small[i], SMALL_TILES[i] as u32)
                }
                _ => (kernels, tc_tokens),
            };
        let tiles = (rows as u32).div_ceil(tc_rows) * (tokens as u32).div_ceil(tc_tokens);
        let splits = if tokens <= SPLIT_TOKENS {
            let by_room = MATMUL_PARTS_BYTES / (rows * tokens * 4);
            (SPLIT_TARGET.div_ceil(tiles) as usize)
                .min(cols / SPLIT_COLS)
                .min(MAX_SPLITS)
                .min(by_room)
                .max(1)
        } else {
            1
        };
        let grid = (
            (rows as u32).div_ceil(tc_rows),
            (tokens as u32).div_ceil(tc_tokens),
            splits as u32,
        );
        let args = [
            Arg::Ptr(w),
            Arg::Ptr(half.ptr()),
            Arg::Ptr(y),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::I32(tokens as i32),
            Arg::I32(splits as i32),
            Arg::Ptr(self.matmul_parts.ptr()),
        ];
        self.run_3d(
            &kernels[Self::kind_index(kind)?],
            grid,
            32 * tc_sg,
            0,
            &args,
        )?;
        if splits > 1 {
            let n = rows * tokens;
            let sum = self
                .kernels
                .sum_parts
                .as_ref()
                .expect("built with matmul_tc");
            let args = [
                Arg::Ptr(self.matmul_parts.ptr()),
                Arg::Ptr(y),
                Arg::I32(n as i32),
                Arg::I32(splits as i32),
            ];
            self.run(sum, ((n as u32).div_ceil(256), 1), 256, &args)?;
        }
        Ok(())
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
        if let Some((gpu, wall, count)) = state.timing {
            eprintln!(
                "{count} command buffers: {:.1} ms on the GPU, {:.1} ms from commit to done",
                gpu * 1e3,
                wall * 1e3
            );
        }
        let Some(profile) = state.profile.as_ref() else {
            return;
        };
        let total: f64 = profile.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = profile.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        eprintln!(
            "GPU time by kernel ({:.1} ms in all, each dispatch on its own):",
            total * 1e3
        );
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
    /// The cache keeps the last `ring` positions only (see [`Gpu::attention_ring`]); 0 for
    /// every position.
    pub ring: usize,
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
    /// For up to [`Gpu::MULTI_TOKENS`] tokens, room for `tokens` whole states (0 for
    /// none): the state before the batch, then after each of its tokens but the last.
    pub steps: CuDevicePtr,
}

/// Tokens per chunk of a prompt's DeltaNet (DC in syrinx.metal: 32, where CUDA's is 64, so
/// the prep's threadgroup memory fits Metal's 32 KB).
const DELTA_CHUNK: usize = 32;

/// The default mode's input to the matrix products ([`Gpu::quantize`]): where the f32
/// activations are (one token's are read as they are), and for a prompt, room for them
/// rounded to f16, as CUDA's 8-bit copy.
pub struct Q8 {
    x: AtomicU64,
    half: Option<Buffer>,
    /// For matmul_i8: the activations in 8 bits and a scale for every I8_K of a token's.
    i8: Option<(Buffer, Buffer)>,
    /// Whether the f16 and the 8-bit copies are of the activations last noted: each is
    /// made when a product first asks for it.
    half_ready: AtomicBool,
    i8_ready: AtomicBool,
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
