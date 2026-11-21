//! Inkling 2 on the GPU where there are no `ink_` kernels yet (CUDA): the Mac model's
//! types, with the same methods, so the server and CLI build everywhere. Loading says the
//! model runs on the Mac's GPU (or the CPU) for now; nothing else can be reached.

use super::inkling2::Config;
use super::{Error, Logits, Precision, Result};
use crate::GgufFile;
use crate::cuda::Gpu;
use std::marker::PhantomData;

pub const BATCH: usize = 512;
pub const REWIND: usize = 2048;

/// No value of it can be made: what makes the methods below unreachable.
enum Never {}

pub struct Inkling2Cuda<'a> {
    pub config: Config,
    never: Never,
    _file: PhantomData<&'a ()>,
}

pub struct Inkling2CudaSession {
    never: Never,
}

pub struct Inkling2Checkpoint {
    never: Never,
}

impl Inkling2Checkpoint {
    pub fn len(&self) -> usize {
        match self.never {}
    }

    pub fn is_empty(&self) -> bool {
        match self.never {}
    }
}

impl Inkling2CudaSession {
    pub fn len(&self) -> usize {
        match self.never {}
    }

    pub fn is_empty(&self) -> bool {
        match self.never {}
    }

    pub fn capacity(&self) -> usize {
        match self.never {}
    }

    pub fn cache_bytes_per_position(&self) -> usize {
        match self.never {}
    }
}

impl<'a> Inkling2Cuda<'a> {
    pub fn load(_file: &'a GgufFile, _gpu: Gpu) -> Result<Self> {
        Err(Error::Unsupported(
            "Inkling 2 on this GPU (it runs on the Mac's GPU, or on the CPU)".into(),
        ))
    }

    pub fn gpu(&self) -> &Gpu {
        match self.never {}
    }

    pub fn set_precision(&mut self, _precision: Precision) {
        match self.never {}
    }

    pub fn precision(&self) -> Precision {
        match self.never {}
    }

    pub fn session(&self, _capacity: usize) -> Result<Inkling2CudaSession> {
        match self.never {}
    }

    pub fn session_with_batch(
        &self,
        _capacity: usize,
        _batch: usize,
    ) -> Result<Inkling2CudaSession> {
        match self.never {}
    }

    pub fn reset(&self, _s: &mut Inkling2CudaSession) -> Result<()> {
        match self.never {}
    }

    pub fn save(
        &self,
        _s: &Inkling2CudaSession,
        _cp: &mut Option<Inkling2Checkpoint>,
    ) -> Result<()> {
        match self.never {}
    }

    pub fn restore(&self, _s: &mut Inkling2CudaSession, _cp: &Inkling2Checkpoint) -> Result<bool> {
        match self.never {}
    }

    pub fn can_restore(&self, _s: &Inkling2CudaSession, _len: usize) -> bool {
        match self.never {}
    }

    pub fn forward<'s>(
        &self,
        _s: &'s mut Inkling2CudaSession,
        _token: u32,
        _want_logits: bool,
    ) -> Result<Option<&'s [f32]>> {
        match self.never {}
    }

    pub fn forward_batch<'s>(
        &self,
        _s: &'s mut Inkling2CudaSession,
        _tokens: &[u32],
        _logits: Logits,
    ) -> Result<Option<&'s [f32]>> {
        match self.never {}
    }

    pub fn forward_batch_held(&self, _s: &mut Inkling2CudaSession, _tokens: &[u32]) -> Result<()> {
        match self.never {}
    }

    pub fn held_logits(
        &self,
        _s: &Inkling2CudaSession,
        _row: usize,
        _out: &mut [f32],
    ) -> Result<()> {
        match self.never {}
    }

    pub fn forward_greedy(&self, _s: &mut Inkling2CudaSession, _token: u32) -> Result<u32> {
        match self.never {}
    }
}
