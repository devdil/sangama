//! One llama.cpp layer-range stage. The first stage takes tokens; later stages take the
//! previous stage's hidden states. A non-final stage returns hidden states for every
//! position; the final stage returns the logits of the last position.
use std::{
    ffi::{CStr, CString, c_char, c_int},
    path::Path,
    sync::{Mutex, Once},
};

#[repr(C)]
struct RawStage {
    _private: [u8; 0],
}
#[repr(C)]
struct RawContext {
    _private: [u8; 0],
}
#[repr(C)]
struct RawMtp {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn sg_init(verbose: c_int);
    fn sg_gpu_memory(
        free: *mut usize,
        total: *mut usize,
        name: *mut c_char,
        name_len: usize,
    ) -> c_int;
    fn sg_stage_open(
        path: *const c_char,
        il_beg: c_int,
        il_end: c_int,
        n_gpu_layers: c_int,
        n_ctx: c_int,
        n_seq: c_int,
        n_rollback: c_int,
        n_threads: c_int,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut RawStage;
    fn sg_stage_n_rollback(stage: *const RawStage) -> c_int;
    fn sg_stage_rollback(stage: *mut RawStage, seq: c_int, pos: c_int) -> c_int;
    fn sg_stage_n_layer(stage: *const RawStage) -> c_int;
    fn sg_stage_n_embd(stage: *const RawStage) -> c_int;
    fn sg_stage_n_vocab(stage: *const RawStage) -> c_int;
    fn sg_stage_n_seq(stage: *const RawStage) -> c_int;
    fn sg_stage_ftype(stage: *const RawStage) -> c_int;
    fn sg_stage_architecture(stage: *const RawStage, buf: *mut c_char, len: usize) -> c_int;
    fn sg_stage_decode(
        stage: *mut RawStage,
        seq: c_int,
        tokens: *const i32,
        hidden: *const f32,
        n: c_int,
        pos: c_int,
        out: *mut f32,
        out_len: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn sg_stage_decode_greedy(
        stage: *mut RawStage,
        seq: c_int,
        tokens: *const i32,
        hidden: *const f32,
        n: c_int,
        pos: c_int,
        ids: *mut i32,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn sg_stage_decode_many(
        stage: *mut RawStage,
        n_items: c_int,
        seqs: *const c_int,
        ns: *const c_int,
        poss: *const c_int,
        tokens: *const i32,
        hidden: *const f32,
        out: *mut f32,
        out_len: usize,
        ids: *mut i32,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn sg_stage_state_size(stage: *mut RawStage, seq: c_int) -> usize;
    fn sg_stage_state_save(stage: *mut RawStage, seq: c_int, buf: *mut u8, len: usize) -> usize;
    fn sg_stage_state_load(stage: *mut RawStage, seq: c_int, buf: *const u8, len: usize) -> usize;
    fn sg_stage_clear(stage: *mut RawStage, seq: c_int);
    fn sg_stage_context(stage: *mut RawStage) -> *mut RawContext;
    fn sg_stage_output_all(stage: *mut RawStage, on: c_int);
    fn sg_mtp_open(
        target: *mut RawContext,
        path: *const c_char,
        n_gpu_layers: c_int,
        n_ctx: c_int,
        n_seq: c_int,
        n_threads: c_int,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut RawMtp;
    fn sg_mtp_step_many(
        mtp: *mut RawMtp,
        target: *mut RawContext,
        n_items: c_int,
        seq: *const c_int,
        row0: *const c_int,
        keep: *const c_int,
        pos: *const c_int,
        next: *const i32,
        n_draft: *const c_int,
        tokens: *const i32,
        p_min: f32,
        max_draft: c_int,
        drafts: *mut i32,
        counts: *mut c_int,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn sg_mtp_clear(mtp: *mut RawMtp, seq: c_int);
    fn sg_mtp_free(mtp: *mut RawMtp);
    fn sg_stage_free(stage: *mut RawStage);
}

#[derive(Debug)]
pub struct Error(String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: called once per process before any other llama.cpp use.
        unsafe { sg_init(std::env::var_os("SANGAMA_LLAMA_LOG").is_some() as c_int) }
    });
}

fn message(buffer: &[c_char]) -> String {
    // SAFETY: the shim always writes a NUL-terminated string within the buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Free and total bytes of the first GPU llama.cpp can use, with its device name.
pub fn gpu_memory() -> Option<(u64, u64, String)> {
    init();
    let (mut free, mut total) = (0usize, 0usize);
    let mut name = [0 as c_char; 128];
    // SAFETY: valid out-pointers and buffer length.
    let found = unsafe { sg_gpu_memory(&mut free, &mut total, name.as_mut_ptr(), name.len()) };
    (found != 0).then(|| (free as u64, total as u64, message(&name)))
}

/// What `forward_many` returns for all its items, in item order.
pub enum Batched {
    /// A non-final stage: every position's hidden state.
    Hidden(Vec<f32>),
    /// The final stage: the greedy token after every position.
    Tokens(Vec<u32>),
}

pub struct Options {
    /// Offload all layers to the GPU backend; false runs on the CPU.
    pub gpu: bool,
    /// Context length of each sequence.
    pub context: usize,
    /// Independent sequences (sessions) the stage can hold at once, each with its own cache.
    pub slots: usize,
    /// Positions a sequence can be rewound without a saved state, e.g. rejected drafts. Costs
    /// one more recurrent state per slot for each position; models without support ignore it.
    pub rollback: usize,
    pub threads: usize,
}

pub struct Stage {
    raw: *mut RawStage,
    start: usize,
    end: usize,
    layers: usize,
    embd: usize,
    vocab: usize,
    slots: usize,
    /// The model's MTP head, on a final stage that drafts tokens.
    mtp: *mut RawMtp,
}

// SAFETY: a Stage owns its llama.cpp model and context; &mut self serialises every call.
unsafe impl Send for Stage {}

impl Stage {
    /// Loads layers `start..end` of a GGUF model. Weights outside the range are not loaded.
    pub fn open(path: &Path, start: usize, end: usize, options: &Options) -> Result<Self> {
        init();
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error("model path contains NUL".into()))?;
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        let mut err = [0 as c_char; 256];
        // Opening sets process-wide environment variables that llama.cpp reads.
        static OPEN: Mutex<()> = Mutex::new(());
        let _open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: valid NUL-terminated path and error buffer.
        let raw = unsafe {
            sg_stage_open(
                path.as_ptr(),
                int(start)?,
                int(end)?,
                if options.gpu { 999 } else { 0 },
                int(options.context)?,
                int(options.slots)?,
                int(options.rollback)?,
                int(options.threads.max(1))?,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if raw.is_null() {
            return Err(Error(message(&err)));
        }
        // SAFETY: raw is a live stage.
        let (layers, embd, vocab, slots) = unsafe {
            (
                sg_stage_n_layer(raw) as usize,
                sg_stage_n_embd(raw) as usize,
                sg_stage_n_vocab(raw) as usize,
                sg_stage_n_seq(raw) as usize,
            )
        };
        Ok(Self {
            raw,
            start,
            end,
            layers,
            embd,
            vocab,
            slots,
            mtp: std::ptr::null_mut(),
        })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn hidden_size(&self) -> usize {
        self.embd
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
    pub fn slots(&self) -> usize {
        self.slots
    }
    /// Positions `rollback` can rewind; 0 if this model needs `save_state`/`load_state`.
    pub fn rollback_depth(&self) -> usize {
        // SAFETY: raw is a live stage.
        unsafe { sg_stage_n_rollback(self.raw) as usize }
    }

    /// Rewinds a slot to `position`, discarding the positions from there on that the last
    /// `forward` or `greedy` call decoded. No state is copied and nothing is decoded again.
    pub fn rollback(&mut self, slot: usize, position: usize) -> Result<()> {
        let seq = self.seq(slot)?;
        let pos = c_int::try_from(position).map_err(|_| Error("parameter too large".into()))?;
        // SAFETY: raw is a live stage and seq is in range.
        if unsafe { sg_stage_rollback(self.raw, seq, pos) } == 0 {
            return Err(Error(format!(
                "cannot rewind slot {slot} to position {position} without a saved state"
            )));
        }
        Ok(())
    }
    pub fn is_first(&self) -> bool {
        self.start == 0
    }
    pub fn is_last(&self) -> bool {
        self.end == self.layers
    }

    /// GGUF `general.architecture`, e.g. "qwen2".
    pub fn architecture(&self) -> String {
        let mut buf = [0 as c_char; 64];
        // SAFETY: valid buffer; the shim NUL-terminates or returns -1.
        let n = unsafe { sg_stage_architecture(self.raw, buf.as_mut_ptr(), buf.len()) };
        if n < 0 { String::new() } else { message(&buf) }
    }

    /// Weight precision from the GGUF file type, e.g. "f32" or "q4_k_m".
    pub fn precision(&self) -> String {
        // SAFETY: raw is a live stage.
        let ftype = unsafe { sg_stage_ftype(self.raw) };
        match ftype {
            0 => "f32".into(),
            1 => "f16".into(),
            2 => "q4_0".into(),
            7 => "q8_0".into(),
            15 => "q4_k_m".into(),
            17 => "q5_k_m".into(),
            18 => "q6_k".into(),
            32 => "bf16".into(),
            other => format!("gguf-ftype-{other}"),
        }
    }

    fn seq(&self, slot: usize) -> Result<c_int> {
        if slot >= self.slots {
            return Err(Error(format!("slot {slot} outside 0-{}", self.slots - 1)));
        }
        Ok(slot as c_int)
    }

    /// Checks a stage's input and converts tokens for llama.cpp.
    fn input(&self, tokens: &[u32], hidden: &[f32], seq_len: usize) -> Result<Vec<i32>> {
        if seq_len == 0 {
            return Err(Error("empty sequence".into()));
        }
        if self.is_first() {
            if tokens.len() != seq_len {
                return Err(Error("first stage needs one token per position".into()));
            }
            tokens
                .iter()
                .map(|&t| i32::try_from(t).map_err(|_| Error("token out of range".into())))
                .collect::<Result<_>>()
        } else {
            if hidden.len() != seq_len * self.embd {
                return Err(Error("hidden state size mismatch".into()));
            }
            Ok(Vec::new())
        }
    }

    /// Runs `seq_len` positions of `slot`'s sequence from `position`. The first stage reads
    /// `tokens`, later stages read `hidden` (`seq_len * hidden_size` values).
    pub fn forward(
        &mut self,
        slot: usize,
        tokens: &[u32],
        hidden: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        let seq = self.seq(slot)?;
        let tokens = self.input(tokens, hidden, seq_len)?;
        let mut out = vec![
            0f32;
            if self.is_last() {
                self.vocab
            } else {
                seq_len * self.embd
            }
        ];
        let mut err = [0 as c_char; 256];
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        // SAFETY: input and output buffers match the sizes the shim checks against.
        let rc = unsafe {
            sg_stage_decode(
                self.raw,
                seq,
                if self.is_first() {
                    tokens.as_ptr()
                } else {
                    std::ptr::null()
                },
                if self.is_first() {
                    std::ptr::null()
                } else {
                    hidden.as_ptr()
                },
                int(seq_len)?,
                int(position)?,
                out.as_mut_ptr(),
                out.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error(message(&err)));
        }
        Ok(out)
    }

    /// Runs one frame for each of several slots in a single device call. Each item is
    /// `(slot, seq_len, position)`; `tokens` (first stage) or `hidden` (later stages) hold the
    /// items' inputs one after another. A non-final stage returns every position's hidden
    /// state in that order; the final stage returns the greedy token after every position.
    pub fn forward_many(
        &mut self,
        items: &[(usize, usize, usize)],
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<Batched> {
        let total: usize = items.iter().map(|i| i.1).sum();
        if items.is_empty() || total == 0 {
            return Err(Error("empty batch".into()));
        }
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        let tokens = self.input(tokens, hidden, total)?;
        let (mut seqs, mut ns, mut poss) = (Vec::new(), Vec::new(), Vec::new());
        for &(slot, n, position) in items {
            seqs.push(self.seq(slot)?);
            ns.push(int(n)?);
            poss.push(int(position)?);
        }
        let last = self.is_last();
        let mut out = vec![0f32; if last { 0 } else { total * self.embd }];
        let mut ids = vec![0i32; if last { total } else { 0 }];
        let mut err = [0 as c_char; 256];
        // SAFETY: the arrays have one entry per item and the buffers match the sizes the shim checks.
        let rc = unsafe {
            sg_stage_decode_many(
                self.raw,
                int(items.len())?,
                seqs.as_ptr(),
                ns.as_ptr(),
                poss.as_ptr(),
                if self.is_first() {
                    tokens.as_ptr()
                } else {
                    std::ptr::null()
                },
                if self.is_first() {
                    std::ptr::null()
                } else {
                    hidden.as_ptr()
                },
                if last {
                    std::ptr::null_mut()
                } else {
                    out.as_mut_ptr()
                },
                out.len(),
                if last {
                    ids.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error(message(&err)));
        }
        Ok(if last {
            Batched::Tokens(ids.into_iter().map(|id| id as u32).collect())
        } else {
            Batched::Hidden(out)
        })
    }

    /// Final stage only: the greedy next token after each of the `seq_len` positions, so a
    /// client can check several drafted tokens in one pass.
    pub fn greedy(
        &mut self,
        slot: usize,
        tokens: &[u32],
        hidden: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u32>> {
        let seq = self.seq(slot)?;
        let tokens = self.input(tokens, hidden, seq_len)?;
        let mut ids = vec![0i32; seq_len];
        let mut err = [0 as c_char; 256];
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        // SAFETY: ids holds seq_len values; inputs match the sizes checked in `input`.
        let rc = unsafe {
            sg_stage_decode_greedy(
                self.raw,
                seq,
                if self.is_first() {
                    tokens.as_ptr()
                } else {
                    std::ptr::null()
                },
                if self.is_first() {
                    std::ptr::null()
                } else {
                    hidden.as_ptr()
                },
                int(seq_len)?,
                int(position)?,
                ids.as_mut_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error(message(&err)));
        }
        Ok(ids.into_iter().map(|id| id as u32).collect())
    }

    /// A slot's cached state (attention KV and recurrent state).
    pub fn save_state(&mut self, slot: usize) -> Result<Vec<u8>> {
        let seq = self.seq(slot)?;
        // SAFETY: raw is a live stage; the buffer holds the size llama.cpp reports.
        unsafe {
            let mut buf = vec![0u8; sg_stage_state_size(self.raw, seq)];
            let written = sg_stage_state_save(self.raw, seq, buf.as_mut_ptr(), buf.len());
            if written == 0 && !buf.is_empty() {
                return Err(Error("could not save the sequence state".into()));
            }
            buf.truncate(written);
            Ok(buf)
        }
    }

    /// Replaces a slot's state with one from `save_state`.
    pub fn load_state(&mut self, slot: usize, state: &[u8]) -> Result<()> {
        let seq = self.seq(slot)?;
        // SAFETY: raw is a live stage; llama.cpp reads at most state.len() bytes.
        let read = unsafe { sg_stage_state_load(self.raw, seq, state.as_ptr(), state.len()) };
        if read == 0 {
            return Err(Error("could not restore the sequence state".into()));
        }
        Ok(())
    }

    /// Drops a slot's cache so a new session can start there at position zero.
    pub fn clear(&mut self, slot: usize) {
        if let Ok(seq) = self.seq(slot) {
            // SAFETY: raw is a live stage, mtp is null or live, and seq is in range.
            unsafe {
                sg_stage_clear(self.raw, seq);
                if !self.mtp.is_null() {
                    sg_mtp_clear(self.mtp, seq);
                }
            }
        }
    }

    /// Loads an MTP-only GGUF of the same model onto this final stage, so it can draft.
    pub fn attach_mtp(&mut self, path: &Path, options: &Options) -> Result<()> {
        if !self.is_last() {
            return Err(Error("only the final stage drafts".into()));
        }
        if !self.mtp.is_null() {
            return Err(Error("an MTP head is already attached".into()));
        }
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error("MTP path contains NUL".into()))?;
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        let mut err = [0 as c_char; 256];
        // SAFETY: raw is a live stage; valid path and error buffer.
        let mtp = unsafe {
            sg_mtp_open(
                sg_stage_context(self.raw),
                path.as_ptr(),
                if options.gpu { 999 } else { 0 },
                int(options.context)?,
                int(self.slots)?,
                int(options.threads.max(1))?,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if mtp.is_null() {
            return Err(Error(message(&err)));
        }
        self.mtp = mtp;
        // SAFETY: raw is a live stage.
        unsafe { sg_stage_output_all(self.raw, 1) };
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        !self.mtp.is_null()
    }

    /// After a forward, greedy or `forward_many` call in which `slot`'s frame began at batch
    /// row `row` (0 unless batched) and `position`, and whose first `inputs.len()` input
    /// tokens were kept: feed them to the MTP head, then draft up to `n_draft` tokens after
    /// `next`, keeping each only while the head gives it at least `p_min`.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_step(
        &mut self,
        slot: usize,
        row: usize,
        inputs: &[u32],
        position: usize,
        next: u32,
        n_draft: usize,
        p_min: f32,
    ) -> Result<Vec<u32>> {
        let step = MtpStep {
            slot,
            row,
            inputs,
            position,
            next,
            n_draft,
        };
        Ok(self.mtp_step_many(&[step], p_min)?.remove(0))
    }

    /// `mtp_step` for one frame of each of several slots that shared a `forward_many` call,
    /// in increasing slot order. The head runs all of them in each of its device calls.
    pub fn mtp_step_many(&mut self, steps: &[MtpStep], p_min: f32) -> Result<Vec<Vec<u32>>> {
        if self.mtp.is_null() {
            return Err(Error("no MTP head attached".into()));
        }
        let int = |v: usize| c_int::try_from(v).map_err(|_| Error("parameter too large".into()));
        let token = |t: u32| i32::try_from(t).map_err(|_| Error("token out of range".into()));
        let column = |f: &dyn Fn(&MtpStep) -> usize| -> Result<Vec<c_int>> {
            steps.iter().map(|s| int(f(s))).collect()
        };
        let seq: Vec<c_int> = steps
            .iter()
            .map(|s| self.seq(s.slot))
            .collect::<Result<_>>()?;
        let row = column(&|s| s.row)?;
        let keep = column(&|s| s.inputs.len())?;
        let position = column(&|s| s.position)?;
        let n_draft = column(&|s| s.n_draft)?;
        let next: Vec<i32> = steps.iter().map(|s| token(s.next)).collect::<Result<_>>()?;
        let tokens: Vec<i32> = steps
            .iter()
            .flat_map(|s| s.inputs.iter().map(|&t| token(t)))
            .collect::<Result<_>>()?;
        let most = steps.iter().map(|s| s.n_draft).max().unwrap_or(0);
        let mut drafts = vec![0i32; steps.len() * most.max(1)];
        let mut counts = vec![0 as c_int; steps.len()];
        let mut err = [0 as c_char; 256];
        // SAFETY: live stage and head; every buffer matches the lengths passed.
        let rc = unsafe {
            sg_mtp_step_many(
                self.mtp,
                sg_stage_context(self.raw),
                int(steps.len())?,
                seq.as_ptr(),
                row.as_ptr(),
                keep.as_ptr(),
                position.as_ptr(),
                next.as_ptr(),
                n_draft.as_ptr(),
                tokens.as_ptr(),
                p_min,
                int(most)?,
                drafts.as_mut_ptr(),
                counts.as_mut_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error(message(&err)));
        }
        Ok(counts
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                drafts[i * most..i * most + n as usize]
                    .iter()
                    .map(|&d| d as u32)
                    .collect()
            })
            .collect())
    }
}

/// One slot's part of an MTP step: see `Stage::mtp_step`.
pub struct MtpStep<'a> {
    pub slot: usize,
    pub row: usize,
    pub inputs: &'a [u32],
    pub position: usize,
    pub next: u32,
    pub n_draft: usize,
}

impl Drop for Stage {
    fn drop(&mut self) {
        // SAFETY: raw came from sg_stage_open and mtp from sg_mtp_open; each is freed once,
        // the head first because it refers to the stage's context.
        unsafe {
            if !self.mtp.is_null() {
                sg_mtp_free(self.mtp);
            }
            sg_stage_free(self.raw)
        }
    }
}
