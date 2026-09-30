//! Execution engines for one shard. Both take tokens (first shard) or F32 hidden states, and
//! return F32 hidden states or the last position's logits, so one route may mix them.
use super::model::ShardedModel;
use anyhow::Result;

/// `candle` runs the F32 checkpoint on CPU, Metal or CUDA. `llamacpp` runs an approved GGUF
/// on any backend llama.cpp was built with (`--features llamacpp-*`).
pub const ENGINES: [&str; 2] = ["candle", "llamacpp"];

/// What `Engine::forward_many` returns for all its items, in item order.
pub enum Many {
    Hidden(Vec<f32>),
    Tokens(Vec<u32>),
}

pub enum Engine {
    Candle(Box<ShardedModel>),
    #[cfg(feature = "llamacpp")]
    LlamaCpp(sangama_llama_stage::Stage),
}

impl Engine {
    /// Sessions this engine can hold at once. Candle keeps one KV cache.
    pub fn slots(&self) -> usize {
        match self {
            Engine::Candle(_) => 1,
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.slots(),
        }
    }

    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn forward(
        &mut self,
        slot: usize,
        tokens: &[u32],
        values: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Engine::Candle(model) => {
                anyhow::ensure!(slot == 0, "the candle engine has one slot");
                Ok(model.forward(tokens, values, seq_len, position)?)
            }
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.forward(slot, tokens, values, seq_len, position)?),
        }
    }

    /// Final stage: the greedy next token after each position, to verify drafted tokens.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn greedy(
        &mut self,
        slot: usize,
        tokens: &[u32],
        values: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u32>> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.greedy(slot, tokens, values, seq_len, position)?),
        }
    }

    /// A slot's cached state, to roll back rejected drafts.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn save_state(&mut self, slot: usize) -> Result<Vec<u8>> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.save_state(slot)?),
        }
    }

    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn load_state(&mut self, slot: usize, state: &[u8]) -> Result<()> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.load_state(slot, state)?),
        }
    }

    /// Positions `rollback` can rewind without a saved state; 0 means use `save_state`.
    pub fn rollback_depth(&self) -> usize {
        match self {
            Engine::Candle(_) => 0,
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.rollback_depth(),
        }
    }

    /// Rewinds a slot to `position`, discarding what the last batch decoded from there on.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn rollback(&mut self, slot: usize, position: usize) -> Result<()> {
        match self {
            Engine::Candle(_) => anyhow::bail!("the candle engine cannot rewind a session"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.rollback(slot, position)?),
        }
    }

    /// Whether `forward_many` can run frames of several slots in one device call.
    pub fn batches(&self) -> bool {
        match self {
            Engine::Candle(_) => false,
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(_) => true,
        }
    }

    /// One frame for each of several slots, `(slot, seq_len, position)`, in one device call.
    /// A non-final stage returns every position's hidden state in item order; the final
    /// stage returns the greedy token after every position.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn forward_many(
        &mut self,
        items: &[(usize, usize, usize)],
        tokens: &[u32],
        values: &[f32],
    ) -> Result<Many> {
        match self {
            Engine::Candle(_) => anyhow::bail!("the candle engine runs one session at a time"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(match stage.forward_many(items, tokens, values)? {
                sangama_llama_stage::Batched::Hidden(values) => Many::Hidden(values),
                sangama_llama_stage::Batched::Tokens(ids) => Many::Tokens(ids),
            }),
        }
    }

    /// Whether this engine can draft tokens with the model's MTP head.
    pub fn has_mtp(&self) -> bool {
        match self {
            Engine::Candle(_) => false,
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.has_mtp(),
        }
    }

    /// Final stage with an MTP head, right after a device call: for each frame in it, feed the
    /// head the kept inputs and draft tokens after the one the model chose. Frames are in
    /// increasing slot order.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn mtp_step_many(&mut self, steps: &[MtpStep], p_min: f32) -> Result<Vec<Vec<u32>>> {
        match self {
            Engine::Candle(_) => anyhow::bail!("MTP drafting needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => {
                let steps: Vec<sangama_llama_stage::MtpStep> = steps
                    .iter()
                    .map(|s| sangama_llama_stage::MtpStep {
                        slot: s.slot,
                        row: s.row,
                        inputs: s.inputs,
                        position: s.position,
                        next: s.next,
                        n_draft: s.n_draft,
                    })
                    .collect();
                Ok(stage.mtp_step_many(&steps, p_min)?)
            }
        }
    }

    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn clear(&mut self, slot: usize) {
        match self {
            Engine::Candle(model) => model.clear(),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.clear(slot),
        }
    }
}

/// One frame's part of an MTP step: its slot, the batch row and position it began at, the
/// inputs that were kept, the token the model chose after them and how many to draft.
pub struct MtpStep<'a> {
    pub slot: usize,
    pub row: usize,
    pub inputs: &'a [u32],
    pub position: usize,
    pub next: u32,
    pub n_draft: usize,
}
