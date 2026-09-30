//! Execution engines for one shard. Both take tokens (first shard) or F32 hidden states, and
//! return F32 hidden states or the last position's logits, so one route may mix them.
use super::model::ShardedModel;
use anyhow::Result;

/// `candle` runs the F32 checkpoint on CPU, Metal or CUDA. `llamacpp` runs an approved GGUF
/// on any backend llama.cpp was built with (`--features llamacpp-*`).
pub const ENGINES: [&str; 2] = ["candle", "llamacpp"];

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

    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn clear(&mut self, slot: usize) {
        match self {
            Engine::Candle(model) => model.clear(),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.clear(slot),
        }
    }
}
