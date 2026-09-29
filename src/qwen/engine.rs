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
    pub fn forward(
        &mut self,
        tokens: &[u32],
        values: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Engine::Candle(model) => Ok(model.forward(tokens, values, seq_len, position)?),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.forward(tokens, values, seq_len, position)?),
        }
    }

    /// Final stage: the greedy next token after each position, to verify drafted tokens.
    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn greedy(
        &mut self,
        tokens: &[u32],
        values: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u32>> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.greedy(tokens, values, seq_len, position)?),
        }
    }

    /// The session's cached state, to roll back rejected drafts.
    pub fn save_state(&mut self) -> Result<Vec<u8>> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.save_state()?),
        }
    }

    #[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
    pub fn load_state(&mut self, state: &[u8]) -> Result<()> {
        match self {
            Engine::Candle(_) => anyhow::bail!("speculative decoding needs the llama.cpp engine"),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => Ok(stage.load_state(state)?),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Engine::Candle(model) => model.clear(),
            #[cfg(feature = "llamacpp")]
            Engine::LlamaCpp(stage) => stage.clear(),
        }
    }
}
