//! Layer implementation adapted from candle-transformers 0.11.0 (Hugging Face).
//! MIT-licensed upstream code: see third_party/CANDLE-LICENSE-MIT.
//! Adaptation adds physical layer-range loading and hidden-state forwarding.
//! This backend uses floating-point weights; it is not quantized inference.

use candle::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
pub use candle_transformers::models::qwen2::Config;
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear, linear_no_bias};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

/// Only the selected decoder layers are materialized, with endpoints as needed.
pub struct ShardedModel {
    embeddings: Option<candle_nn::Embedding>,
    layers: Vec<DecoderLayer>,
    norm: Option<RmsNorm>,
    head: Option<Linear>,
    device: Device,
    hidden_size: usize,
}

impl ShardedModel {
    pub fn new(cfg: &Config, vb: VarBuilder<'_>, start: usize, end: usize) -> Result<Self> {
        let device = vb.device().clone();
        let embeddings = if start == 0 {
            Some(candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("model.embed_tokens"),
            )?)
        } else {
            None
        };
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, &device)?);
        let mut layers = Vec::new();
        for i in start..end {
            layers.push(DecoderLayer::new(
                rotary.clone(),
                cfg,
                vb.pp(format!("model.layers.{i}")),
            )?);
        }
        let (norm, head) = if end == cfg.num_hidden_layers {
            let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
            let prefix = if vb.contains_tensor("lm_head.weight") {
                "lm_head"
            } else {
                "model.embed_tokens"
            };
            let head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp(prefix))?;
            (Some(norm), Some(head))
        } else {
            (None, None)
        };
        Ok(Self {
            embeddings,
            layers,
            norm,
            head,
            device,
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        values: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        let mut xs = if let Some(embeddings) = &self.embeddings {
            let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
            embeddings.forward(&input)?
        } else {
            Tensor::from_slice(values, (1, seq_len, self.hidden_size), &self.device)?
        };
        let mask = if seq_len > 1 {
            let values: Vec<_> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_vec(values, (seq_len, seq_len), &self.device)?;
            let mask = if position > 0 {
                let prefix = Tensor::zeros((seq_len, position), DType::F32, &self.device)?;
                Tensor::cat(&[&prefix, &mask], D::Minus1)?
            } else {
                mask
            };
            Some(mask.expand((1, 1, seq_len, seq_len + position))?)
        } else {
            None
        };
        for layer in &mut self.layers {
            xs = layer.forward(&xs, mask.as_ref(), position)?;
        }
        if let (Some(norm), Some(head)) = (&self.norm, &self.head) {
            xs = xs.apply(norm)?.narrow(1, seq_len - 1, 1)?.apply(head)?;
        }
        xs.flatten_all()?.to_vec1()
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
