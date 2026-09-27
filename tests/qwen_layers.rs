use candle::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use p2p_inference::qwen::model::ShardedModel;

#[test]
fn qwen_shards_match_upstream_prefill_decode_and_cache_reset() {
    // Tiny random-weight architecture test, separate from downloaded-checkpoint validation.
    let cfg = Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: 64,
        max_window_layers: 4,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    };
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
    let mut reference = ModelForCausalLM::new(&cfg, vb.clone()).unwrap();
    let mut a = ShardedModel::new(&cfg, vb.clone(), 0, 2).unwrap();
    let mut b = ShardedModel::new(&cfg, vb, 2, 4).unwrap();
    for _ in 0..2 {
        let mut position = 0;
        for input in [vec![1_u32, 2, 3, 4], vec![5], vec![6]] {
            let tensor = Tensor::new(input.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let expected = reference
                .forward(&tensor, position)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let hidden = a.forward(&input, &[], input.len(), position).unwrap();
            let actual = b.forward(&[], &hidden, input.len(), position).unwrap();
            let error = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 1e-5, "logit error {error} at position {position}");
            position += input.len();
        }
        reference.clear_kv_cache();
        a.clear();
        b.clear();
    }
}
