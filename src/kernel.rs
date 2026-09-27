use crate::protocol::ModelSpec;
use anyhow::{Result, ensure};
use std::time::Instant;

/// Real resident dense matrices; deterministic weights have no learned knowledge.
pub struct Shard {
    pub model: ModelSpec,
    pub start: usize,
    pub end: usize,
    weights: Vec<Vec<f32>>,
}

impl Shard {
    pub fn new(model: ModelSpec, start: usize, end: usize, budget_mib: usize) -> Result<Self> {
        model.validate()?;
        ensure!(start < end && end <= model.layers, "invalid layer range");
        let bytes = (end - start) * model.width * model.width * 4;
        ensure!(
            budget_mib > 0 && budget_mib <= 1024,
            "weight budget must be 1..=1024 MiB"
        );
        ensure!(
            bytes <= budget_mib * 1024 * 1024,
            "shard exceeds weight budget: {bytes} bytes"
        );
        let scale = 0.15 / (model.width as f32).sqrt();
        let weights = (start..end)
            .map(|layer| {
                let mut state = (layer as u64 + 1) * 0x9e3779b1;
                (0..model.width * model.width)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (((state >> 32) as u32 as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0) * scale
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            model,
            start,
            end,
            weights,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.iter().map(|x| x.len() * 4).sum()
    }

    pub fn forward(&self, mut input: Vec<f32>) -> Result<Vec<f32>> {
        ensure!(
            input.len() == self.model.width && input.iter().all(|v| v.is_finite()),
            "invalid activation"
        );
        for matrix in &self.weights {
            let output = matrix
                .chunks_exact(self.model.width)
                .zip(&input)
                .map(|(row, residual)| {
                    let value: f32 = row.iter().zip(&input).map(|(w, x)| w * x).sum();
                    (value + residual).tanh()
                })
                .collect();
            input = output;
        }
        Ok(input)
    }

    pub fn calibrate_ms(&self) -> Result<f64> {
        let input = fixture_input(self.model.width);
        self.forward(input.clone())?;
        let start = Instant::now();
        for _ in 0..5 {
            std::hint::black_box(self.forward(input.clone())?);
        }
        Ok(start.elapsed().as_secs_f64() * 1000.0 / 5.0)
    }
}

pub fn fixture_input(width: usize) -> Vec<f32> {
    (0..width).map(|i| (i as f32 * 0.17).sin() * 0.5).collect()
}
