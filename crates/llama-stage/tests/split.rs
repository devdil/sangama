//! Needs a real model: SANGAMA_TEST_GGUF=/path/to/qwen2.5-0.5b-f32.gguf. Skipped otherwise.
use sangama_llama_stage::{Options, Stage};
use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    std::env::var_os("SANGAMA_TEST_GGUF").map(PathBuf::from)
}

fn options() -> Options {
    Options {
        gpu: std::env::var_os("SANGAMA_TEST_CPU").is_none(),
        context: 1024,
        threads: 4,
    }
}

fn argmax(values: &[f32]) -> usize {
    (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

#[test]
fn split_stages_match_unsplit_generation() {
    let Some(path) = model() else {
        eprintln!("SANGAMA_TEST_GGUF not set; skipping");
        return;
    };
    // "Explain peer-to-peer computing in one short sentence." in Sangama's chat format.
    let prompt: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 840,
        20772, 14397, 4686, 78597, 24231, 304, 825, 2805, 11652, 13, 151645, 198, 151644, 77091,
        198,
    ];
    let generate = |stages: &mut [Stage]| {
        let mut out = Vec::new();
        let (mut input, mut position) = (prompt.clone(), 0);
        for _ in 0..12 {
            let mut values = Vec::new();
            for stage in stages.iter_mut() {
                values = stage
                    .forward(&input, &values, input.len(), position)
                    .unwrap();
            }
            position += input.len();
            let token = argmax(&values) as u32;
            out.push(token);
            input = vec![token];
        }
        out
    };
    let layers = {
        let full = Stage::open(&path, 0, 24, &options()).unwrap();
        assert_eq!(full.architecture(), "qwen2");
        full.layers()
    };
    let mut full = vec![Stage::open(&path, 0, layers, &options()).unwrap()];
    let reference = generate(&mut full);
    drop(full);
    let mut split = vec![
        Stage::open(&path, 0, 12, &options()).unwrap(),
        Stage::open(&path, 12, layers, &options()).unwrap(),
    ];
    assert!(split[0].is_first() && !split[0].is_last() && split[1].is_last());
    assert_eq!(generate(&mut split), reference);
    // A cleared stage restarts at position zero with the same result.
    for stage in &mut split {
        stage.clear();
    }
    assert_eq!(generate(&mut split), reference);
}

#[test]
fn rejects_bad_ranges_and_inputs() {
    let Some(path) = model() else {
        return;
    };
    assert!(Stage::open(&path, 12, 12, &options()).is_err());
    assert!(Stage::open(&path, 0, 99, &options()).is_err());
    let mut later = Stage::open(&path, 12, 24, &options()).unwrap();
    assert!(later.forward(&[1], &[0.0; 3], 1, 0).is_err());
}
