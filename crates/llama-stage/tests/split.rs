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
        slots: 4,
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
                    .forward(0, &input, &values, input.len(), position)
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
        stage.clear(0);
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
    assert!(later.forward(0, &[1], &[0.0; 3], 1, 0).is_err());
    assert!(later.forward(4, &[], &[0.0; 896], 1, 0).is_err());
}

/// Sessions in different slots, stepped in turn, each give exactly their solo output, and
/// clearing one slot leaves the others untouched.
#[test]
fn interleaved_slots_match_solo_generation() {
    let Some(path) = model() else {
        return;
    };
    let layers = Stage::open(&path, 0, 24, &options()).unwrap().layers();
    let mut stages = vec![
        Stage::open(&path, 0, 12, &options()).unwrap(),
        Stage::open(&path, 12, layers, &options()).unwrap(),
    ];
    assert_eq!(stages[0].slots(), 4);
    // Distinct prompts of different lengths: chat-format prefixes of "Explain ..." and two others.
    let prompts: Vec<Vec<u32>> = vec![
        vec![151644, 872, 198, 840, 20772, 14397, 4686, 78597, 24231, 13, 151645, 198, 151644, 77091, 198],
        vec![151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198],
        vec![151644, 872, 198, 7985, 264, 32794, 911, 279, 9396, 13, 151645, 198, 151644, 77091, 198, 785],
    ];
    struct Run {
        input: Vec<u32>,
        position: usize,
        out: Vec<u32>,
    }
    let step = |stages: &mut [Stage], slot: usize, run: &mut Run| {
        let mut values = Vec::new();
        for stage in stages.iter_mut() {
            values = stage
                .forward(slot, &run.input, &values, run.input.len(), run.position)
                .unwrap();
        }
        run.position += run.input.len();
        let token = argmax(&values) as u32;
        run.out.push(token);
        run.input = vec![token];
    };
    let fresh = |p: &Vec<u32>| Run {
        input: p.clone(),
        position: 0,
        out: Vec::new(),
    };
    let solo: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            for stage in &mut stages {
                stage.clear(0);
            }
            let mut run = fresh(p);
            for _ in 0..10 {
                step(&mut stages, 0, &mut run);
            }
            run.out
        })
        .collect();
    for stage in &mut stages {
        stage.clear(0);
    }
    // Slots 1-3, stepped round-robin. Midway, slot 0 runs and is cleared, which must not
    // disturb them.
    let mut runs: Vec<Run> = prompts.iter().map(fresh).collect();
    for i in 0..10 {
        for (k, run) in runs.iter_mut().enumerate() {
            step(&mut stages, k + 1, run);
        }
        if i == 4 {
            let mut other = fresh(&prompts[0]);
            step(&mut stages, 0, &mut other);
            for stage in &mut stages {
                stage.clear(0);
            }
        }
    }
    for (run, expected) in runs.iter().zip(&solo) {
        assert_eq!(&run.out, expected);
    }
    // A slot's saved state restores only that slot.
    let saved = stages[1].save_state(2).unwrap();
    assert!(!saved.is_empty());
    stages[1].load_state(2, &saved).unwrap();
}
