//! Builds the pinned, patched llama.cpp (see scripts/fetch-llama-cpp.sh) as static libraries
//! and the C shim that wraps it.
use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = env::var_os("SANGAMA_LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../.tools/llama.cpp"));
    println!("cargo:rerun-if-env-changed=SANGAMA_LLAMA_CPP_DIR");
    println!(
        "cargo:rerun-if-changed={}",
        source.join(".sangama-pin").display()
    );
    println!("cargo:rerun-if-changed=src/shim.c");
    if !source.join(".sangama-pin").is_file() {
        panic!(
            "llama.cpp source not prepared at {}; run scripts/fetch-llama-cpp.sh",
            source.display()
        );
    }
    let feature =
        |name: &str| env::var_os(format!("CARGO_FEATURE_{}", name.to_uppercase())).is_some();
    let on_off = |on: bool| if on { "ON" } else { "OFF" };
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // One binary for mixed GPUs: e.g. SANGAMA_CUDA_ARCHS="86;89;120" for RTX 3090, 4090 and
    // 5090. Unset, llama.cpp builds only for the GPU on the build machine.
    println!("cargo:rerun-if-env-changed=SANGAMA_CUDA_ARCHS");
    let mut config = cmake::Config::new(&source);
    if let Ok(archs) = env::var("SANGAMA_CUDA_ARCHS") {
        config.define("CMAKE_CUDA_ARCHITECTURES", archs);
        config.define("GGML_NATIVE", "OFF");
    }
    let dst = config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_UI", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        // OpenMP would need its runtime linked separately; ggml's own thread pool is used instead.
        .define("GGML_OPENMP", "OFF")
        .define("GGML_METAL", on_off(feature("metal")))
        .define("GGML_METAL_EMBED_LIBRARY", "ON")
        .define("GGML_BLAS", "OFF")
        .define("GGML_CUDA", on_off(feature("cuda")))
        // Each worker uses one GPU, so multi-GPU all-reduce (and linking NCCL) is not needed.
        .define("GGML_CUDA_NCCL", "OFF")
        .define("GGML_VULKAN", on_off(feature("vulkan")))
        .define("GGML_HIP", on_off(feature("hip")))
        .build();

    // The shim must precede the llama.cpp archives: static linkers resolve left to right.
    cc::Build::new()
        .file("src/shim.c")
        .include(source.join("include"))
        .include(source.join("ggml/include"))
        .warnings(true)
        .compile("sangama_llama_shim");

    let mut libs = Vec::new();
    for dir in ["lib", "lib64"] {
        let dir = dst.join(dir);
        println!("cargo:rustc-link-search=native={}", dir.display());
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(lib) = name
                    .strip_prefix("lib")
                    .and_then(|n| n.strip_suffix(".a"))
                    .or_else(|| name.strip_suffix(".lib"))
                {
                    libs.push(lib.to_string());
                }
            }
        }
    }
    // Static link order: llama, then ggml, then each backend, then ggml-base last.
    let mut order = vec!["llama".to_string(), "ggml".to_string()];
    let mut backends: Vec<_> = libs
        .iter()
        .filter(|l| l.starts_with("ggml-") && l.as_str() != "ggml-base")
        .cloned()
        .collect();
    backends.sort();
    order.extend(backends);
    order.push("ggml-base".into());
    for lib in &order {
        assert!(libs.contains(lib), "llama.cpp build did not produce {lib}");
        println!("cargo:rustc-link-lib=static={lib}");
    }

    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        "linux" | "android" => println!("cargo:rustc-link-lib=stdc++"),
        _ => {}
    }
    if feature("cuda") {
        // Standard toolkit location when neither variable is set (e.g. NVIDIA's CUDA images).
        let root = env::var_os("CUDA_PATH")
            .or_else(|| env::var_os("CUDA_HOME"))
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from("/usr/local/cuda")).filter(|p| p.exists()));
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        if let Some(root) = root {
            println!(
                "cargo:rustc-link-search=native={}",
                root.join("lib64").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                root.join("lib/x64").display()
            );
        }
        for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    if feature("vulkan") {
        println!(
            "cargo:rustc-link-lib={}",
            if target_os == "windows" {
                "vulkan-1"
            } else {
                "vulkan"
            }
        );
    }
    if feature("hip") {
        let root = PathBuf::from(env::var_os("ROCM_PATH").unwrap_or_else(|| "/opt/rocm".into()));
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("lib").display()
        );
        for lib in ["amdhip64", "hipblas", "rocblas"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
