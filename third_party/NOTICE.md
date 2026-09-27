# Third-party notices

`src/qwen/model.rs` adapts Qwen2 layer implementation from Hugging Face Candle
(candle-transformers 0.11.0), Copyright 2023 Hugging Face, Inc., under the MIT license.
The original license is reproduced in `CANDLE-LICENSE-MIT`. Changes add layer-range loading,
explicit embedding/output endpoints, and hidden-state input/output.

Qwen checkpoint files are downloaded separately under their upstream license, saved alongside the files.
