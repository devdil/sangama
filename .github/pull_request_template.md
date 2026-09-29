## Problem

<!-- What triggered this change? Link the issue if there is one. -->

## Change

<!-- The new behavior. Keep unrelated refactors in a separate PR. -->

## Validation

<!-- What passed, on which backend/hardware, and which paths remain untested. -->

- [ ] `./scripts/cargo fmt --all -- --check`
- [ ] `./scripts/cargo clippy --locked --all-targets -- -D warnings`
- [ ] `./scripts/cargo test --locked` (plus `--manifest-path` for `crates/network-auth` or `portal` if touched)
- [ ] Docs updated for configuration or wire-protocol changes
- [ ] No tokens, keys or invitation codes in code, logs or reports

## Limitations

<!-- Known gaps or follow-ups. -->
