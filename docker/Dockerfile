# syntax=docker/dockerfile:1
FROM rust:slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/network-auth ./crates/network-auth
COPY crates/llama-stage ./crates/llama-stage
COPY src ./src
COPY ui ./ui
ENV CARGO_BUILD_JOBS=2
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/usr/local/cargo/git --mount=type=cache,target=/app/target \
    cargo build --release --locked && cp target/release/sangama /usr/local/bin/sangama

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server python3 ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /bin/sh --password x sangama \
    && mkdir -p /run/sshd /app /model /credentials /reports \
    && chown sangama:sangama /app /reports
# The account has no usable password hash; sshd accepts only the isolated test keys.
COPY --from=build /usr/local/bin/sangama /usr/local/bin/sangama
COPY docker /app/docker
USER 10001:10001
WORKDIR /app
ENV RAYON_NUM_THREADS=2
ENTRYPOINT ["python3", "/app/docker/entrypoint.py"]
