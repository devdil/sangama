# syntax=docker/dockerfile:1
FROM rust:slim-bookworm AS build
WORKDIR /app
COPY portal/Cargo.toml portal/Cargo.lock ./
COPY portal/src ./src
COPY portal/static ./static
COPY portal/schema.sql ./schema.sql
ENV CARGO_BUILD_JOBS=1
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/app/target cargo build --release --locked && cp target/release/sangama-portal /portal
FROM debian:bookworm-slim
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sangama
COPY --from=build /portal /usr/local/bin/sangama-portal
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/sangama-portal"]
