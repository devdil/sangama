use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Endpoint {
    pub shard: usize,
    pub address: SocketAddr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trace {
    pub shard: usize,
    pub start: usize,
    pub end: usize,
    pub forward_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Tokens,
    Hidden,
    Logits,
    Sampled,
    /// A detached stage's reply: it computed and passed the frame on. The trace ends with its
    /// own entry; the result is collected from the last stage.
    Accepted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub protocol: u32,
    #[serde(default)]
    pub sample: bool,
    /// Each stage replies once it has passed the frame on, and the last stage keeps the
    /// sampled token for the client to collect. The token then crosses each hop once instead
    /// of unwinding back through every stage.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub detached: bool,
    /// Hidden states cross the wire as BF16, half the bytes of F32. Qwen is trained in BF16;
    /// stages still compute in F32.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bf16: bool,
    pub model_hash: String,
    pub session: String,
    pub position: usize,
    pub seq_len: usize,
    pub kind: Kind,
    pub tokens: Vec<u32>,
    pub route: Vec<Endpoint>,
    pub trace: Vec<Trace>,
}

pub struct Frame {
    pub header: Header,
    pub values: Vec<f32>,
}

impl Frame {
    /// A detached stage's acknowledgement of `sent`.
    pub fn accepts(&self, sent: &Header) -> bool {
        let h = &self.header;
        h.kind == Kind::Accepted
            && self.values.is_empty()
            && h.session == sent.session
            && h.position == sent.position
            && h.seq_len == sent.seq_len
            && h.model_hash == sent.model_hash
            && h.trace.len() == sent.trace.len() + 1
    }

    pub fn valid_output(&self, vocab_size: usize) -> bool {
        if self.header.sample {
            self.header.kind == Kind::Sampled
                && self.values.is_empty()
                && self.header.tokens.len() == 1
                && (self.header.tokens[0] as usize) < vocab_size
        } else {
            self.header.kind == Kind::Logits
                && self.values.len() == vocab_size
                && self.header.tokens.is_empty()
        }
    }

    /// Bytes per tensor value on the wire.
    fn width(header: &Header) -> usize {
        if header.bf16 && header.kind == Kind::Hidden {
            2
        } else {
            4
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(&self.header)?;
        ensure!(json.len() <= MAX_HEADER_BYTES, "header too large");
        let width = Self::width(&self.header);
        ensure!(
            self.values.len() <= (MAX_FRAME_BYTES - 4 - json.len()) / width,
            "frame too large"
        );
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "non-finite tensor"
        );
        let mut bytes = Vec::with_capacity(4 + json.len() + self.values.len() * width);
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&json);
        for value in &self.values {
            if width == 2 {
                bytes.extend_from_slice(&to_bf16(*value).to_le_bytes());
            } else {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    fn split(bytes: &[u8]) -> Result<(Header, &[u8])> {
        ensure!(
            (4..=MAX_FRAME_BYTES).contains(&bytes.len()),
            "invalid frame size"
        );
        let len = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
        ensure!(
            len <= MAX_HEADER_BYTES && len <= bytes.len() - 4,
            "invalid header length"
        );
        Ok((
            serde_json::from_slice(&bytes[4..4 + len])?,
            &bytes[4 + len..],
        ))
    }

    /// Reads only the header, for routing decisions that do not need the tensor.
    pub fn header(bytes: &[u8]) -> Result<Header> {
        Ok(Self::split(bytes)?.0)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (header, payload) = Self::split(bytes)?;
        let width = Self::width(&header);
        ensure!(payload.len().is_multiple_of(width), "truncated payload");
        let values: Vec<_> = if width == 2 {
            payload
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect()
        } else {
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        ensure!(values.iter().all(|v| v.is_finite()), "non-finite tensor");
        Ok(Self { header, values })
    }
}

/// Rounds an F32 to the nearest BF16, ties to even. Values are finite (checked by the caller).
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub async fn response(mut response: reqwest::Response) -> Result<Frame> {
    let status = response.status();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len() + chunk.len() <= MAX_FRAME_BYTES,
            "response too large"
        );
        bytes.extend_from_slice(&chunk);
    }
    ensure!(
        status.is_success(),
        "worker returned {status}: {}",
        String::from_utf8_lossy(&bytes)
    );
    Frame::decode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> Frame {
        Frame {
            header: Header {
                protocol: 1,
                sample: false,
                detached: false,
                bf16: false,
                model_hash: "a".into(),
                session: "s".into(),
                position: 0,
                seq_len: 1,
                kind: Kind::Hidden,
                tokens: vec![],
                route: vec![],
                trace: vec![],
            },
            values: vec![0.0, -0.125, 1.25],
        }
    }
    #[test]
    fn binary_frames_preserve_values_and_reject_malformed_payloads() {
        let frame = frame();
        let mut bytes = frame.encode().unwrap();
        assert_eq!(Frame::decode(&bytes).unwrap().values, frame.values);
        assert_eq!(Frame::header(&bytes).unwrap().session, "s");
        bytes.pop();
        assert!(Frame::decode(&bytes).is_err());
        assert!(Frame::decode(&[255; 4]).is_err());
        let mut frame = frame;
        frame.values[0] = f32::NAN;
        assert!(frame.encode().is_err());
    }
    #[test]
    fn bf16_hidden_states_halve_the_payload_and_round_to_nearest() {
        let mut frame = frame();
        frame.header.bf16 = true;
        frame.values = vec![0.0, -0.125, 1.25, 1.0 + 1.0 / 256.0 + 1.0 / 1024.0];
        let bytes = frame.encode().unwrap();
        let json = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(bytes.len() - 4 - json, 2 * frame.values.len());
        let decoded = Frame::decode(&bytes).unwrap().values;
        // Exact in BF16, and a value between two BF16 steps rounds to the nearer one.
        assert_eq!(decoded, vec![0.0, -0.125, 1.25, 1.0 + 1.0 / 128.0]);
        // Only hidden states are narrowed; logits stay F32.
        frame.header.kind = Kind::Logits;
        assert_eq!(
            Frame::decode(&frame.encode().unwrap()).unwrap().values,
            frame.values
        );
    }
}
