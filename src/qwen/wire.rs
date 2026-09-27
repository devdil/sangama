use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub protocol: u32,
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
    pub fn encode(&self) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(&self.header)?;
        ensure!(json.len() <= MAX_HEADER_BYTES, "header too large");
        ensure!(
            self.values.len() <= (MAX_FRAME_BYTES - 4 - json.len()) / 4,
            "frame too large"
        );
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "non-finite tensor"
        );
        let mut bytes = Vec::with_capacity(4 + json.len() + self.values.len() * 4);
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&json);
        for value in &self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            (4..=MAX_FRAME_BYTES).contains(&bytes.len()),
            "invalid frame size"
        );
        let len = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
        ensure!(
            len <= MAX_HEADER_BYTES && len <= bytes.len() - 4,
            "invalid header length"
        );
        let header: Header = serde_json::from_slice(&bytes[4..4 + len])?;
        ensure!(
            (bytes.len() - 4 - len).is_multiple_of(4),
            "truncated f32 payload"
        );
        let values: Vec<_> = bytes[4 + len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        ensure!(values.iter().all(|v| v.is_finite()), "non-finite tensor");
        Ok(Self { header, values })
    }
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
        bytes.pop();
        assert!(Frame::decode(&bytes).is_err());
        assert!(Frame::decode(&[255; 4]).is_err());
        let mut frame = frame;
        frame.values[0] = f32::NAN;
        assert!(frame.encode().is_err());
    }
}
