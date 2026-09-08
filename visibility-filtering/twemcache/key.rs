use std::collections::HashSet;

use thiserror::Error;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Server {
    pub host: String,
    pub port: u16,
    pub ring_key: Option<String>,
}

impl Server {
    pub fn new(host: String, port: u16) -> Self {
        Self {
            host,
            port,
            ring_key: None,
        }
    }

    pub fn with_ring_key(host: String, port: u16, ring_key: String) -> Self {
        Self {
            host,
            port,
            ring_key: Some(ring_key),
        }
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn ring_identifier(&self) -> String {
        self.ring_key.clone().unwrap_or_else(|| self.address())
    }
}

pub type ServerSet = HashSet<Server>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KeyError {
    #[error("invalid key length")]
    Length,
    #[error("key contains whitespace")]
    Whitespace,
    #[error("key contains invalid UTF-8")]
    InvalidUtf8,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Key(Box<str>);

impl Key {
    pub fn new(bytes: Vec<u8>) -> std::result::Result<Self, KeyError> {
        if bytes.is_empty() || bytes.len() > 250 {
            return Err(KeyError::Length);
        }
        if bytes.iter().any(|&b| matches!(b, b' ' | b'\n' | b'\r')) {
            return Err(KeyError::Whitespace);
        }
        let s = String::from_utf8(bytes).map_err(|_| KeyError::InvalidUtf8)?;
        Ok(Key(s.into_boxed_str()))
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl AsRef<[u8]> for Key {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub type Value = Vec<u8>;

pub fn fnv1_hash_key(key: &Key) -> u64 {
    const FNV_32_INIT: u32 = 2166136261;
    const FNV_32_PRIME: u32 = 16777619;

    let hash = key.as_bytes().iter().fold(FNV_32_INIT, |h, &b| {
        h.wrapping_mul(FNV_32_PRIME) ^ u32::from(b)
    });
    u64::from(hash)
}
