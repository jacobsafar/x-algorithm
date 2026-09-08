use std::collections::{BTreeMap, HashSet};

use md5::{Digest, Md5};

use super::error::{Result, TwemcacheError};
use super::key::{fnv1_hash_key, Key, Server};

const VNODES_PER_SERVER: usize = 160;

#[derive(Clone)]
pub struct HashRing {
    ring: BTreeMap<u64, Server>,
}

impl HashRing {
    pub fn new(servers: HashSet<Server>) -> Self {
        let mut ring = BTreeMap::new();
        let mut md = Md5::new();

        for server in &servers {
            let points = ((VNODES_PER_SERVER as f64 / 4.0) + 0.0000000001) as usize;
            let prefix = server.ring_identifier().into_bytes();

            for i in 0..points {
                md.update(&prefix);
                md.update(b"-");
                md.update(i.to_string().as_bytes());
                let hash = md.finalize_reset();

                let (words, _remainder) = hash.as_slice().as_chunks::<4>();
                for &word in words {
                    let value = u32::from_le_bytes(word) as u64;
                    ring.insert(value, server.clone());
                }
            }
        }

        Self { ring }
    }

    pub fn server(&self, key: &Key) -> Result<Server> {
        let hash = (fnv1_hash_key(key) & 0xFFFFFFFF) as u32;
        self.ring
            .range(hash as u64..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, server)| server.clone())
            .ok_or(TwemcacheError::Unavailable)
    }

    #[cfg(test)]
    pub(crate) fn update(&mut self, servers: HashSet<Server>) {
        let current: HashSet<Server> = self.ring.values().cloned().collect();
        if servers != current {
            *self = Self::new(servers);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn servers() -> HashSet<Server> {
        (0..50)
            .map(|i| {
                Server::with_ring_key(
                    format!("10.0.{}.{}", i / 250, i % 250),
                    11211,
                    format!("vf-cache-shard-{i}"),
                )
            })
            .collect()
    }

    fn keys() -> Vec<Key> {
        let ids: [u64; 26] = [
            0,
            1,
            2,
            42,
            100,
            255,
            256,
            1000,
            65535,
            65536,
            123456789,
            1000000007,
            9999999999,
            18446744073709551615,
            18446744073709551614,
            1234567890123456789,
            1111111111111111111,
            9223372036854775807,
            777,
            8675309,
            314159265358979,
            42424242,
            999999999999,
            13,
            1729,
            271828182845904,
        ];
        ids.iter()
            .map(|id| Key::new(format!("slm_{id}").into_bytes()).unwrap())
            .collect()
    }

    fn reference_shard(servers: &HashSet<Server>, key: &Key) -> String {
        let mut ring: BTreeMap<u64, String> = BTreeMap::new();

        let mut sorted: Vec<&Server> = servers.iter().collect();
        sorted.sort_by_key(|s| s.ring_identifier());

        for server in sorted {
            let id = server.ring_identifier();
            for i in 0..40usize {
                let mut md = Md5::new();
                md.update(id.as_bytes());
                md.update(b"-");
                md.update(i.to_string().as_bytes());
                let digest = md.finalize();
                for offset in [0usize, 4, 8, 12] {
                    let value = u32::from_le_bytes([
                        digest[offset],
                        digest[offset + 1],
                        digest[offset + 2],
                        digest[offset + 3],
                    ]) as u64;
                    ring.insert(value, id.clone());
                }
            }
        }

        const FNV_32_INIT: u32 = 2166136261;
        const FNV_32_PRIME: u32 = 16777619;
        let h = key.as_bytes().iter().fold(FNV_32_INIT, |h, &b| {
            h.wrapping_mul(FNV_32_PRIME) ^ u32::from(b)
        });
        let hash = (u64::from(h) & 0xFFFFFFFF) as u32;

        ring.range(hash as u64..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, id)| id.clone())
            .unwrap()
    }

    #[test]
    fn ring_matches_independent_reference() {
        let servers = servers();
        let ring = HashRing::new(servers.clone());
        for key in keys() {
            let got = ring.server(&key).unwrap().ring_identifier();
            let expected = reference_shard(&servers, &key);
            assert_eq!(
                got, expected,
                "continuum drift for {key}: ring={got} reference={expected}"
            );
        }
    }

    #[test]
    fn ring_matches_golden_snapshot() {
        const GOLDEN: &[(&str, &str)] = &[
            ("slm_0", "vf-cache-shard-40"),
            ("slm_1", "vf-cache-shard-40"),
            ("slm_2", "vf-cache-shard-40"),
            ("slm_42", "vf-cache-shard-41"),
            ("slm_100", "vf-cache-shard-14"),
            ("slm_255", "vf-cache-shard-31"),
            ("slm_256", "vf-cache-shard-31"),
            ("slm_1000", "vf-cache-shard-22"),
            ("slm_65535", "vf-cache-shard-33"),
            ("slm_65536", "vf-cache-shard-33"),
            ("slm_123456789", "vf-cache-shard-0"),
            ("slm_1000000007", "vf-cache-shard-28"),
            ("slm_9999999999", "vf-cache-shard-29"),
            ("slm_18446744073709551615", "vf-cache-shard-26"),
            ("slm_18446744073709551614", "vf-cache-shard-26"),
            ("slm_1234567890123456789", "vf-cache-shard-47"),
            ("slm_1111111111111111111", "vf-cache-shard-21"),
            ("slm_9223372036854775807", "vf-cache-shard-44"),
            ("slm_777", "vf-cache-shard-8"),
            ("slm_8675309", "vf-cache-shard-0"),
            ("slm_314159265358979", "vf-cache-shard-26"),
            ("slm_42424242", "vf-cache-shard-48"),
            ("slm_999999999999", "vf-cache-shard-44"),
            ("slm_13", "vf-cache-shard-19"),
            ("slm_1729", "vf-cache-shard-28"),
            ("slm_271828182845904", "vf-cache-shard-18"),
        ];

        let ring = HashRing::new(servers());
        for (key_str, expected) in GOLDEN {
            let key = Key::new(key_str.as_bytes().to_vec()).unwrap();
            let got = ring.server(&key).unwrap().ring_identifier();
            assert_eq!(&got, expected, "golden continuum drift for {key_str}");
        }
    }

    #[test]
    fn keyspace_is_distributed() {
        let ring = HashRing::new(servers());
        let distinct: HashSet<String> = keys()
            .iter()
            .map(|k| ring.server(k).unwrap().ring_identifier())
            .collect();
        assert!(
            distinct.len() > 10,
            "keys clustered onto {} shards",
            distinct.len()
        );
    }

    #[test]
    fn empty_ring_is_unavailable() {
        let ring = HashRing::new(HashSet::new());
        let key = Key::new(b"slm_1".to_vec()).unwrap();
        assert!(matches!(
            ring.server(&key),
            Err(TwemcacheError::Unavailable)
        ));
    }

    #[test]
    fn update_rebuilds_on_change() {
        let mut ring = HashRing::new(servers());
        let key = Key::new(b"slm_42".to_vec()).unwrap();
        let before = ring.server(&key).unwrap().ring_identifier();

        let mut fewer = servers();
        fewer.retain(|s| s.ring_identifier() != before);
        ring.update(fewer);

        assert_ne!(ring.server(&key).unwrap().ring_identifier(), before);
    }
}
