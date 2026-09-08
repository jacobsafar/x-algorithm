use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use futures::future::join_all;

use super::discovery::{Discovery, WilyDiscovery};
use super::error::{Result, TwemcacheError};
use super::host_pool::{HostPool, DEFAULT_CONNECTIONS_PER_HOST, DEFAULT_DEPTH_CAP};
use super::key::{Key, Server, ServerSet, Value};
use super::metrics::Metrics;
use super::ring::HashRing;

const DEPTH_SAMPLE_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) trait PoolFactory: Send + Sync {
    fn build(&self, server: &Server) -> Arc<HostPool>;
}

struct TcpPoolFactory {
    tls: Option<Arc<tokio_rustls::client::TlsConnector>>,
    connect_timeout: Duration,
    request_timeout: Duration,
    connections_per_host: usize,
    depth_cap: usize,
    metrics: Metrics,
}

impl PoolFactory for TcpPoolFactory {
    fn build(&self, server: &Server) -> Arc<HostPool> {
        Arc::new(HostPool::new(
            server.clone(),
            self.tls.clone(),
            self.connect_timeout,
            self.request_timeout,
            self.connections_per_host,
            self.depth_cap,
            self.metrics.clone(),
        ))
    }
}

pub struct TwemcacheClient {
    ring: Arc<ArcSwap<HashRing>>,
    pools: Arc<ArcSwap<HashMap<Server, Arc<HostPool>>>>,
    _discovery: Arc<dyn Discovery>,
    _handle: tokio::task::JoinHandle<()>,
    _sampler: tokio::task::JoinHandle<()>,
    metrics: Metrics,
}

impl Drop for TwemcacheClient {
    fn drop(&mut self) {
        self._handle.abort();
        self._sampler.abort();
    }
}

fn apply_membership(
    servers: &ServerSet,
    ring: &ArcSwap<HashRing>,
    pools: &ArcSwap<HashMap<Server, Arc<HostPool>>>,
    factory: &dyn PoolFactory,
    metrics: &Metrics,
) {
    let current = pools.load();
    let mut new_map = HashMap::with_capacity(servers.len());
    for s in servers {
        let pool = current.get(s).cloned().unwrap_or_else(|| factory.build(s));
        new_map.insert(s.clone(), pool);
    }
    pools.store(Arc::new(new_map));
    ring.store(Arc::new(HashRing::new(servers.clone())));
    metrics.set_discovery_servers(servers.len());
}

impl TwemcacheClient {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        path: impl Into<String>,
        client_name: impl Into<String>,
        zone: impl Into<String>,
        tls: Option<Arc<tokio_rustls::client::TlsConnector>>,
        request_timeout: Duration,
        connect_timeout: Duration,
        connections_per_host: usize,
        depth_cap: usize,
    ) -> Result<Self> {
        let discovery = Arc::new(WilyDiscovery::new(path, client_name, zone).await?);
        let metrics = Metrics::global();
        let factory = Arc::new(TcpPoolFactory {
            tls,
            connect_timeout,
            request_timeout,
            connections_per_host,
            depth_cap,
            metrics: metrics.clone(),
        });
        Ok(Self::with_parts(discovery, factory, metrics))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_tls_paths(
        path: impl Into<String>,
        client_name: impl Into<String>,
        zone: impl Into<String>,
        ca_cert_path: &str,
        client_cert_path: &str,
        client_key_path: &str,
        request_timeout: Duration,
        connect_timeout: Duration,
        connections_per_host: usize,
        depth_cap: usize,
    ) -> Result<Self> {
        let connector = super::connection::build_mtls_connector(
            ca_cert_path,
            client_cert_path,
            client_key_path,
        )?;
        Self::new(
            path,
            client_name,
            zone,
            Some(Arc::new(connector)),
            request_timeout,
            connect_timeout,
            connections_per_host,
            depth_cap,
        )
        .await
    }

    fn with_parts(
        discovery: Arc<dyn Discovery>,
        factory: Arc<dyn PoolFactory>,
        metrics: Metrics,
    ) -> Self {
        let mut rx = discovery.subscribe();
        let initial = rx.borrow_and_update().clone();

        let ring = Arc::new(ArcSwap::from_pointee(HashRing::new(ServerSet::new())));
        let pools = Arc::new(ArcSwap::from_pointee(HashMap::new()));
        apply_membership(&initial, &ring, &pools, factory.as_ref(), &metrics);

        let ring_task = Arc::clone(&ring);
        let pools_task = Arc::clone(&pools);
        let metrics_task = metrics.clone();
        let handle = tokio::spawn(async move {
            while rx.changed().await.is_ok() {
                let servers = rx.borrow_and_update().clone();
                apply_membership(
                    &servers,
                    &ring_task,
                    &pools_task,
                    factory.as_ref(),
                    &metrics_task,
                );
            }
        });

        let pools_sampler = Arc::clone(&pools);
        let metrics_sampler = metrics.clone();
        let sampler = tokio::spawn(async move {
            let mut interval = tokio::time::interval(DEPTH_SAMPLE_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let snapshot: Vec<Arc<HostPool>> = pools_sampler.load().values().cloned().collect();
                for pool in snapshot {
                    pool.for_each_depth(|d| metrics_sampler.observe_pipeline_depth(d));
                }
            }
        });

        Self {
            ring,
            pools,
            _discovery: discovery,
            _handle: handle,
            _sampler: sampler,
            metrics,
        }
    }

    pub async fn get(&self, key: &Key) -> Result<Option<Value>> {
        let server = self.ring.load().server(key)?;
        let pool = self.pools.load().get(&server).cloned();
        match pool {
            Some(p) => p.get(key).await,
            None => Err(TwemcacheError::Unavailable),
        }
    }

    pub async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, Result<Option<Value>>> {
        let futures = keys
            .iter()
            .map(|k| async move { (k.clone(), self.get(k).await) });
        let results: Vec<(Key, Result<Option<Value>>)> = join_all(futures).await;

        let shed = results
            .iter()
            .filter(|(_, r)| matches!(r, Err(TwemcacheError::Backpressure)))
            .count() as u64;
        self.metrics.record_backpressure(shed);

        results.into_iter().collect()
    }
}

pub const fn default_connections_per_host() -> usize {
    DEFAULT_CONNECTIONS_PER_HOST
}
pub const fn default_depth_cap() -> usize {
    DEFAULT_DEPTH_CAP
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twemcache::connection::PipelinedConnection;
    use crate::twemcache::discovery::ManualDiscovery;
    use crate::twemcache::host_pool::ConnFactory;
    use quanta::Clock;
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt};

    fn server(id: &str) -> Server {
        Server::with_ring_key("10.0.0.1".into(), 11211, id.into())
    }

    fn key(s: &str) -> Key {
        Key::new(s.as_bytes().to_vec()).unwrap()
    }

    async fn scripted_conn(reply: Option<Vec<u8>>) -> Arc<PipelinedConnection> {
        let (client, server) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            let (sr, mut sw) = tokio::io::split(server);
            let mut sr = BufReader::new(sr);
            let mut line = String::new();
            loop {
                line.clear();
                match sr.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let k = line.trim().strip_prefix("get ").unwrap_or("k").to_string();
                        let resp = match &reply {
                            Some(v) => {
                                let mut r = format!("VALUE {k} 0 {}\r\n", v.len()).into_bytes();
                                r.extend_from_slice(v);
                                r.extend_from_slice(b"\r\nEND\r\n");
                                r
                            }
                            None => b"END\r\n".to_vec(),
                        };
                        if sw.write_all(&resp).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Arc::new(PipelinedConnection::from_stream(
            Box::new(client),
            Duration::from_secs(60),
            Metrics::disabled(),
        ))
    }

    async fn saturated_conn() -> Arc<PipelinedConnection> {
        let (client, server) = tokio::io::duplex(256 * 1024);
        tokio::spawn(async move {
            let mut server = server;
            let mut buf = [0u8; 4096];
            loop {
                match server.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        });
        let conn = Arc::new(PipelinedConnection::from_stream(
            Box::new(client),
            Duration::from_secs(60),
            Metrics::disabled(),
        ));
        for _ in 0..100 {
            let c = Arc::clone(&conn);
            tokio::spawn(async move {
                let _ = c.get(&key("x")).await;
            });
        }
        while conn.in_flight() < 100 {
            tokio::task::yield_now().await;
        }
        conn
    }

    #[derive(Clone)]
    enum Behavior {
        Hit(Vec<u8>),
        Miss,
        Dead,
        Saturated,
    }

    struct ScriptedConnFactory {
        behavior: Behavior,
    }

    #[tonic::async_trait]
    impl ConnFactory for ScriptedConnFactory {
        async fn open(&self) -> Result<Arc<PipelinedConnection>> {
            match &self.behavior {
                Behavior::Hit(v) => Ok(scripted_conn(Some(v.clone())).await),
                Behavior::Miss => Ok(scripted_conn(None).await),
                Behavior::Dead => Err(TwemcacheError::Unavailable),
                Behavior::Saturated => Ok(saturated_conn().await),
            }
        }
    }

    struct DeadConnFactory;

    #[tonic::async_trait]
    impl ConnFactory for DeadConnFactory {
        async fn open(&self) -> Result<Arc<PipelinedConnection>> {
            Err(TwemcacheError::Unavailable)
        }
    }

    #[derive(Default)]
    struct RecordingReceiver {
        counters: StdMutex<HashMap<String, u64>>,
        gauges: StdMutex<HashMap<String, f64>>,
    }

    fn stat_key(name: &str, scopes: &[(&str, &str)]) -> String {
        let mut s = name.to_string();
        for (k, v) in scopes {
            s.push('|');
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s
    }

    impl StatsReceiverExt for RecordingReceiver {
        fn incr(&self, name: &str, scopes: &[(&str, &str)], value: u64) {
            *self
                .counters
                .lock()
                .unwrap()
                .entry(stat_key(name, scopes))
                .or_default() += value;
        }
        fn observe(&self, _: &str, _: &[(&str, &str)], _: f64, _: HistogramBuckets) {}
        fn observe_expo(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn observe_vm(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn gauge(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
            self.gauges
                .lock()
                .unwrap()
                .insert(stat_key(name, scopes), value);
        }
    }

    impl RecordingReceiver {
        fn counter(&self, key: &str) -> u64 {
            *self.counters.lock().unwrap().get(key).unwrap_or(&0)
        }
        fn gauge_of(&self, key: &str) -> Option<f64> {
            self.gauges.lock().unwrap().get(key).copied()
        }
    }

    struct FakePoolFactory {
        behaviors: HashMap<Server, Behavior>,
        default: Behavior,
        metrics: Metrics,
    }

    impl FakePoolFactory {
        fn uniform(default: Behavior) -> Self {
            Self {
                behaviors: HashMap::new(),
                default,
                metrics: Metrics::disabled(),
            }
        }
    }

    impl PoolFactory for FakePoolFactory {
        fn build(&self, server: &Server) -> Arc<HostPool> {
            let behavior = self
                .behaviors
                .get(server)
                .cloned()
                .unwrap_or(self.default.clone());
            let factory: Arc<dyn ConnFactory> = Arc::new(ScriptedConnFactory { behavior });
            Arc::new(HostPool::with_factory(
                factory,
                2,
                100,
                Clock::new(),
                self.metrics.clone(),
            ))
        }
    }

    fn client_with(
        servers: ServerSet,
        factory: Arc<dyn PoolFactory>,
        metrics: Metrics,
    ) -> (TwemcacheClient, Arc<ManualDiscovery>) {
        let disco = Arc::new(ManualDiscovery::new(servers));
        let client =
            TwemcacheClient::with_parts(Arc::clone(&disco) as Arc<dyn Discovery>, factory, metrics);
        (client, disco)
    }

    async fn wait_until<F: Fn() -> bool>(cond: F) {
        for _ in 0..200 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("condition not met in time");
    }

    #[tokio::test]
    async fn route_to_hit() {
        let servers = HashSet::from([server("shard-a"), server("shard-b")]);
        let factory = Arc::new(FakePoolFactory::uniform(Behavior::Hit(b"value".to_vec())));
        let (client, _d) = client_with(servers, factory, Metrics::disabled());
        assert_eq!(
            client.get(&key("slm_42")).await.unwrap(),
            Some(b"value".to_vec())
        );
    }

    #[tokio::test]
    async fn route_to_miss() {
        let servers = HashSet::from([server("shard-a"), server("shard-b")]);
        let factory = Arc::new(FakePoolFactory::uniform(Behavior::Miss));
        let (client, _d) = client_with(servers, factory, Metrics::disabled());
        assert_eq!(client.get(&key("slm_42")).await.unwrap(), None);
    }

    #[tokio::test]
    async fn empty_ring_is_unavailable() {
        let factory = Arc::new(FakePoolFactory::uniform(Behavior::Miss));
        let (client, _d) = client_with(HashSet::new(), factory, Metrics::disabled());
        assert!(matches!(
            client.get(&key("slm_42")).await,
            Err(TwemcacheError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn dead_host_fails_fast_others_isolated() {
        let servers: ServerSet = (0..5).map(|i| server(&format!("shard-{i}"))).collect();
        let ring = HashRing::new(servers.clone());

        let dead_key = key("slm_42");
        let dead_owner = ring.server(&dead_key).unwrap();
        let live_key = (0u64..)
            .map(|n| key(&format!("slm_{n}")))
            .find(|k| ring.server(k).unwrap() != dead_owner)
            .unwrap();

        let mut behaviors = HashMap::new();
        behaviors.insert(dead_owner.clone(), Behavior::Dead);
        let factory = Arc::new(FakePoolFactory {
            behaviors,
            default: Behavior::Hit(b"ok".to_vec()),
            metrics: Metrics::disabled(),
        });
        let (client, _d) = client_with(servers, factory, Metrics::disabled());

        assert!(matches!(
            client.get(&dead_key).await,
            Err(TwemcacheError::Unavailable)
        ));
        assert_eq!(client.get(&live_key).await.unwrap(), Some(b"ok".to_vec()));
    }

    #[tokio::test]
    async fn membership_change_rebuilds_ring_and_preserves_pools() {
        let a = server("shard-a");
        let b = server("shard-b");
        let c = server("shard-c");
        let factory = Arc::new(FakePoolFactory::uniform(Behavior::Miss));
        let (client, disco) = client_with(
            HashSet::from([a.clone(), b.clone()]),
            factory,
            Metrics::disabled(),
        );

        let pool_a_before = client.pools.load().get(&a).cloned().unwrap();
        let key_for_c = (0u64..)
            .map(|n| key(&format!("slm_{n}")))
            .find(|k| {
                HashRing::new(HashSet::from([a.clone(), b.clone(), c.clone()]))
                    .server(k)
                    .unwrap()
                    == c
            })
            .unwrap();
        assert_ne!(client.ring.load().server(&key_for_c).unwrap(), c);

        disco.set(HashSet::from([a.clone(), b.clone(), c.clone()]));
        wait_until(|| client.pools.load().contains_key(&c)).await;

        assert_eq!(client.ring.load().server(&key_for_c).unwrap(), c);
        let pool_a_after = client.pools.load().get(&a).cloned().unwrap();
        assert!(Arc::ptr_eq(&pool_a_before, &pool_a_after));
    }

    #[tokio::test]
    async fn client_retains_discovery_after_caller_drops_arc() {
        let a = server("shard-a");
        let b = server("shard-b");
        let disco = Arc::new(ManualDiscovery::new(HashSet::from([a.clone()])));
        let weak = Arc::downgrade(&disco);

        let factory = Arc::new(FakePoolFactory::uniform(Behavior::Miss));
        let client = TwemcacheClient::with_parts(
            Arc::clone(&disco) as Arc<dyn Discovery>,
            factory,
            Metrics::disabled(),
        );

        drop(disco);
        let retained = weak
            .upgrade()
            .expect("client must retain the discovery Arc for its lifetime");

        retained.set(HashSet::from([a.clone(), b.clone()]));
        wait_until(|| client.pools.load().contains_key(&b)).await;
    }

    #[tokio::test]
    async fn multi_get_returns_per_key_outcomes() {
        let servers: ServerSet = (0..6).map(|i| server(&format!("s{i}"))).collect();
        let ring = HashRing::new(servers.clone());

        let mut chosen: Vec<(Key, Server)> = Vec::new();
        for n in 0u64.. {
            let k = key(&format!("slm_{n}"));
            let owner = ring.server(&k).unwrap();
            if !chosen.iter().any(|(_, o)| *o == owner) {
                chosen.push((k, owner));
            }
            if chosen.len() == 3 {
                break;
            }
        }
        let (k_hit, owner_hit) = chosen[0].clone();
        let (k_miss, owner_miss) = chosen[1].clone();
        let (k_dead, owner_dead) = chosen[2].clone();

        let behaviors = HashMap::from([
            (owner_hit, Behavior::Hit(b"v".to_vec())),
            (owner_miss, Behavior::Miss),
            (owner_dead, Behavior::Dead),
        ]);
        let factory = Arc::new(FakePoolFactory {
            behaviors,
            default: Behavior::Miss,
            metrics: Metrics::disabled(),
        });
        let (client, _d) = client_with(servers, factory, Metrics::disabled());

        let out = client
            .multi_get(&[k_hit.clone(), k_miss.clone(), k_dead.clone()])
            .await;

        assert_eq!(
            out.get(&k_hit).unwrap().as_ref().unwrap(),
            &Some(b"v".to_vec())
        );
        assert_eq!(out.get(&k_miss).unwrap().as_ref().unwrap(), &None);
        assert!(matches!(
            out.get(&k_dead).unwrap(),
            Err(TwemcacheError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn metrics_emission_paths_move_counters() {
        let rec = Arc::new(RecordingReceiver::default());
        let metrics = Metrics::fixed(Arc::clone(&rec) as Arc<dyn StatsReceiverExt>);

        let servers = HashSet::from([server("shard-a"), server("shard-b")]);
        let factory = Arc::new(FakePoolFactory {
            behaviors: HashMap::new(),
            default: Behavior::Miss,
            metrics: metrics.clone(),
        });
        let (_client, _d) = client_with(servers, factory, metrics.clone());
        assert_eq!(rec.gauge_of("twemcache_discovery_servers"), Some(2.0));

        let pool = HostPool::with_factory(
            Arc::new(DeadConnFactory),
            1,
            100,
            Clock::new(),
            metrics.clone(),
        );
        assert!(matches!(
            pool.get(&key("k")).await,
            Err(TwemcacheError::Unavailable)
        ));
        assert!(rec.counter("twemcache_connection_events_total|event=tripped") >= 1);

        let before = rec.counter("twemcache_backpressure_total");
        let sat_factory = Arc::new(FakePoolFactory {
            behaviors: HashMap::new(),
            default: Behavior::Saturated,
            metrics: metrics.clone(),
        });
        let (sat_client, _d2) = client_with(
            HashSet::from([server("only")]),
            sat_factory,
            metrics.clone(),
        );
        let out = sat_client.multi_get(&[key("slm_1")]).await;
        assert!(matches!(
            out.get(&key("slm_1")).unwrap(),
            Err(TwemcacheError::Backpressure)
        ));
        assert_eq!(rec.counter("twemcache_backpressure_total"), before + 1);
    }
}
