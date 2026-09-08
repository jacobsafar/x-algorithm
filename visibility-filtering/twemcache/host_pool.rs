use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use quanta::{Clock, Instant};
use tokio::sync::Mutex as TokioMutex;

use super::accrual::{ConnHealth, Outcome};
use super::connection::PipelinedConnection;
use super::error::{Result, TwemcacheError};
use super::key::{Key, Server, Value};
use super::metrics::{ConnEvent, Metrics};

pub const DEFAULT_CONNECTIONS_PER_HOST: usize = 2;
pub const DEFAULT_DEPTH_CAP: usize = 100;

const DEFAULT_PROBE_STALE_AFTER: Duration = Duration::from_secs(5);

fn probe_stale_bound(connect_timeout: Duration, request_timeout: Duration) -> Duration {
    (connect_timeout + request_timeout) * 2
}

#[tonic::async_trait]
pub(crate) trait ConnFactory: Send + Sync {
    async fn open(&self) -> Result<Arc<PipelinedConnection>>;
}

struct TcpConnFactory {
    server: Server,
    tls: Option<Arc<tokio_rustls::client::TlsConnector>>,
    connect_timeout: Duration,
    request_timeout: Duration,
    metrics: Metrics,
}

#[tonic::async_trait]
impl ConnFactory for TcpConnFactory {
    async fn open(&self) -> Result<Arc<PipelinedConnection>> {
        let conn = PipelinedConnection::connect(
            &self.server,
            self.tls.as_deref(),
            self.connect_timeout,
            self.request_timeout,
            self.metrics.clone(),
        )
        .await?;
        Ok(Arc::new(conn))
    }
}

struct ConnSlot {
    conn: Option<Arc<PipelinedConnection>>,
    health: ConnHealth,
}

impl ConnSlot {
    fn has_living_conn(&self) -> bool {
        matches!(&self.conn, Some(c) if !c.is_dead())
    }

    fn live_conn(&self) -> Option<&Arc<PipelinedConnection>> {
        if !self.health.is_live() {
            return None;
        }
        match &self.conn {
            Some(c) if !c.is_dead() => Some(c),
            _ => None,
        }
    }
}

enum Selected {
    Live(Arc<PipelinedConnection>, usize),
    Probe(Arc<PipelinedConnection>, usize),
}

type BestConn = Option<(usize, Arc<PipelinedConnection>, usize)>;

pub struct HostPool {
    factory: Arc<dyn ConnFactory>,
    slots: Vec<Mutex<ConnSlot>>,
    open_lock: TokioMutex<()>,
    depth_cap: usize,
    probe_stale_after: Duration,
    clock: Clock,
    metrics: Metrics,
}

fn classify(result: &Result<Option<Value>>) -> Outcome {
    match result {
        Ok(_) => Outcome::Success,
        Err(TwemcacheError::Backpressure) => Outcome::Backpressure,
        Err(_) => Outcome::Failure,
    }
}

fn is_dead_socket_err(err: &TwemcacheError) -> bool {
    matches!(err, TwemcacheError::Unavailable | TwemcacheError::Io(_))
}

fn seed_for(idx: usize) -> u64 {
    (idx as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03
}

impl HostPool {
    pub fn new(
        server: Server,
        tls: Option<Arc<tokio_rustls::client::TlsConnector>>,
        connect_timeout: Duration,
        request_timeout: Duration,
        connections_per_host: usize,
        depth_cap: usize,
        metrics: Metrics,
    ) -> Self {
        let factory = Arc::new(TcpConnFactory {
            server,
            tls,
            connect_timeout,
            request_timeout,
            metrics: metrics.clone(),
        });
        let mut pool = Self::with_factory(
            factory,
            connections_per_host,
            depth_cap,
            Clock::new(),
            metrics,
        );
        pool.probe_stale_after = probe_stale_bound(connect_timeout, request_timeout);
        pool
    }

    pub(crate) fn with_factory(
        factory: Arc<dyn ConnFactory>,
        connections_per_host: usize,
        depth_cap: usize,
        clock: Clock,
        metrics: Metrics,
    ) -> Self {
        let slots = (0..connections_per_host.max(1))
            .map(|i| {
                Mutex::new(ConnSlot {
                    conn: None,
                    health: ConnHealth::new(seed_for(i)),
                })
            })
            .collect();
        Self {
            factory,
            slots,
            open_lock: TokioMutex::new(()),
            depth_cap,
            probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
            clock,
            metrics,
        }
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "every idx comes from enumerate() over self.slots"
    )]
    fn slot_guard(&self, idx: usize) -> MutexGuard<'_, ConnSlot> {
        lock_slot(&self.slots[idx])
    }

    pub async fn get(&self, key: &Key) -> Result<Option<Value>> {
        let now = self.clock.now();
        let (conn, idx, is_probe) = match self.select(now).await? {
            Selected::Live(c, i) => (c, i, false),
            Selected::Probe(c, i) => (c, i, true),
        };
        let result = conn.get(key).await;

        if !is_probe && result.as_ref().err().is_some_and(is_dead_socket_err) {
            {
                let mut g = self.slot_guard(idx);
                if g.conn.as_ref().is_some_and(|c| Arc::ptr_eq(c, &conn)) {
                    g.conn = None;
                }
            }

            let now = self.clock.now();
            let (conn2, idx2, is_probe2) = match self.select(now).await? {
                Selected::Live(c, i) => (c, i, false),
                Selected::Probe(c, i) => (c, i, true),
            };
            self.metrics
                .record_connection_event(ConnEvent::IdleCloseRetry);
            let result2 = conn2.get(key).await;
            self.record(idx2, &result2, is_probe2, self.clock.now());
            return result2;
        }

        self.record(idx, &result, is_probe, self.clock.now());
        result
    }

    async fn select(&self, now: Instant) -> Result<Selected> {
        let (best, live_exists) = {
            let (best, live_exists, any_need) = self.scan();
            if any_need {
                self.ensure_live_open(now).await;
                let (best, live_exists, _) = self.scan();
                (best, live_exists)
            } else {
                (best, live_exists)
            }
        };

        if let Some((idx, conn, _)) = best
            && conn.in_flight() < self.depth_cap
        {
            return Ok(Selected::Live(conn, idx));
        }

        if let Some((idx, conn)) = self.try_open_probe(now).await {
            return Ok(Selected::Probe(conn, idx));
        }

        if live_exists {
            Err(TwemcacheError::Backpressure)
        } else {
            Err(TwemcacheError::Unavailable)
        }
    }

    fn scan(&self) -> (BestConn, bool, bool) {
        let mut best: BestConn = None;
        let mut live_exists = false;
        let mut any_need = false;
        for (idx, slot) in self.slots.iter().enumerate() {
            let g = lock_slot(slot);
            if g.health.is_live() && !g.has_living_conn() {
                any_need = true;
            }
            if let Some(c) = g.live_conn() {
                live_exists = true;
                let in_flight = c.in_flight();
                if best.as_ref().is_none_or(|(_, _, b)| in_flight < *b) {
                    best = Some((idx, Arc::clone(c), in_flight));
                }
            }
        }
        (best, live_exists, any_need)
    }

    async fn ensure_live_open(&self, now: Instant) {
        let any_need = self.slots.iter().any(|s| {
            let g = lock_slot(s);
            g.health.is_live() && !g.has_living_conn()
        });
        if !any_need {
            return;
        }

        let _guard = self.open_lock.lock().await;
        for slot in &self.slots {
            let need = {
                let g = lock_slot(slot);
                g.health.is_live() && !g.has_living_conn()
            };
            if !need {
                continue;
            }
            match self.factory.open().await {
                Ok(conn) => {
                    let mut g = lock_slot(slot);
                    if g.health.is_live() && !g.has_living_conn() {
                        g.conn = Some(conn);
                    }
                }
                Err(_) => {
                    let tripped = {
                        let mut g = lock_slot(slot);
                        if g.health.is_live() {
                            g.health.record_connect_failure(now);
                            true
                        } else {
                            false
                        }
                    };
                    if tripped {
                        self.metrics.record_connection_event(ConnEvent::Tripped);
                    }
                }
            }
        }
    }

    async fn try_open_probe(&self, now: Instant) -> Option<(usize, Arc<PipelinedConnection>)> {
        let idx = {
            let mut claimed = None;
            for (i, slot) in self.slots.iter().enumerate() {
                let mut g = lock_slot(slot);
                if g.health.is_probe_eligible(now, self.probe_stale_after) {
                    g.health.begin_probe(now);
                    g.conn = None;
                    claimed = Some(i);
                    break;
                }
            }
            claimed?
        };

        let _guard = self.open_lock.lock().await;
        match self.factory.open().await {
            Ok(conn) => {
                let mut g = self.slot_guard(idx);
                g.conn = Some(Arc::clone(&conn));
                Some((idx, conn))
            }
            Err(_) => {
                let mut g = self.slot_guard(idx);
                g.health.note_probe(Outcome::Failure, now);
                None
            }
        }
    }

    fn record(&self, idx: usize, result: &Result<Option<Value>>, is_probe: bool, now: Instant) {
        let outcome = classify(result);
        let event = {
            let mut g = self.slot_guard(idx);
            if is_probe {
                g.health.note_probe(outcome, now);
                match outcome {
                    Outcome::Success => Some(ConnEvent::ProbeOk),
                    Outcome::Failure => Some(ConnEvent::ProbeFail),
                    Outcome::Backpressure => None,
                }
            } else if g.health.note(outcome, now).is_some() {
                Some(ConnEvent::Tripped)
            } else {
                None
            }
        };
        if let Some(e) = event {
            self.metrics.record_connection_event(e);
        }
    }

    pub(crate) fn for_each_depth(&self, mut sink: impl FnMut(usize)) {
        for slot in &self.slots {
            let g = lock_slot(slot);
            if let Some(c) = &g.conn {
                sink(c.in_flight());
            }
        }
    }
}

#[expect(clippy::unwrap_used, reason = "propagate slot-lock poisoning")]
fn lock_slot(slot: &Mutex<ConnSlot>) -> MutexGuard<'_, ConnSlot> {
    slot.lock().unwrap()
}

#[cfg(test)]
#[path = "host_pool_tests.rs"]
mod tests;
