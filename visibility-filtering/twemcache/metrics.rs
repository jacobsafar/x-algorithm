use std::sync::Arc;

use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt, global_stats_receiver};

const BACKPRESSURE: &str = "twemcache_backpressure_total";
const PIPELINE_DEPTH: &str = "twemcache_pipeline_depth";
const CONNECTION_EVENTS: &str = "twemcache_connection_events_total";
const DISCOVERY_SERVERS: &str = "twemcache_discovery_servers";

#[derive(Clone, Copy)]
pub(crate) enum ConnEvent {
    Tripped,
    ProbeOk,
    ProbeFail,
    TornDown,
    IdleCloseRetry,
}

impl ConnEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tripped => "tripped",
            Self::ProbeOk => "probe_ok",
            Self::ProbeFail => "probe_fail",
            Self::TornDown => "torn_down",
            Self::IdleCloseRetry => "idle_close_retry",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Metrics(Sink);

#[derive(Clone)]
enum Sink {
    Global,
    #[cfg(test)]
    Disabled,
    #[cfg(test)]
    Fixed(Arc<dyn StatsReceiverExt>),
}

impl Metrics {
    pub(crate) fn global() -> Self {
        Self(Sink::Global)
    }

    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self(Sink::Disabled)
    }

    #[cfg(test)]
    pub(crate) fn fixed(sink: Arc<dyn StatsReceiverExt>) -> Self {
        Self(Sink::Fixed(sink))
    }

    fn sink(&self) -> Option<Arc<dyn StatsReceiverExt>> {
        match &self.0 {
            Sink::Global => global_stats_receiver(),
            #[cfg(test)]
            Sink::Disabled => None,
            #[cfg(test)]
            Sink::Fixed(s) => Some(Arc::clone(s)),
        }
    }

    pub(crate) fn record_backpressure(&self, count: u64) {
        if count > 0
            && let Some(sr) = self.sink()
        {
            sr.incr(BACKPRESSURE, &[], count);
        }
    }

    pub(crate) fn observe_pipeline_depth(&self, depth: usize) {
        if let Some(sr) = self.sink() {
            sr.observe(
                PIPELINE_DEPTH,
                &[],
                depth as f64,
                HistogramBuckets::Bucket0To50,
            );
        }
    }

    pub(crate) fn record_connection_event(&self, event: ConnEvent) {
        if let Some(sr) = self.sink() {
            sr.incr(CONNECTION_EVENTS, &[("event", event.as_str())], 1);
        }
    }

    pub(crate) fn set_discovery_servers(&self, count: usize) {
        if let Some(sr) = self.sink() {
            sr.gauge(DISCOVERY_SERVERS, &[], count as f64);
        }
    }
}
