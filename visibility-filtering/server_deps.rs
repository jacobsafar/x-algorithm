use crate::clients::socialgraph_client::ProdSocialgraphClient;
use crate::filter::{FilterRequest, FilterResponse, FilterTweets};
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use crate::hydration::HydrationPipeline;
use crate::models::{RawCandidate, TweetId};
use crate::reference_compare::ReferenceCompareHarness;
use crate::rules::{SafetyLevel, Verdict};
use crate::safety_label_source::lookup::RemoteSource;
use crate::safety_label_source::manhattan::ManhattanSource;
use crate::safety_label_source::twemcache::TwemcacheSource;
use crate::safety_label_source::warmer::{CacheWarmer, StratoWarmFetcher, Warmer};
use crate::safety_label_source::{ManhattanLabelFetcher, MhLabelClient, SafetyLabelSource};
use crate::server::VFServer;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use xai_core_entities::gizmoduck_client::{GizmoduckClientConfig, ProdGizmoduckClient};
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants, TESRpcConstants};
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_core_entities::tweet_entity_service_client::{ProdTESClient, TESClientConfig};
use xai_visibility_filtering::vf_client::{StratoVfClient, VfClient};
use xai_x_rpc::balanced_channel::LbPolicy;

const CACHE_PATH: &str = "/s/cache/safety_label_store:twemcaches";
const TES_READINESS_PROBE_PORT: u16 = 8081;
const GIZMODUCK_READINESS_PROBE_PORT: u16 = 8081;

const CLIENT_INIT_RETRY_BUDGET: Duration = Duration::from_secs(240);
const CLIENT_INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);
const CLIENT_INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

async fn init_client_with_retry<T, E, Fut>(
    client: &str,
    deadline: tokio::time::Instant,
    mut build: impl FnMut() -> Fut,
) -> Result<T, String>
where
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1u32;
    loop {
        if tokio::time::Instant::now() >= deadline {
            error!(
                client,
                attempt, "Client init skipped; init deadline exhausted"
            );
            return Err("init deadline exhausted before attempt".to_string());
        }
        let error = match tokio::time::timeout(CLIENT_INIT_ATTEMPT_TIMEOUT, build()).await {
            Ok(Ok(t)) => {
                if attempt > 1 {
                    info!(client, attempt, "Client init succeeded after retry");
                }
                return Ok(t);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ),
        };
        if tokio::time::Instant::now() + backoff >= deadline {
            error!(
                client,
                attempt,
                error = %error,
                "Client init failed; retry budget exhausted"
            );
            return Err(error);
        }
        warn!(
            client,
            attempt,
            backoff_secs = backoff.as_secs(),
            error = %error,
            "Client init failed; retrying"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(CLIENT_INIT_MAX_BACKOFF);
        attempt += 1;
    }
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub async fn build_prod_server(
    datacenter: &str,
    feature_switches: Arc<xai_feature_switches::FeatureSwitches>,
) -> VFServer {
    info!("Initializing prod clients for datacenter={}", datacenter);

    let init_deadline = tokio::time::Instant::now() + CLIENT_INIT_RETRY_BUDGET;

    let deterministic_aperture = std::env::var("APP_ENV").as_deref() == Ok("prod");
    let fallback_cache_enabled = crate::config::fallback_cache_enabled();
    let fallback_cache = fallback_cache_enabled
        .then(crate::hydration::gizmoduck_hydrator::GizmoduckAuthorHydrator::fallback_cache);

    let tes_client: Arc<
        dyn xai_core_entities::tweet_entity_service_client::TESClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("tes", init_deadline, || {
            ProdTESClient::new_with_config(
                None,
                datacenter,
                None,
                tes_client_config(deterministic_aperture),
            )
        })
        .await
        .expect("Failed to initialize TES client"),
    );

    let gizmoduck_client_id = crate::config::gizmoduck_client_id();
    let gizmoduck_client: Arc<
        dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("gizmoduck", init_deadline, || {
            ProdGizmoduckClient::new_with_config(
                None,
                datacenter,
                Some(gizmoduck_client_id.clone()),
                gizmoduck_client_config(deterministic_aperture),
            )
        })
        .await
        .expect("Failed to initialize Gizmoduck client"),
    );
    info!(
        deterministic_aperture,
        "TES client: aperture + penalized peak-EWMA P2C + readiness probe; Gizmoduck client: \
         aperture + least-request P2C + readiness probe"
    );

    let sg_client: Arc<dyn crate::clients::socialgraph_client::SocialgraphClient + Send + Sync> =
        Arc::new(
            init_client_with_retry("socialgraph", init_deadline, || {
                ProdSocialgraphClient::new(
                    datacenter,
                    &S2S_CHAIN_PATH,
                    &S2S_CRT_PATH,
                    &S2S_KEY_PATH,
                )
            })
            .await
            .expect("Failed to initialize SocialGraph client"),
        );

    let mh_label_client: Arc<dyn ManhattanLabelFetcher> = Arc::new(
        init_client_with_retry("manhattan", init_deadline, || {
            let s2s = xai_manhattan::s2s::S2sConfig {
                client_cert_path: S2S_CRT_PATH.clone(),
                client_key_path: S2S_KEY_PATH.clone(),
                ca_cert_path: S2S_CHAIN_PATH.clone(),
            };
            MhLabelClient::new(datacenter, s2s, deterministic_aperture)
        })
        .await
        .expect("Failed to initialize MhLabelClient"),
    );

    let twemcache_client_name = crate::config::twemcache_client_name();
    let twemcache = Arc::new(
        init_client_with_retry("twemcache", init_deadline, || {
            crate::twemcache::TwemcacheClient::new_with_tls_paths(
                CACHE_PATH,
                twemcache_client_name.clone(),
                datacenter,
                &S2S_CHAIN_PATH,
                &S2S_CRT_PATH,
                &S2S_KEY_PATH,
                Duration::from_millis(20),
                Duration::from_secs(5),
                crate::twemcache::client::default_connections_per_host(),
                crate::twemcache::client::default_depth_cap(),
            )
        })
        .await
        .expect("Failed to create twemcache client"),
    );
    info!("Cache client connected to {CACHE_PATH}");

    let reference_compare = build_reference_compare_harness(datacenter, init_deadline).await;

    warm_cache(&twemcache).await;
    warm_manhattan(mh_label_client.as_ref()).await;

    let cache_warmer = build_cache_warmer(datacenter, init_deadline).await;

    let twemcache_source = Arc::new(TwemcacheSource::new(twemcache));
    let manhattan_source = Arc::new(ManhattanSource::new(mh_label_client));
    let mut remote = RemoteSource::new(twemcache_source, manhattan_source);
    if let Some(warmer) = cache_warmer {
        remote = remote.with_warmer(warmer);
    }
    let remote = Arc::new(remote);
    let safety_label_source = Arc::new(SafetyLabelSource::new(remote));

    let hydration_pipeline = HydrationPipeline::new(
        tes_client,
        gizmoduck_client,
        sg_client,
        safety_label_source.clone(),
        fallback_cache,
    );
    let gating_countries = Arc::new(crate::params::NsfwGatingCountries::starting_at_default());
    let fs_path = crate::config::fs_path();
    gating_countries.refresh_and_check_drift(&feature_switches, &fs_path);
    gating_countries.spawn_refresh(feature_switches, fs_path);
    let rule_engine = crate::rules::RuleEngine::with_nsfw_gating_countries(gating_countries);
    let (home_rule_count, recommendations_rule_count) = rule_engine.rule_counts();
    let filter_tweets = FilterTweets::new(hydration_pipeline, rule_engine);

    warm_filter_tweets(&filter_tweets).await;

    info!(
        hydrator_count = 5,
        fallback_cache_enabled,
        home_rule_count,
        recommendations_rule_count,
        "VFServer initialized with prod clients"
    );

    VFServer::from_endpoints(
        FilterTweetsEndpoint::new(filter_tweets, reference_compare),
        GetSafetyLabelsEndpoint::new(safety_label_source),
    )
}

async fn build_reference_compare_harness(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
) -> Option<Arc<ReferenceCompareHarness>> {
    #[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
    let should_build = crate::reference_compare::should_build_harness(
        crate::config::dual_call_harness_enabled(),
        std::env::var("APP_ENV").ok().as_deref(),
    )
    .unwrap_or_else(|misconfiguration| panic!("{misconfiguration}"));
    if !should_build {
        return None;
    }

    let client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "staging".to_string())
    );
    #[expect(
        clippy::expect_used,
        reason = "startup fail-fast: init failure is fatal"
    )]
    let strato: Arc<dyn VfClient + Send + Sync> = Arc::new(
        init_client_with_retry("strato_vf", init_deadline, || {
            let client_id = client_id.clone();
            async move {
                StratoVfClient::new(
                    S2S_CHAIN_PATH.clone(),
                    S2S_CRT_PATH.clone(),
                    S2S_KEY_PATH.clone(),
                    client_id,
                    datacenter.to_string(),
                )
                .await
                .map_err(|e| e.to_string())
            }
        })
        .await
        .expect("Failed to initialize Strato VF client (reference comparator)"),
    );
    Some(Arc::new(ReferenceCompareHarness::new(strato, datacenter)))
}

const CACHE_WARM_REQUEST_TIMEOUT_MS: u64 = 500;

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
async fn build_cache_warmer(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
) -> Option<Arc<dyn Warmer>> {
    if !crate::config::cache_warm_enabled() {
        return None;
    }

    let client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
    );
    let grpc = init_client_with_retry("strato_cache_warm", init_deadline, || {
        let config = xai_strato::StratoGrpcConfig {
            ca_cert_path: S2S_CHAIN_PATH.clone(),
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            num_endpoints: Some(12),
            connect_timeout_ms: 400,
            request_timeout_ms: CACHE_WARM_REQUEST_TIMEOUT_MS,
            client_id: Some(client_id.clone()),
            service_url: format!("stratostore.stratoserver.prod.{datacenter}.s2s.twttr.net"),
            zone: datacenter.to_string(),
            ..Default::default()
        };
        async move {
            xai_strato::StratoGrpc::new(config)
                .await
                .map_err(|e| e.to_string())
        }
    })
    .await
    .expect("Failed to initialize Strato cache-warm client");
    info!("L2 cache warmer enabled");
    Some(CacheWarmer::spawn(Arc::new(StratoWarmFetcher::new(grpc))))
}

const TES_STRATO_REQUEST_TIMEOUT_MS: u64 = 100;

fn tes_client_config(deterministic_aperture: bool) -> TESClientConfig {
    TESClientConfig {
        aperture_size: Some(TESRpcConstants::num_endpoints()),
        deterministic_aperture,
        lb_policy: Some(LbPolicy::penalized_peak_ewma()),
        readiness_probe_port: Some(TES_READINESS_PROBE_PORT),
        request_timeout_ms: Some(TES_STRATO_REQUEST_TIMEOUT_MS),
        enable_serve_within: true,
        ..Default::default()
    }
}

const GIZMODUCK_STRATO_REQUEST_TIMEOUT_MS: u64 = 80;

fn gizmoduck_client_config(deterministic_aperture: bool) -> GizmoduckClientConfig {
    GizmoduckClientConfig {
        aperture_size: Some(GizmoduckRpcConstants::num_endpoints()),
        deterministic_aperture,
        lb_policy: Some(LbPolicy::least_request()),
        readiness_probe_port: Some(GIZMODUCK_READINESS_PROBE_PORT),
        request_timeout_ms: Some(GIZMODUCK_STRATO_REQUEST_TIMEOUT_MS),
        ..Default::default()
    }
}

async fn warm_cache(twemcache: &crate::twemcache::TwemcacheClient) {
    let start = Instant::now();
    let key = match crate::twemcache::Key::new(b"slm_warmup".to_vec()) {
        Ok(key) => key,
        Err(e) => {
            warn!(error = %e, "Cache warmup key construction failed (non-fatal)");
            return;
        }
    };
    let latency_ms = || start.elapsed().as_millis() as u64;
    match twemcache
        .multi_get(std::slice::from_ref(&key))
        .await
        .get(&key)
    {
        Some(Ok(_)) => info!(latency_ms = latency_ms(), "Cache warmup succeeded"),
        Some(Err(e)) => warn!(
            latency_ms = latency_ms(),
            error = %e,
            "Cache warmup failed (non-fatal)"
        ),
        None => warn!(
            latency_ms = latency_ms(),
            "Cache warmup returned no response (non-fatal)"
        ),
    }
}

const WARM_FILTER_TWEETS_TWEET_ID: u64 = 20;
const WARM_FILTER_TWEETS_CONSECUTIVE: u32 = 3;
const WARM_FILTER_TWEETS_DEADLINE: Duration = Duration::from_secs(20);
const WARM_FILTER_TWEETS_SLEEP: Duration = Duration::from_millis(500);

fn warm_filter_tweets_request() -> FilterRequest {
    FilterRequest {
        viewer_id: None,
        country_code: None,
        safety_level: SafetyLevel::TimelineHomeRecommendations,
        candidates: vec![RawCandidate {
            tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
            request_author_id: None,
        }],
    }
}

fn filter_tweets_response_is_warm(response: &FilterResponse) -> bool {
    response
        .outcomes
        .iter()
        .all(|outcome| outcome.verdict.decided_by != Verdict::unresolved_author().decided_by)
}

async fn warm_filter_tweets(filter_tweets: &FilterTweets) {
    let start = Instant::now();
    let mut attempts = 0u32;
    let mut consecutive_warm = 0u32;
    while start.elapsed() < WARM_FILTER_TWEETS_DEADLINE {
        attempts += 1;
        let response = filter_tweets.run(warm_filter_tweets_request()).await;
        if filter_tweets_response_is_warm(&response) {
            consecutive_warm += 1;
            if consecutive_warm >= WARM_FILTER_TWEETS_CONSECUTIVE {
                info!(
                    attempts,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "filter_tweets warmup succeeded"
                );
                return;
            }
        } else {
            consecutive_warm = 0;
        }
        if start.elapsed() + WARM_FILTER_TWEETS_SLEEP < WARM_FILTER_TWEETS_DEADLINE {
            tokio::time::sleep(WARM_FILTER_TWEETS_SLEEP).await;
        } else {
            break;
        }
    }
    warn!(
        attempts,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "filter_tweets warmup deadline expired (non-fatal)"
    );
}

async fn warm_manhattan(manhattan: &dyn ManhattanLabelFetcher) {
    let start = Instant::now();
    match manhattan.fetch_labels(&[0]).await {
        Ok(_) => info!(
            latency_ms = start.elapsed().as_millis() as u64,
            "Manhattan warmup succeeded"
        ),
        Err(e) => warn!(
            latency_ms = start.elapsed().as_millis() as u64,
            error = %e,
            "Manhattan warmup failed (non-fatal)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterOutcome;
    use crate::models::VfAction;
    use std::cell::Cell;
    use xai_visibility_filtering::models::FilteredReason;

    fn deadline_in(budget: Duration) -> tokio::time::Instant {
        tokio::time::Instant::now() + budget
    }

    #[tokio::test]
    async fn reference_compare_harness_not_built_without_flag() {
        let harness =
            build_reference_compare_harness("atla", deadline_in(Duration::from_secs(1))).await;
        assert!(harness.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_recovers_from_transient_failures() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();

        let result = init_client_with_retry("test", deadline_in(Duration::from_secs(240)), || {
            attempts.set(attempts.get() + 1);
            let n = attempts.get();
            async move {
                if n < 3 {
                    Err("discovery timed out")
                } else {
                    Ok(n)
                }
            }
        })
        .await;

        assert_eq!(result, Ok(3));
        assert_eq!(attempts.get(), 3);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_gives_up_at_deadline() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let budget = Duration::from_secs(240);

        let result: Result<(), String> =
            init_client_with_retry("test", deadline_in(budget), || {
                attempts.set(attempts.get() + 1);
                async { Err("discovery timed out") }
            })
            .await;

        assert_eq!(result, Err("discovery timed out".to_string()));
        assert!(attempts.get() > 1);
        assert!(start.elapsed() < budget);
        assert!(start.elapsed() >= budget - CLIENT_INIT_MAX_BACKOFF);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_skips_attempt_when_deadline_already_passed() {
        let deadline = tokio::time::Instant::now();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let attempts = Cell::new(0u32);

        let result: Result<(), String> = init_client_with_retry("test", deadline, || {
            attempts.set(attempts.get() + 1);
            async { Err("unreachable") }
        })
        .await;

        assert_eq!(
            result,
            Err("init deadline exhausted before attempt".to_string())
        );
        assert_eq!(attempts.get(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_times_out_hanging_attempt() {
        let start = tokio::time::Instant::now();

        let result: Result<(), String> = init_client_with_retry(
            "test",
            deadline_in(CLIENT_INIT_ATTEMPT_TIMEOUT / 2),
            || async {
                std::future::pending::<()>().await;
                Err("unreachable")
            },
        )
        .await;

        assert_eq!(
            result,
            Err(format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ))
        );
        assert_eq!(start.elapsed(), CLIENT_INIT_ATTEMPT_TIMEOUT);
    }

    fn response(verdicts: Vec<Verdict>) -> FilterResponse {
        let outcomes: Vec<FilterOutcome> = verdicts
            .into_iter()
            .map(|verdict| FilterOutcome {
                tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
                verdict,
                safety_labels: None,
            })
            .collect();
        FilterResponse { outcomes }
    }

    #[test]
    fn allow_verdict_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Allow,
            decided_by: None,
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn drop_by_real_rule_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Drop(FilteredReason::AuthorIsSuspended),
            decided_by: Some("DropSuspendedAuthorRule"),
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn unresolved_author_verdict_is_not_warm() {
        let response = response(vec![Verdict::unresolved_author()]);
        assert!(!filter_tweets_response_is_warm(&response));
    }
}
