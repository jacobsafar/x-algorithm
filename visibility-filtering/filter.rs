use crate::hydration::{HydrationOutput, HydrationPipeline, HydrationRequest};
use crate::models::{RawCandidate, TweetId};
use crate::rules::metrics as ft_metrics;
use crate::rules::{RuleEngine, SafetyLevel, Verdict};
use std::collections::HashMap;
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterRequest {
    pub viewer_id: Option<u64>,
    pub country_code: Option<String>,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<RawCandidate>,
}

pub struct FilterOutcome {
    pub tweet_id: TweetId,
    pub verdict: Verdict,
    pub safety_labels: Option<vf_pb::SafetyLabelMap>,
}

pub struct FilterResponse {
    pub outcomes: Vec<FilterOutcome>,
}

pub struct FilterTweets {
    hydration_pipeline: HydrationPipeline,
    rule_engine: RuleEngine,
}

impl FilterTweets {
    pub(crate) fn new(hydration_pipeline: HydrationPipeline, rule_engine: RuleEngine) -> Self {
        Self {
            hydration_pipeline,
            rule_engine,
        }
    }

    pub async fn run(&self, request: FilterRequest) -> FilterResponse {
        let hydration = self
            .hydration_pipeline
            .hydrate(HydrationRequest::new(
                request.viewer_id,
                request.country_code,
                &request.candidates,
                request.safety_level,
            ))
            .await;
        let HydrationOutput {
            viewer_features,
            candidates: hydrated_candidates,
            safety_labels,
        } = hydration;
        let evaluated: HashMap<TweetId, Verdict> = hydrated_candidates
            .iter()
            .map(|candidate| {
                (
                    TweetId(candidate.tweet_id),
                    self.rule_engine
                        .evaluate(request.safety_level, &viewer_features, candidate),
                )
            })
            .collect();

        let outcomes: Vec<FilterOutcome> = request
            .candidates
            .iter()
            .map(|candidate| {
                let verdict = evaluated
                    .get(&candidate.tweet_id)
                    .cloned()
                    .unwrap_or_else(Verdict::unresolved_author);
                FilterOutcome {
                    tweet_id: candidate.tweet_id,
                    verdict,
                    safety_labels: safety_labels
                        .get(&candidate.tweet_id)
                        .map(|labels| vf_pb::SafetyLabelMap::clone(labels)),
                }
            })
            .collect();

        ft_metrics::record_verdicts(
            request.safety_level,
            outcomes.iter().map(|outcome| &outcome.verdict),
        );

        FilterResponse { outcomes }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::clients::socialgraph_client::FakeSocialgraphClient;
    use crate::safety_label_source::lookup::{ManhattanLookup, RemoteSource, TwemcacheLookup};
    use crate::safety_label_source::types::{ManhattanOutcome, TwemcacheOutcome};
    use crate::safety_label_source::SafetyLabelSource;
    use std::sync::Arc;
    use tonic::async_trait;
    use xai_core_entities::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient};
    use xai_core_entities::tweet_entity_service_client::{MockTESClient, TESClient};

    fn full_label_map() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        }
    }

    struct FakeTwemcache;

    #[async_trait]
    impl TwemcacheLookup for FakeTwemcache {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, TwemcacheOutcome> {
            ids.iter()
                .copied()
                .map(|id| {
                    let outcome = if id == 2 {
                        TwemcacheOutcome::Hit(full_label_map())
                    } else {
                        TwemcacheOutcome::Miss
                    };
                    (id, outcome)
                })
                .collect()
        }
    }

    struct FakeManhattan;

    #[async_trait]
    impl ManhattanLookup for FakeManhattan {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, ManhattanOutcome> {
            ids.iter()
                .copied()
                .map(|id| (id, ManhattanOutcome::Resolved(full_label_map())))
                .collect()
        }
    }

    pub(crate) fn filter_tweets() -> FilterTweets {
        filter_tweets_with_gizmoduck(Arc::new(MockGizmoduckClient::default()))
    }

    pub(crate) fn filter_tweets_with_gizmoduck(
        gizmoduck: Arc<dyn GizmoduckClient + Send + Sync>,
    ) -> FilterTweets {
        let tes: Arc<dyn TESClient + Send + Sync> = Arc::new(MockTESClient::default());
        let socialgraph = Arc::new(FakeSocialgraphClient);
        let twemcache = Arc::new(FakeTwemcache);
        let manhattan = Arc::new(FakeManhattan);
        let labels = Arc::new(SafetyLabelSource::new(Arc::new(RemoteSource::new(
            twemcache, manhattan,
        ))));

        FilterTweets::new(
            HydrationPipeline::new(tes, gizmoduck, socialgraph, labels, None),
            RuleEngine::for_tests(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::test_support::filter_tweets;
    use crate::models::VfAction;

    fn candidate(tweet_id: u64, author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: author_id,
        }
    }

    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_authors_and_labels() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![
                    candidate(2, Some(20)),
                    candidate(1, None),
                    candidate(2, Some(20)),
                ],
            })
            .await;

        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.tweet_id)
                .collect::<Vec<_>>(),
            vec![TweetId(2), TweetId(1), TweetId(2)]
        );
        assert!(matches!(
            response.outcomes[0].verdict.action,
            VfAction::Allow
        ));
        assert!(matches!(
            response.outcomes[1].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            response.outcomes[1].verdict.decided_by,
            Some("unresolved_author_id")
        );
        assert!(matches!(
            response.outcomes[2].verdict.action,
            VfAction::Allow
        ));
        assert!(response
            .outcomes
            .iter()
            .all(|outcome| outcome.safety_labels.is_some()));
        assert!(!response.outcomes[0]
            .safety_labels
            .as_ref()
            .unwrap()
            .labels
            .is_empty());
        assert_eq!(
            response.outcomes[0].safety_labels,
            response.outcomes[2].safety_labels
        );
    }

    #[tokio::test]
    async fn run_selects_policy_from_safety_level() {
        let service = filter_tweets();
        let request = |safety_level| FilterRequest {
            viewer_id: None,
            country_code: None,
            safety_level,
            candidates: vec![candidate(1, Some(10))],
        };

        let home = service.run(request(SafetyLevel::TimelineHome)).await;
        let filter_all = service.run(request(SafetyLevel::FilterAll)).await;

        assert!(matches!(home.outcomes[0].verdict.action, VfAction::Allow));
        assert!(matches!(
            filter_all.outcomes[0].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            filter_all.outcomes[0].verdict.decided_by,
            Some("FilterAllRule")
        );
    }
}
