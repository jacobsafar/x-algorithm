use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::hydration::batch::{AuthorHydrationBatch, HydrationBatch, TweetHydrationBatch};
use crate::hydration::fallback_cache::FallbackCache;
use crate::hydration::metrics::{record_batch_size, timed_results};
use crate::hydration::{keyed_by_author, tweets_per_author};
use crate::models::{AuthorFeatures, AuthorId, TweetCandidateInput, UserLabelSet};
use crate::rules::SafetyLevel;
use std::time::Duration;
use xai_core_entities::entities::GizmoduckUserResult;
use xai_core_entities::gizmoduck_client::QueryFields;
use xai_x_thrift::user_labels::LabelValue;

const CLIENT_TIMEOUT: Duration = Duration::from_millis(150);
const CLIENT: &str = "gizmoduck";
const CACHE_CAPACITY: usize = 1_000_000;

pub struct GizmoduckAuthorHydrator {
    pub gizmoduck_client: GizmoduckLookup,
    fallback_cache: Option<FallbackCache<AuthorId, AuthorFeatures>>,
}

impl GizmoduckAuthorHydrator {
    pub(crate) fn new(
        gizmoduck_client: GizmoduckLookup,
        fallback_cache: Option<FallbackCache<AuthorId, AuthorFeatures>>,
    ) -> Self {
        Self {
            gizmoduck_client,
            fallback_cache,
        }
    }

    pub(crate) fn fallback_cache() -> FallbackCache<AuthorId, AuthorFeatures> {
        FallbackCache::new("author", CACHE_CAPACITY)
    }

    pub(crate) async fn hydrate(
        &self,
        candidates: &[TweetCandidateInput],
        safety_level: SafetyLevel,
    ) -> TweetHydrationBatch<AuthorFeatures> {
        let cache_request = self
            .fallback_cache
            .as_ref()
            .map(|cache| (cache, cache.begin_request()));
        let candidate_count_by_key = tweets_per_author(candidates);
        let author_ids: Vec<u64> = candidate_count_by_key.keys().map(|a| a.get()).collect();

        let user_results: AuthorHydrationBatch<GizmoduckUserResult> = if author_ids.is_empty() {
            HydrationBatch::empty()
        } else {
            record_batch_size(CLIENT, author_ids.len());
            timed_results(
                CLIENT,
                "get_users",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                async {
                    let response = self
                        .gizmoduck_client
                        .get_users(author_ids, &[QueryFields::SAFETY, QueryFields::LABELS])
                        .await;
                    keyed_by_author(&candidate_count_by_key, response)
                },
            )
            .await
        };

        let author_features = user_results.map(author_features);
        let author_features = if let Some((cache, generation)) = cache_request {
            cache.resolve_hydration_batch(generation, author_features)
        } else {
            author_features
        };
        author_features.project(candidates.iter().map(|c| (c.tweet_id, c.author_id)))
    }
}

fn author_features(user_result: GizmoduckUserResult) -> AuthorFeatures {
    user_result
        .user
        .map(|user| AuthorFeatures {
            is_suspended: user.safety.suspended,
            is_deactivated: user.safety.deactivated,
            is_protected: user.safety.is_protected,
            is_nsfw_user: user.safety.nsfw_user,
            is_nsfw_admin: user.safety.nsfw_admin,
            is_erased: user.safety.erased,
            is_offboarded: user.safety.offboarded,
            user_labels: UserLabelSet::new(
                user.labels
                    .labels
                    .iter()
                    .map(|label| LabelValue::from(label.label_value))
                    .collect(),
            ),
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::Hydrated;
    use crate::models::{resolve_candidate, RawCandidate, TweetId};
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use xai_core_entities::entities::{GizmoduckUser, PCFLabel, PureCoreData, Safety};
    use xai_core_entities::gizmoduck_client::{
        GizmoduckClient, MockGizmoduckClient, UserFields, ViewerData,
    };

    struct FailingAfterFirstClient {
        calls: AtomicUsize,
    }

    #[tonic::async_trait]
    impl GizmoduckClient for FailingAfterFirstClient {
        async fn get_users(
            &self,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            let succeeds = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
            user_ids
                .into_iter()
                .map(|id| {
                    let result = if succeeds {
                        Ok(Some(GizmoduckUserResult {
                            user: Some(GizmoduckUser {
                                user_id: id as u64,
                                safety: Safety {
                                    suspended: true,
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                        }))
                    } else {
                        Err(anyhow::anyhow!("gizmoduck unavailable"))
                    };
                    (id, result)
                })
                .collect()
        }

        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            unreachable!()
        }

        async fn get_viewer_roles(&self, _user_id: u64) -> Result<Vec<String>> {
            unreachable!()
        }

        async fn get_viewer_data(&self, _user_id: u64) -> Result<ViewerData> {
            unreachable!()
        }

        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> Result<ViewerData> {
            unreachable!()
        }

        async fn get_pcf_labels(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<PCFLabel>> {
            unreachable!()
        }

        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<String>>> {
            unreachable!()
        }

        async fn get_user_fields(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<UserFields>> {
            unreachable!()
        }

        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> Result<Option<GizmoduckUserResult>> {
            unreachable!()
        }
    }

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: Some(author_id),
            },
            &HashMap::<TweetId, PureCoreData>::new(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn not_found_authors_default_features_and_share_one_backend_key() {
        let client = Arc::new(MockGizmoduckClient::default());
        let hydrator = GizmoduckAuthorHydrator::new(
            GizmoduckLookup::new(client.clone()),
            Some(FallbackCache::with_test_capacity("author")),
        );
        let candidates = vec![candidate(1, 10), candidate(2, 10)];

        let features = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;

        assert_eq!(client.call_count(), 1);
        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                features.hydrated(&tweet_id),
                Some(Hydrated::NotFound)
            ));
            let feature = features.get_or_default(&tweet_id);
            assert!(!feature.is_suspended);
            assert!(!feature.is_deactivated);
            assert!(!feature.is_protected);
            assert!(!feature.is_nsfw_user);
            assert!(!feature.is_nsfw_admin);
            assert!(!feature.is_erased);
            assert!(!feature.is_offboarded);
        }
    }

    #[tokio::test]
    async fn stale_recovery_uses_resident_value() {
        let candidates = vec![candidate(1, 10)];
        let hydrator = GizmoduckAuthorHydrator::new(
            GizmoduckLookup::new(Arc::new(FailingAfterFirstClient {
                calls: AtomicUsize::new(0),
            })),
            Some(FallbackCache::with_test_capacity("author")),
        );
        let first = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;
        assert!(first.get_or_default(&TweetId(1)).is_suspended);

        let second = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;
        assert!(second.get_or_default(&TweetId(1)).is_suspended);
    }

    #[test]
    fn error_and_missing_author_results_fail_open_but_stay_failed() {
        let (a10, a20) = (candidate(1, 10).author_id, candidate(2, 20).author_id);
        let user_results: AuthorHydrationBatch<GizmoduckUserResult> = HydrationBatch::from_results(
            [a10, a20],
            HashMap::from([(a10, Err(anyhow::anyhow!("gizmoduck unavailable")))]),
        );

        let by_tweet = user_results
            .map(author_features)
            .project([(TweetId(1), a10), (TweetId(2), a20)]);

        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                by_tweet.hydrated(&tweet_id),
                Some(Hydrated::Failed(_))
            ));
            let feature = by_tweet.get_or_default(&tweet_id);
            assert!(!feature.is_suspended);
            assert!(!feature.is_deactivated);
        }
    }
}
