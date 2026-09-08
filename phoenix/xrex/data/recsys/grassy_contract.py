# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""Grassy's explicit unavailable-feature policy; independent of JAX/CUDA."""
from __future__ import annotations

GRASSY_RUNTIME_UPSTREAM_COMMIT = "902a06fd616ed815f660e5546d16d492fa1ca825"
GRASSY_FEATURE_POLICY = "grassy_unavailable_features_disabled_v1"
GRASSY_RELATIONSHIP_COLUMNS = (
    "isAuthorFollowedByViewerSeq", "isAuthorFollowingViewerSeq",
)
# These are unavailable at the immutable exposure cutoff, never observed false.
GRASSY_DISABLED_PREP_FLAGS = (
    "enable_ip_address", "enable_user_country", "enable_user_language",
    "enable_user_state", "enable_user_dma_code", "enable_user_location",
    "enable_user_gender", "enable_user_age", "enable_user_installed_apps",
    "enable_timezone", "enable_time_of_day", "enable_time_of_week",
    "enable_hour_of_day", "enable_day_of_week", "enable_bridge_prob",
    "enable_is_author_followed_by_viewer", "enable_is_author_following_viewer",
    "enable_engagement_counts",
)


def grassy_feature_prep_overrides() -> dict[str, bool]:
    return {**dict.fromkeys(GRASSY_DISABLED_PREP_FLAGS, False),
            "reserve_unavailable_user_feature_token": True}


def validate_grassy_model_contract(model: object) -> None:
    """Reject CLI/config drift before the model builds features or parameters."""
    if not getattr(model, "require_label_observation_masks", False):
        return
    if getattr(model, "grassy_feature_policy", None) != GRASSY_FEATURE_POLICY:
        raise ValueError("Grassy observation masks require the explicit feature policy")
    if not getattr(model, "feature_prep_enabled", False):
        raise ValueError("Grassy requires its validated feature-prep route")
    prep = model.feature_prep
    for name in GRASSY_DISABLED_PREP_FLAGS:
        if getattr(prep, name, None) is not False:
            raise ValueError(f"Grassy unavailable feature must remain disabled: {name}")
    if getattr(prep, "reserve_unavailable_user_feature_token", None) is not True:
        raise ValueError("Grassy requires the fixed zero context token for 1022 history slots")
    age_tuple = {
        "legacy_48h_v1": ("phoenix_post_age_1h_80bins_v1", 60, 4800),
        "grassy_candidate_age_90d_v1": ("grassy_post_age_30h_80bins_v1", 1800, 144000),
    }.get(getattr(model, "grassy_candidate_age_policy", None))
    if age_tuple is None or (
            getattr(model, "grassy_post_age_encoding", None),
            getattr(prep, "post_age_granularity_mins", None), getattr(prep, "post_age_max_mins", None)) != age_tuple:
        raise ValueError("Grassy candidate age encoding/policy mismatch")
    if (getattr(model, "post_age_granularity_mins", None), getattr(model, "post_age_max_mins", None)) != age_tuple[1:]:
        raise ValueError("Grassy nested and legacy age encoders disagree")
    if getattr(model, "use_ip_address", None) is not False:
        raise ValueError("Grassy IP inputs must remain disabled")
    if getattr(model.context_features, "enable_engagement_counts", None) is not False:
        raise ValueError("Grassy legacy engagement-count inputs must remain disabled")
    if (model.history_seq_len, model.candidate_seq_len,
            model.model_config.output_vocab_size, model.num_continuous_actions) != (1022, 64, 64, 8):
        raise ValueError("Grassy requires the versioned 1022/64 sequence and 64/8 heads")
    if getattr(model, "log_q_correction", None) is not False:
        raise ValueError("Grassy exact exposure objective does not use sampled-negative log-Q correction")
    if getattr(model, "split_head_training_by_source", False):
        raise ValueError("Grassy does not observe delayed conversion-source semantics")
