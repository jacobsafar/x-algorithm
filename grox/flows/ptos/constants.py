POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS = "post_min_traction_stream_for_grox_ptos"
POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS = "post_min_impression_stream_for_grox_ptos"
SAFETY_PTOS_RECOVERY = "safety_ptos_recovery"
SAFETY_PTOS_DELUXE = "safety_ptos_deluxe"
SAFETY_PTOS_BACKFILL = "safety_ptos_backfill"
SAFETY_PTOS_REALTIME_WITH_SIGNALS = "safety_ptos_realtime_with_signals"
SAFETY_PTOS_LIVE_CLUSTER_ANCHORS = "safety_ptos_live_cluster_anchors"
TOPIC_MIN_TRACTION = (
    "content-understanding-realtime-unified-posts-min-traction-for-grox-ptos"
)
TOPIC_MIN_IMPRESSION = (
    "content-understanding-realtime-unified-posts-min-impression-for-grox-ptos"
)
TOPIC_RECOVERY = "safety-ptos-recovery"
TOPIC_DELUXE = "safety-ptos-deluxe"
TOPIC_BACKFILL = "safety-ptos-backfill"
TOPIC_DELAYED_REPLICATION = (
    "content-understanding-realtime-unified-posts-delayed-replication"
)
TOPIC_LIVE_CLUSTER_ANCHORS = "safety-ptos-live-cluster-anchors"
TOPIC_LIVE_CLUSTER_ANCHOR_VERDICTS = "safety-ptos-live-cluster-anchor-verdicts"
GEMMA = "oai-gemma4-26b"
GEMMA_PTOS_REALTIME = "oai-gemma4-26b-ptos-realtime"

HIGH_FAV_THRESHOLD = 128

SAFETY_PTOS_SPECIAL_VIDEO = "safety_ptos_special_video"
TOPIC_SPECIAL_VIDEO = "safety-ptos-special-video"
DELUXE_TIER_TASK_TYPES = frozenset({SAFETY_PTOS_DELUXE, SAFETY_PTOS_SPECIAL_VIDEO})
