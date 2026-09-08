# Grassy label-observability masks

Phoenix's released ranking loss interprets every false action bit as a negative.
That is only valid when the producer can observe that action head. This fork
adds optional per-example, per-position, per-head observability masks without
changing the released Parquet schema or legacy X behavior when masks are absent.

## Semantics

For both mask families:

- `true`: the label is observable at the example's fixed observation cutoff;
  either a positive or a negative value is valid training evidence.
- `false`: the label is unknown, right-censored, unsupported, or padding. It
  contributes zero discrete/continuous loss and zero head-specific metrics.
- An action value of `false` plus an observability value of `false` means
  **unknown**, never a negative.

Masks are not global configuration. They vary by example and sequence position.
The raw shapes are:

| Family | In-memory Arrow column | Shape | Training batch key |
|---|---|---|---|
| Discrete | `actionObservationMaskSeqSeq` | `[B, S, A]` boolean | `candidate_seq.action_observation_mask` |
| Continuous | `continuousActionObservationMaskSeqSeq` | `[B, S, C]` boolean | `candidate_seq.continuous_action_observation_mask` |

`B` is the RecordBatch row count, `S` is the fixed 1,086-position sequence,
`A` is the configured discrete width (64 for the released nano dump), and `C`
is the continuous width (8). Matching optional keys are also carried on
`history_seq`; after the standard split their shapes are `[B, 1022, A/C]` and
`[B, 64, A/C]`. Sequence packing reshapes candidate masks in exactly the same
way as actions, to `[devices, users_per_device * candidate_slots, A/C]`.

Grassy's discrete slots 60–63 and continuous slots 5–7 are padding in the
released 64/8 model contract. Ingestion and the model both force them false,
even if a caller supplies true. Padding positions created by batching or model
alignment also receive all-false masks.

## Keep the 36-column base file exact

The two Arrow columns are optional and are not written into Grassy's upstream-
schema-exact 36-column Parquet file. The production exporter stores immutable
NumPy masks and opaque ordered row identities beside each base file. With
`require_label_observation_masks=True`, `PhoenixDataset` automatically finds
`batch_N.observability.json` beside `batch_N.parquet`, resolves every referenced
path relative to the dataset root, validates the root and sidecar manifests,
and attaches the exact mask row window before `from_record_batch`.

The automatic loader verifies the backend dataset-manifest v4 and exporter v2
contract, plus the pinned upstream and sidecar-manifest versions,
all three model gates, outcome cutoff, exact 36-column base path/hash/row count, sidecar
manifest hash, ordered identity hash/count/uniqueness/rollout provenance, mask
hashes, boolean dtypes, exact `[rows,1086,64]` / `[rows,1086,8]` shapes, and
reserved-false heads. Absolute paths, parent traversal, symlink escapes,
missing files, and all binding mismatches fail closed. Arrays are memory mapped;
each `LazyRecordBatchIterator` retains its own absolute row offset, so partial
resume skips, file interleaving, and first-batch prefetch use the correct slice.

The lower-level bridge remains available for isolated tooling:

```python
from xrex.data.recsys.observability import append_observation_mask_columns

masked_record_batch = append_observation_mask_columns(
    base_record_batch,
    action_mask=sidecar_action_mask,                 # bool[B, 1086, 64]
    continuous_action_mask=sidecar_continuous_mask, # bool[B, 1086, 8]
)
```

Pass `masked_record_batch` to the existing `from_record_batch` path. The helper
does not mutate `base_record_batch`; it only appends columns to the returned
in-memory batch. It rejects non-boolean values, wrong rank, row/sequence/head
count mismatches, and duplicate mask columns. The reader also rejects non-fixed
Arrow list types and any outer, inner, or scalar null.

The production reader never joins independently rewritten files by row position
alone: all aligned files and the ordered opaque identities are content-hash
bound before any row is read.

## Loss and metric behavior

- Discrete observability multiplies the existing per-head loss mask. Existing
  negative-feedback and click-conditioned masks still apply after it.
- Each action metric is evaluated only where all action heads that define that
  metric are observable. (Current metric groups use one action per metric.)
- Continuous observability is applied independently for each configured
  continuous head to its loss and every metric slice.
- Candidate padding is still governed by Phoenix's existing padding mask; mask
  padding is additionally false.
- Cross-user and global synthetic negatives have no exact outcome authority
  for the receiving user. Their per-head masks and trained-candidate masks stay
  false. Grassy's production recipe creates no sampled negatives.
- If a mask family is absent, the batch omits its optional key and the model
  follows the original all-heads-observed code path. This preserves existing X
  training and inference input structure.

An explicit all-true Grassy sidecar reproduces legacy behavior for all semantic
heads; the seven reserved padding heads remain forcibly unobserved by contract.

## Training gate

A Grassy dump is training-eligible only when its base and sidecar manifests are
version-pinned together and every candidate position has explicit observability
for both mask families. Manifest v4 requires this exact model contract:

- `require_label_observation_masks=True`;
- `enable_engagement_counts=False`; and
- `compute_post_unexplored_label=False`.

The first parameter is propagated to the Python
offline-Parquet dataset reader and the model: the reader rejects a missing or
invalid sidecar before conversion, and the model independently rejects a batch
if either optional key is absent. Do not omit the masks for Grassy: absence
intentionally means legacy X semantics, not "nothing is observable."

The latter two gates are equally load-bearing. Feature snapshot v3 carries
like/reply/repost/quote as exact zeros explicitly marked unavailable; enabling
engagement counts would reinterpret those sentinels as observed model input.
Post-unexplored derivation depends on count columns Grassy does not claim to
observe and therefore also stays disabled. The named
`grassy_home_direct_packed_nano_offline_kafka_dump` job pins all three gates
and `num_kafka_partitions=1` together.

When the parameter is false, Phoenix never probes for or loads Grassy sidecars;
released X datasets retain their original schema and all-heads-observed
behavior unchanged.

For the current single-partition Grassy export, use the named job above. The
partition parameter is still configurable on the generic factory (and still
defaults to X's 1,024), so legacy jobs are unchanged.

## Runtime update to September 2026 upstream

The runtime is based on upstream `902a06fd616ed815f660e5546d16d492fa1ca825`
(2026-09-04). Legacy dataset v4/exporter v2 still pins its original
`a389166f6cf5da70a286b568c87695d4dcdce3a1` provenance. Advancing the runtime
never relabels, promotes, or rewrites an existing dataset contract.

The named Grassy job additionally requires
`grassy_feature_policy=grassy_unavailable_features_disabled_v1`. At model
construction, the validator rejects re-enabling unavailable counts, IP,
demographic/location/app context, timezone/local time, bridge probability,
or either follow relationship feature. These gates apply to the nested
`FeaturePrepConfig` as well as the legacy count/IP routes. Upstream defaults
now enable several of those inputs, so disabling only the legacy count flag
is insufficient.

Upstream's required-column inventory now includes
`isAuthorFollowedByViewerSeq` and `isAuthorFollowingViewerSeq`. Grassy does
not invent observed false values or rewrite its 36-column artifacts to meet
the streaming schema. Its Python offline reader accepts missing relationship
columns; the validated Grassy model never reads their fallback storage.
Only the Python `offline_kafka_dump` reader is registered for Grassy. The
`aggregated_kafka` factory is live Kafka and does not bind sidecars; it is excluded.

A fixed zero context token reserves the second prefix position required by
1,022 history positions and the upstream 1,024-position attention geometry.
It reads no unavailable context and creates no learned demographic embedding.
This changes the model feature policy and requires a fresh training run, not
reuse of an earlier checkpoint with silently different input semantics.

Upstream's new `trained_candidate_mask` remains a candidate-selection mask.
It is carried alongside Grassy's independent per-head masks, including through
batch padding. Continuous loss applies both upstream's source/candidate mask
and each head's observation mask. Conversion-source split training remains
disabled because Grassy has no authoritative delayed conversion-source data.

The named starter recipe follows the current upstream nano Muon/sparse-AdaGrad
configuration but caps its default at 100 optimizer updates (6,368 nominal
samples, batch size 32, upstream's 100-step offset gives `max_steps=99`; the
zero-based loop stops only when `step > max_steps`). This is a smoke-run bound,
not training authorization or a quality threshold. Production orchestration
must also enforce accelerator, wall-clock, spend, immutable data, and complete
serving-bundle gates. CPU contract tests do not validate CUDA execution,
convergence, or a deployable inference bundle.

## Candidate age is a separately versioned policy

Latest Home Mixer still filters at 48 hours. Phoenix's age feature itself can
represent older posts: age beyond 4,800 minutes uses an overflow bucket and
the stale-post feature is derived from actual creation/impression timestamps
at 1,213,200 seconds (14 days plus one hour). The stale candidate-count path
zeros old counts; Grassy disables all count embeddings anyway.

A 90-day dataset may therefore be introduced only through a new explicit
snapshot/export/model/serving contract with true immutable creation timestamps,
closed outcome windows, and age-stratified evaluation. Keep legacy 48-hour
v3 feature/v4 dataset admission unchanged. The overflow bucket does not
distinguish a 4-day post from a 90-day post except for the stale flag, so model
quality for old candidates must be measured before they influence live order.


## V5 source and real representation admission

Dataset v5/exporter v3/sidecar v2 add immutable positive-operation authority,
frozen close claims, explicit per-producer protected-head cohorts, stable
HMAC full-generation identities and physical user-level train/validation/test
splits. All original raw source receipts remain pinned. The v4 reader branch
continues to validate its unchanged historical contract separately.

The real ranking recipe consumes SID6; its `multimodal_embedding_type=None`.
Real 1024-dimensional MM embeddings feed the trained SID materializer. There is
no training-time requirement for an MM or SID gRPC service. Grassy training
requires both `PhoenixDataset.grassy_representation_manifest_path` and
`PhoenixDataset.grassy_representation_manifest_sha256`. These are explicit
operator inputs, with no environment or synthetic fallback. The loader is:

```python
from xrex.data.recsys.grassy_representations import load_grassy_representations
provider = load_grassy_representations(
    dataset_root=train_directory,
    manifest_path=representation_manifest,
    expected_sha256=admitted_representation_sha256,
)
```

It verifies the source dataset index and each split's generation map, snapshot
SHA/generation, identity registry, pinned real encoder and train-only codebook,
all representation file hashes, MM geometry/normalization, vector hashes and
SID6 code ranges. Each nonpadding history or candidate must match an exact
model-post/full-generation/author tuple. `artifactPurpose=serving_extension`
cannot enter training. `provider.attach(batch)` adds raw `semanticIdSeq`
0..255 codes in memory; upstream adds its own one-based sentinel offset.
The immutable base stays exactly 36 columns. Unknown or missing representations
are fatal before any training batch, never silently all-zero Semantic IDs.

This admission validates artifact provenance/contracts; it does not claim that
an operator-selected model has useful predictive quality. Upstream ranking
`eval_every_n` is unimplemented, so bounded production orchestration must run
separate held-out checkpoint evaluation and promotion gates. The reference
`train_synth.py` disables all evals and cannot serve as the production launcher.


## Exact exposure objective and sampled negatives

The v5 recipe pins `dataset.num_negatives_per_example=0`,
`dataset.num_global_negatives_per_example=0`, and
`model_config.log_q_correction=False`. These are explicit v5 manifest gates.
There is no separate model negative-count field: both dataset and model retain
64 candidate slots. Raw sequences stay 1022+64=1086; model attention reserves
its two prefix slots separately (1088 total). The exporter retains every exact
row and does not impose cross-user adjacency or balance users.

Both generic sampling helpers now keep masked synthesized slots' discrete and
continuous observation masks false and their `trained_candidate_mask` false.
They never copy a donor user's label authority. Unmasked X datasets preserve
their existing sampling behavior. The current ranking loss is supervised
per-head multihot loss; no distinct contrastive auxiliary objective has been
admitted for Grassy. Unknown unexposed posts cannot be labeled negatives.

## Prospective V6 / 90-day model

V6 is a separate reader branch for snapshot V2, feature4, proof2, immutable
admission policy and proof-bound served content. It requires sidecar3 and the
exact `model_policy` tuple sealed into the dataset/index. Its base model is
`grassy_home_direct_packed_nano_90d`, with training config
`grassy_home_direct_packed_nano_90d_offline_kafka_dump`.
`grassy_post_age_30h_80bins_v1` explicitly sets both age granularity1800 minutes
and maximum144000 minutes in feature preprocessing and the model. Legacy maximum
4800 remains unchanged. CPU regression checks actual buckets at 1h/48h/7d/30d/
90d/100d and saturation; changing granularity alone cannot extend the maximum.

Generation maps additionally seal globally consistent content references and one
representative per full post generation. The representation loader checks that
representative's hash and full source revision. Only V6 with caption capture can
use `qwen3_vl_caption_truncate_l2_1024_v1`/`caption_only`; text capture requires the
existing text encoder pair. No historical rows are projected into new policies,
no existing base36 bytes are edited, and all nonpadding positions still require
real SID6. Sampled negative counts remain zero. Age-stratified heldout quality
and independent restored-checkpoint inference remain rollout prerequisites.
