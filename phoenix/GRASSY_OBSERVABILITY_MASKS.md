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
- Cross-user synthetic negatives inherit the sampled source example's head
  availability. Global negatives inherit the current example's first-candidate
  availability. This retains deliberate negative sampling for supported heads
  without re-enabling unavailable heads.
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

The latter two gates are equally load-bearing. Feature snapshot v2 carries
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
