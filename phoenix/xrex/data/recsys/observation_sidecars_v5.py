# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""Strict v5 eight-source/producer/split reader; legacy v4 remains separate."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from xrex.data.recsys import observation_sidecars as base
from xrex.data.recsys.grassy_contract import GRASSY_FEATURE_POLICY, GRASSY_RUNTIME_UPSTREAM_COMMIT

SOURCE_CONTRACT = "grassy_phoenix_dataset_v5"
IDENTITY_CONTRACT = "grassy_hmac_identity_user_split_v1"
POSITIVE_AUTHORITY = "atomic_product_exact_exposure_v1"
PROTECTED_HEADS = [1, 2, 4, 7, 34, 36]
COLLECTIONS = {
    "FeedImpressions", "FeedLabelEvents", "FeedExposureFeatures", "FeedExposureSlateProofs",
    "FeedExposureSlateEntryClaims", "FeedPositiveActionOperations", "FeedImpressionCloseClaims",
}
AUTHORITY = {
    "contract": "grassy_protected_actions_close_claims_v1", "legacyLabelContractVersion": 2,
    "protectedActionAuthority": POSITIVE_AUTHORITY, "protectedActionSchemaVersion": 1,
    "protectedActionOrdinals": PROTECTED_HEADS, "closeAuthority": "immutable_close_claim_payload_v1",
    "candidateAgePolicy": "legacy_48h_v1",
}
MODEL_GATES = {**base.GRASSY_REQUIRED_MODEL_GATES, "grassy_feature_policy": GRASSY_FEATURE_POLICY,
    "num_negatives_per_example": 0, "num_global_negatives_per_example": 0, "log_q_correction": False}


def _hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def load_v5(*, base_path: Path, root: Path, rows: int, columns: int) -> base.ObservationMaskSidecar:
    require = base._require
    require(base_path.is_relative_to(root) and columns == 36 and rows > 0, "v5 base geometry/path mismatch")
    sidecar_path = base_path.with_suffix(".observability.json")
    sidecar, sidecar_bytes = base._read_json(sidecar_path, "v5 sidecar")
    base._require_exact_keys(sidecar, {
        "manifestVersion", "exporterVersion", "upstreamCommit", "labelContractVersion", "outcomeCutoffMs",
        "rows", "split", "observationAuthority", "base", "rowIdentity", "actionObservation",
        "continuousActionObservation", "rolloutsUsed", "protectedActionRollouts", "identityRegistry",
        "postGenerationIdentities",
    }, "v5 sidecar")
    require(sidecar["manifestVersion"] == 2 and sidecar["exporterVersion"] == 3
            and sidecar["upstreamCommit"] == GRASSY_RUNTIME_UPSTREAM_COMMIT and sidecar["labelContractVersion"] == 2
            and sidecar["rows"] == rows and sidecar["observationAuthority"] == AUTHORITY,
            "v5 sidecar authority/version mismatch")
    manifest, _ = base._read_json(root / base.GRASSY_DATASET_MANIFEST, "v5 dataset")
    require(manifest.get("manifest_version") == 5 and manifest.get("source_contract") == SOURCE_CONTRACT
            and manifest.get("upstream_commit") == GRASSY_RUNTIME_UPSTREAM_COMMIT
            and manifest.get("training_eligible") is True and manifest.get("required_model_gates") == MODEL_GATES
            and manifest.get("observation_authority") == AUTHORITY
            and manifest.get("candidate_age_policy") == "legacy_48h_v1",
            "v5 dataset/model authority mismatch")
    exporter = base._object(manifest.get("exporter"), "v5 exporter")
    base._require_exact_keys(exporter, {"name", "version", "input_schema_version", "source_sha256",
        "base_adapter_sha256", "legacy_validator_sha256", "identity_mapper_sha256"}, "v5 exporter")
    require(exporter["name"] == "grassy-feed-eight-source-to-phoenix" and exporter["version"] == 3
            and exporter["input_schema_version"] == 2, "v5 exporter pin mismatch")
    for key, value in exporter.items():
        if key.endswith("sha256"):
            require(_hex(value), "v5 exporter source hash invalid")
    require(manifest.get("shape") == {"history_positions": 1022, "candidate_positions": 64,
        "sequence_positions": 1086, "discrete_action_width": 64, "continuous_action_width": 8,
        "base_columns": 36}, "v5 shape mismatch")
    split = sidecar["split"]
    require(split in {"train", "validation", "test"} and manifest.get("split") == split
            and manifest.get("rows") == rows and manifest.get("action_contract_version") == 2
            and manifest.get("outcome_cutoff_ms") == sidecar["outcomeCutoffMs"], "v5 split/cutoff mismatch")
    require(base._safe_dataset_path(root, manifest.get("parquet"), "v5 parquet") == base_path
            and base._safe_dataset_path(root, manifest.get("sidecar_manifest"), "v5 sidecar") == sidecar_path
            and manifest.get("sidecar_manifest_sha256") == base._sha256_bytes(sidecar_bytes),
            "v5 manifest sidecar/path binding mismatch")
    binding = base._object(sidecar["base"], "v5 base binding")
    base._require_exact_keys(binding, {"path", "sha256", "columns"}, "v5 base binding")
    digest = base._sha256_file(base_path)
    require(binding["columns"] == 36 and base._safe_dataset_path(root, binding["path"], "v5 base") == base_path
            and binding["sha256"] == digest and manifest.get("parquet_sha256") == digest, "v5 base hash mismatch")
    source_binding = base._object(manifest.get("source_snapshot"), "v5 snapshot")
    source_path = base._safe_dataset_path(root, source_binding.get("path"), "v5 source snapshot")
    require(source_path.is_file() and base._sha256_file(source_path) == source_binding.get("sha256")
            and re.fullmatch("[1-9][0-9]*", str(source_binding.get("generation", ""))), "v5 snapshot seal mismatch")
    source, _ = base._read_json(source_path, "v5 source snapshot")
    require(source.get("contract") == "grassy_feed_training_snapshot_v1" and source.get("complete") is True
            and source.get("trainingEligible") is False and source.get("outcomeCutoffMs") == sidecar["outcomeCutoffMs"]
            and source.get("snapshotId") == source_binding.get("snapshotId"), "v5 source contract mismatch")
    sources = base._object(manifest.get("source_snapshots"), "v5 selected sources")
    require(set(sources) == COLLECTIONS, "v5 eight-source inventory mismatch")
    source_objects = base._object(source.get("objects"), "v5 source objects")
    required_source_paths = {f"{kind}/{collection}.jsonl" for kind in ("raw", "selected") for collection in COLLECTIONS}
    required_source_paths |= {"ObservabilityRollouts.json", "rollout-attestation.json", "selection-report.json",
                              "user-split-assignments.jsonl", "exposure-identities.jsonl"}
    require(required_source_paths <= set(source_objects), "v5 snapshot lacks retained raw authority")
    for collection in COLLECTIONS:
        require(sources[collection] == source_objects.get(f"selected/{collection}.jsonl"),
                "v5 selected source receipt mismatch")
    require(manifest.get("observability_rollout_sha256") == source.get("rollout", {}).get("sha256"),
            "v5 rollout snapshot mismatch")
    registry = base._object(sidecar["identityRegistry"], "v5 identity registry")
    require(registry.get("contract") == IDENTITY_CONTRACT and registry.get("splitNamespace") == "global_user_8000_1000_1000_v1"
            and isinstance(registry.get("keyId"), str) and bool(registry["keyId"]) and _hex(registry.get("keySha256"))
            and isinstance(registry.get("projectId"), str) and bool(registry["projectId"])
            and manifest.get("identity_registry") == registry and source.get("identity") == registry,
            "v5 stable identity registry mismatch")
    require(sidecar["postGenerationIdentities"] == manifest.get("post_generation_identities"),
            "v5 post generation identity binding mismatch")
    from xrex.data.recsys.generation_identities import load_generation_identities
    load_generation_identities(root, manifest)
    identity = base._object(sidecar["rowIdentity"], "v5 row identity")
    base._require_exact_keys(identity, {"path", "sha256", "encoding"}, "v5 row identity")
    identity_path = base._safe_dataset_path(root, identity["path"], "v5 row identity")
    require(identity["encoding"] == "jsonl-hmac-sha256-user-split-producer-v1"
            and base._sha256_file(identity_path) == identity["sha256"]
            and manifest.get("row_identity_sha256") == identity["sha256"], "v5 row identity binding mismatch")
    action = base._load_mask(root=root, sidecar=sidecar, section_name="actionObservation",
                            expected_shape=(rows, 1086, 64), expected_forced_false_heads=[60, 61, 62, 63])
    continuous = base._load_mask(root=root, sidecar=sidecar, section_name="continuousActionObservation",
                                expected_shape=(rows, 1086, 8), expected_forced_false_heads=[5, 6, 7])
    require(not action[:, :, 0].any() and not continuous[:, :, 0].any(), "v5 sentinel head observable")
    protected = base._array(sidecar["protectedActionRollouts"], "v5 protected rollouts")
    by_producer = {}
    evidence = {item["origin"]["sha256"] for item in source.get("rollout", {}).get("evidenceOrigins", [])}
    for item in protected:
        item = base._object(item, "v5 producer")
        base._require_exact_keys(item, {"releaseId", "platform", "appVersion", "effectiveFromMs", "effectiveToMs",
            "surfaces", "discreteOrdinals", "producerAuthority", "producerSchemaVersion", "evidenceSha256"}, "v5 producer")
        require(type(item["effectiveFromMs"]) is int and type(item["effectiveToMs"]) is int
                and 0 < item["effectiveFromMs"] < item["effectiveToMs"], "v5 producer range invalid")
        require(item.get("producerAuthority") == POSITIVE_AUTHORITY and item.get("producerSchemaVersion") == 1
                and isinstance(item.get("releaseId"), str) and item["releaseId"] not in by_producer
                and isinstance(item.get("discreteOrdinals"), list) and bool(item["discreteOrdinals"])
                and all(type(head) is int for head in item["discreteOrdinals"])
                and len(item["discreteOrdinals"]) == len(set(item["discreteOrdinals"]))
                and set(item["discreteOrdinals"]) <= set(PROTECTED_HEADS)
                and bool(item.get("evidenceSha256")) and set(item["evidenceSha256"]) <= evidence,
                "v5 protected producer evidence/head mismatch")
        by_producer[item["releaseId"]] = item
    known_rollouts = {}
    for item in base._array(sidecar["rolloutsUsed"], "v5 rollouts"):
        item = base._object(item, "v5 rollout")
        key = (item.get("releaseId"), item.get("platform"), item.get("appVersion"))
        require(key not in known_rollouts and isinstance(item.get("surfaces"), list) and item["surfaces"],
                "v5 rollout provenance collision")
        base._require_exact_keys(item, {"releaseId", "platform", "appVersion", "effectiveFromMs", "effectiveToMs", "surfaces"}, "v5 rollout")
        require(type(item["effectiveFromMs"]) is int and type(item["effectiveToMs"]) is int
                and 0 < item["effectiveFromMs"] < item["effectiveToMs"], "v5 rollout range invalid")
        known_rollouts[key] = item
    identities = [json.loads(line) for line in identity_path.read_text().splitlines()]
    require(len(identities) == rows, "v5 identity row count mismatch")
    geometry = pq.read_table(base_path, columns=["length", "impressedTimeMsSeq"]).to_pydict()
    lengths = geometry["length"]
    seen = set()
    for index, item in enumerate(identities):
        base._require_exact_keys(item, {"exampleId", "userId", "split", "rolloutReleaseId", "protectedReleaseId",
                                        "platform", "appVersion", "surface"}, "v5 row identity")
        require(_hex(item["exampleId"]) and item["exampleId"] not in seen and _hex(item["userId"])
                and item["split"] == split, "v5 row identity/split collision")
        seen.add(item["exampleId"])
        key = (item["rolloutReleaseId"], item["platform"], item["appVersion"])
        require(key in known_rollouts and item["surface"] in known_rollouts[key]["surfaces"], "v5 row rollout mismatch")
        candidate = lengths[index] - 1
        require(0 <= candidate < 1086, "v5 row candidate position invalid")
        occurred = geometry["impressedTimeMsSeq"][index][candidate]
        require(known_rollouts[key]["effectiveFromMs"] <= occurred < known_rollouts[key]["effectiveToMs"],
                "v5 row rollout time mismatch")
        observed = {head for head in PROTECTED_HEADS if action[index, candidate, head]}
        producer = by_producer.get(item["protectedReleaseId"])
        if producer is None:
            require(item["protectedReleaseId"] is None and not observed, "v5 old cohort gained protected labels")
        else:
            require(producer["platform"] == item["platform"] and producer["appVersion"] == item["appVersion"]
                    and item["surface"] in producer["surfaces"] and observed <= set(producer["discreteOrdinals"])
                    and producer["effectiveFromMs"] <= occurred < producer["effectiveToMs"],
                    "v5 row protected producer mismatch")
    return base.ObservationMaskSidecar(base_path=base_path, rows=rows, outcome_cutoff_ms=sidecar["outcomeCutoffMs"],
                                      action_mask=action, continuous_action_mask=continuous)
