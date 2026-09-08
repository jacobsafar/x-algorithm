# SPDX-License-Identifier: Apache-2.0
"""Admit real immutable representation artifacts and attach raw SID6 in memory."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from xrex.data.recsys import observation_sidecars as base
from xrex.data.recsys.generation_identities import load_generation_identities
from xrex.data.recsys.grassy_contract import GRASSY_RUNTIME_UPSTREAM_COMMIT

CONTRACT = "grassy_phoenix_representations_v1"
ARTIFACTS = {"records", "mmSnapshot", "sidSnapshot", "featureAtlas", "nativeSid", "postIds", "embeddings"}


def _hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


@dataclass(frozen=True)
class GrassyRepresentations:
    records: dict[str, dict]
    manifest_sha256: str

    def attach(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        require = base._require
        require("semanticIdSeq" not in batch.schema.names, "preexisting SID column lacks admitted provenance")
        values = {key: batch.column(key).to_pylist() for key in ("tweetIdSeq", "authorIdSeq", "paddingMask")}
        sids = np.zeros((batch.num_rows, 1086, 6), dtype=np.int16)
        for row, (posts, authors, active) in enumerate(zip(values["tweetIdSeq"], values["authorIdSeq"], values["paddingMask"], strict=True)):
            require(len(posts) == len(authors) == len(active) == 1086, "SID source sequence shape mismatch")
            for pos, (post, author, valid) in enumerate(zip(posts, authors, active, strict=True)):
                if not valid:
                    require(post == 0 and author == 0, "SID padding contains unadmitted identity")
                    continue
                record = self.records.get(str(post))
                require(record is not None and record["modelAuthorId"] == str(author),
                        "missing exact nonpadding SID generation/author")
                sids[row, pos] = record["semanticIds"]
        per_post = pa.FixedSizeListArray.from_arrays(pa.array(sids.reshape(-1), type=pa.int16()), 6)
        per_row = pa.FixedSizeListArray.from_arrays(per_post, 1086)
        return batch.append_column("semanticIdSeq", per_row)


def load_grassy_representations(*, dataset_root: Path | str, manifest_path: Path | str,
                               expected_sha256: str) -> GrassyRepresentations:
    """CPU-only full admission; no downloads, inference, random codes or missing-ID fallback."""
    require = base._require
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    require(_hex(expected_sha256) and base._sha256_file(manifest_path) == expected_sha256,
            "representation manifest hash mismatch")
    manifest, _ = base._read_json(manifest_path, "representations")
    base._require_exact_keys(manifest, {"contract", "schemaVersion", "sourceDatasetIndexSha256",
        "sourceSnapshotSha256", "sourceSnapshotGeneration", "identityRegistry", "upstreamCommit", "synthetic",
        "encoder", "contentSnapshotManifestSha256", "generationMaps", "codebook", "artifacts", "counts", "artifactPurpose"}, "representations")
    require(manifest["contract"] == CONTRACT and type(manifest["schemaVersion"]) is int
            and manifest["schemaVersion"] == 1 and manifest["synthetic"] is False
            and manifest["artifactPurpose"] == "training"
            and manifest["upstreamCommit"] == GRASSY_RUNTIME_UPSTREAM_COMMIT
            and _hex(manifest["contentSnapshotManifestSha256"])
            and isinstance(manifest["sourceSnapshotGeneration"], str)
            and re.fullmatch("[1-9][0-9]*", manifest["sourceSnapshotGeneration"]),
            "representation contract/provenance mismatch")
    index_root = dataset_root.parent
    index_path = index_root / "dataset-index.json"
    require(base._sha256_file(index_path) == manifest["sourceDatasetIndexSha256"], "representation dataset index mismatch")
    index, _ = base._read_json(index_path, "representation source index")
    source_contract = index.get("sourceContract")
    require(source_contract in {"grassy_phoenix_dataset_v5", "grassy_phoenix_dataset_v6"}
            and index.get("sourceSnapshotSha256") == manifest["sourceSnapshotSha256"]
            and index.get("identityRegistry") == manifest["identityRegistry"], "representation source identity mismatch")
    require(dataset_root.name in {"train", "validation", "test"}, "representation split root invalid")
    expected = {}
    train_posts = set()
    split_maps = base._object(manifest["generationMaps"], "representation generation maps")
    nonempty = {name for name, item in index["splits"].items() if item["rows"] > 0}
    require(set(split_maps) == nonempty and dataset_root.name in nonempty, "representation split map inventory mismatch")
    for split in sorted(nonempty):
        require(split in {"train", "validation", "test"}, "representation unknown split")
        entry = index["splits"][split]
        path = base._safe_dataset_path(index_root, entry["manifest"], "representation split manifest")
        require(path.parent == index_root / split and base._sha256_file(path) == entry["sha256"],
                "representation split manifest hash mismatch")
        dataset, _ = base._read_json(path, "representation split dataset")
        require(dataset.get("split") == split and dataset.get("source_contract") == source_contract
                and dataset.get("identity_registry") == manifest["identityRegistry"]
                and dataset.get("source_snapshot", {}).get("sha256") == manifest["sourceSnapshotSha256"]
                and dataset.get("source_snapshot", {}).get("generation") == manifest["sourceSnapshotGeneration"]
                and split_maps[split] == {**dataset.get("post_generation_identities", {}), "datasetManifestSha256": entry["sha256"]},
                "representation full-generation source binding mismatch")
        if source_contract == "grassy_phoenix_dataset_v6":
            from xrex.data.recsys.observation_sidecars_v6 import expected_model_policy
            capture = index.get("capturePolicy", {})
            require(capture == dataset.get("capture_policy")
                    and dataset.get("model_policy") == index.get("model_policy") == expected_model_policy(capture.get("renderPolicy")),
                    "representation prospective model/capture policy mismatch")
        source_records = load_generation_identities(path.parent, dataset)
        for post, record in source_records.items():
            require(post not in expected or expected[post] == record, "cross-split generation identity collision")
            expected[post] = record
        if split == "train": train_posts.update(source_records)
    encoder = base._object(manifest["encoder"], "real representation encoder")
    base._require_exact_keys(encoder, {"modelId", "revision", "inventorySha256", "referenceFiles", "contract", "modality"}, "encoder")
    require(isinstance(encoder["modelId"], str) and bool(encoder["modelId"])
            and isinstance(encoder["revision"], str) and re.fullmatch("[a-f0-9]{40}", encoder["revision"])
            and _hex(encoder["inventorySha256"])
            and (encoder["contract"], encoder["modality"]) == (
                ("qwen3_vl_caption_truncate_l2_1024_v1", "caption_only")
                if source_contract == "grassy_phoenix_dataset_v6" and index.get("capturePolicy", {}).get("renderPolicy") == "grassy_caption_only_v1"
                else ("qwen3_vl_text_truncate_l2_1024_v1", "text_only"))
            and isinstance(encoder["referenceFiles"], dict) and len(encoder["referenceFiles"]) == 4
            and all(_hex(value) for value in encoder["referenceFiles"].values()), "real encoder provenance missing")
    codebook = base._object(manifest["codebook"], "real SID codebook")
    base._require_exact_keys(codebook, {"kind", "levels", "centroids", "fitPartition", "trainMembershipSha256",
                                       "sha256", "iterations", "seed"}, "codebook")
    require(codebook["kind"] == "rq_kmeans" and codebook["levels"] == 6 and codebook["centroids"] == 256
            and codebook["fitPartition"] == "train" and codebook["seed"] == 42
            and type(codebook["iterations"]) is int and 0 < codebook["iterations"] <= 1000,
            "real train-only SID codebook contract mismatch")
    root = manifest_path.parent
    membership_path = root / "train-membership.json"
    require(base._sha256_file(membership_path) == codebook["trainMembershipSha256"]
            and json.loads(membership_path.read_text()) == sorted(train_posts, key=int), "SID codebook train membership mismatch")
    codebook_path = root / "codebook.npy"
    require(base._sha256_file(codebook_path) == codebook["sha256"], "SID codebook hash mismatch")
    centroids = np.load(codebook_path, mmap_mode="r", allow_pickle=False)
    require(centroids.shape == (6, 256, 1024) and centroids.dtype == np.float32 and np.isfinite(centroids).all(),
            "SID codebook geometry/finite values mismatch")
    artifacts = base._object(manifest["artifacts"], "representation artifacts")
    require(set(artifacts) == ARTIFACTS, "representation artifact inventory mismatch")
    paths = {}
    for name, binding in artifacts.items():
        base._require_exact_keys(binding, {"path", "sha256", "count"}, "representation artifact")
        path = base._safe_dataset_path(root, binding["path"], name)
        require(path not in paths.values() and _hex(binding["sha256"])
                and base._sha256_file(path) == binding["sha256"] and type(binding["count"]) is int
                and binding["count"] == len(expected), "representation artifact hash/count mismatch")
        paths[name] = path
    require(manifest["counts"] == {"posts": len(expected), "trainPosts": len(train_posts)}, "representation coverage count mismatch")
    post_ids = np.load(paths["postIds"], mmap_mode="r", allow_pickle=False)
    embeddings = np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False)
    require(post_ids.dtype == np.int64 and post_ids.shape == (len(expected),)
            and post_ids.tolist() == sorted(map(int, expected)) and embeddings.dtype == np.float32
            and embeddings.shape == (len(expected), 1024) and np.isfinite(embeddings).all(),
            "real MM table identity/geometry mismatch")
    require(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-4, atol=1e-5), "real MM embeddings are not L2-normalized")
    records = {}
    rows = [json.loads(line) for line in paths["records"].read_text().splitlines()]
    require(len(rows) == len(expected), "representation records coverage mismatch")
    for index, record in enumerate(rows):
        base._require_exact_keys(record, {"modelPostId", "generationId", "modelAuthorId", "createdAtMs",
            "createTimeSeconds", "createTimeNanoseconds", "contentSha256", "sourceUpdateTimeSeconds",
            "sourceUpdateTimeNanoseconds", "embeddingSha256", "semanticIds"}, "representation record")
        post = record["modelPostId"]
        require(post == str(post_ids[index]) and post in expected
                and all(record[key] == expected[post][key] for key in ("modelPostId", "generationId", "modelAuthorId", "createdAtMs", "createTimeSeconds", "createTimeNanoseconds"))
                and _hex(record["contentSha256"])
                and type(record["sourceUpdateTimeSeconds"]) is int and type(record["sourceUpdateTimeNanoseconds"]) is int
                and 0 <= record["sourceUpdateTimeNanoseconds"] < 1_000_000_000
                and (record["sourceUpdateTimeSeconds"], record["sourceUpdateTimeNanoseconds"]) >=
                    (record["createTimeSeconds"], record["createTimeNanoseconds"]), "representation full generation/content mismatch")
        if source_contract == "grassy_phoenix_dataset_v6":
            source_record = expected[post]
            representative = next(ref for ref in source_record["contentReferences"]
                                  if ref["contentSnapshotId"] == source_record["contentSnapshotId"])
            require(record["contentSha256"] == source_record["contentSha256"] == representative["contentSha256"]
                    and record["sourceUpdateTimeSeconds"] == representative["sourceUpdateTimeSeconds"]
                    and record["sourceUpdateTimeNanoseconds"] == representative["sourceUpdateTimeNanoseconds"],
                    "representation does not match proof-bound representative content")
        require(record["embeddingSha256"] == hashlib.sha256(embeddings[index].tobytes()).hexdigest(),
                "representation real MM vector hash mismatch")
        require(isinstance(record["semanticIds"], list) and len(record["semanticIds"]) == 6
                and all(type(code) is int and 0 <= code < 256 for code in record["semanticIds"]),
                "representation raw SID6 invalid; zero missing sentinels cannot substitute for provenance")
        records[post] = record
    return GrassyRepresentations(records=records, manifest_sha256=expected_sha256)
