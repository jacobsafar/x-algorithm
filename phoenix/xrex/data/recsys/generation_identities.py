# SPDX-License-Identifier: Apache-2.0
"""Source-bound full-generation identities; independent of model representations."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from xrex.data.recsys import observation_sidecars as base

TWITTER_EPOCH_MS = 1288834974657


def load_generation_identities(root: Path, manifest: dict, *, verify_base: bool = True) -> dict[str, dict]:
    require = base._require
    binding = base._object(manifest.get("post_generation_identities"), "post generation identities")
    base._require_exact_keys(binding, {"path", "sha256", "encoding", "rows"}, "post generation binding")
    path = base._safe_dataset_path(root, binding["path"], "post generation identities")
    require(binding["encoding"] == "jsonl-model-post-full-generation-v1"
            and base._sha256_file(path) == binding["sha256"], "post generation identity hash mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(records) == binding["rows"] and bool(records), "post generation identity count mismatch")
    prospective = manifest.get("source_contract") == "grassy_phoenix_dataset_v6"
    result = {}
    previous = 0
    generations = set()
    for item in records:
        expected_keys = {"modelPostId", "generationId", "modelAuthorId", "createdAtMs", "createTimeSeconds", "createTimeNanoseconds"}
        if prospective:
            expected_keys |= {"renderPolicy", "encoderInputSha256", "contentSnapshotId", "contentSha256", "contentReferences"}
        base._require_exact_keys(item, expected_keys, "post generation identity")
        if prospective:
            require(item["renderPolicy"] == manifest.get("capture_policy", {}).get("renderPolicy")
                    and all(isinstance(item[key], str) and re.fullmatch("[a-f0-9]{64}", item[key])
                            for key in ("encoderInputSha256", "contentSnapshotId", "contentSha256")),
                    "v6 generation content binding malformed")
            refs = item["contentReferences"]
            require(isinstance(refs, list) and bool(refs), "v6 generation content inventory absent")
            reference_ids = []
            for ref in refs:
                base._require_exact_keys(ref, {"contentSnapshotId", "contentSha256", "sourceUpdateTimeSeconds", "sourceUpdateTimeNanoseconds"}, "v6 content reference")
                require(all(isinstance(ref[key], str) and re.fullmatch("[a-f0-9]{64}", ref[key])
                            for key in ("contentSnapshotId", "contentSha256"))
                        and type(ref["sourceUpdateTimeSeconds"]) is int and type(ref["sourceUpdateTimeNanoseconds"]) is int
                        and 0 <= ref["sourceUpdateTimeNanoseconds"] < 1_000_000_000, "v6 content reference invalid")
                reference_ids.append(ref["contentSnapshotId"])
            require(reference_ids == sorted(set(reference_ids)) and any(ref["contentSnapshotId"] == item["contentSnapshotId"]
                    and ref["contentSha256"] == item["contentSha256"] for ref in refs), "v6 representative/ref inventory mismatch")
        require(all(isinstance(item[k], str) and re.fullmatch("[1-9][0-9]*", item[k])
                    and 0 < int(item[k]) < 2 ** 63 for k in ("modelPostId", "modelAuthorId")), "model ID is not int63")
        number = int(item["modelPostId"])
        require(number > previous and isinstance(item["generationId"], str)
                and re.fullmatch("[a-f0-9]{64}", item["generationId"])
                and item["generationId"] not in generations, "full generation identity collision/order mismatch")
        require(all(type(item[k]) is int for k in ("createdAtMs", "createTimeSeconds", "createTimeNanoseconds"))
                and 0 <= item["createTimeNanoseconds"] < 1_000_000_000
                and item["createdAtMs"] == item["createTimeSeconds"] * 1000 + item["createTimeNanoseconds"] // 1_000_000
                and item["createdAtMs"] == (number >> 22) + TWITTER_EPOCH_MS,
                "model Snowflake age/full generation mismatch")
        result[item["modelPostId"]] = item
        previous = number
        generations.add(item["generationId"])
    if verify_base:
        parquet = base._safe_dataset_path(root, manifest["parquet"], "generation base")
        require(base._sha256_file(parquet) == manifest["parquet_sha256"], "generation base hash mismatch")
        seen = set()
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["tweetIdSeq", "authorIdSeq", "paddingMask"]):
            values = batch.to_pydict()
            for posts, authors, active in zip(values["tweetIdSeq"], values["authorIdSeq"], values["paddingMask"], strict=True):
                require(len(posts) == len(authors) == len(active) == 1086, "generation sequence shape mismatch")
                for post, author, valid in zip(posts, authors, active, strict=True):
                    if valid:
                        require(str(post) in result and result[str(post)]["modelAuthorId"] == str(author),
                                "nonpadding post lacks exact generation/author identity")
                        seen.add(str(post))
                    else:
                        require(post == 0 and author == 0, "padding has unadmitted identity")
        require(seen == set(result), "generation inventory differs from actual base sequences")
    return result
