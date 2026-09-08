"""CPU-only admission and wire validation shared by preflight and HTTP serving."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

CONTRACT = "grassy_phoenix_shadow_v1"
INPUT_CONTRACT = "grassy_phoenix_online_features_v1"
UPSTREAM = "902a06fd616ed815f660e5546d16d492fa1ca825"
MAX_BODY = 1024 * 1024
MAX_RESPONSE = 256 * 1024
MAX_AGE_MS = 172800000
EPOCH_MS = 1288834974657
IDENTITY_KEYS = {"generationKey", "modelPostId", "modelAuthorId", "createTimeSeconds",
                 "createTimeNanoseconds", "createdAtMs", "embeddingSha256", "semanticIds"}


class ContractError(ValueError):
    """A fixed public reason; never include input, paths or identifiers."""


def require(value: object, reason: str = "contract_invalid") -> None:
    if not value:
        raise ContractError(reason)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def integer(value: object, lo: int, hi: int) -> bool:
    return type(value) is int and lo <= value <= hi


def decimal_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,18}", value) is not None and int(value) < 2**63


def load_json_bytes(data: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ContractError("contract_invalid")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContractError("contract_invalid") from exc


def load_manifest(path: Path, expected_sha: str) -> dict:
    data = path.read_bytes()
    require(0 < len(data) <= 65536 and is_digest(expected_sha) and sha256(data) == expected_sha)
    m = load_json_bytes(data)
    require(isinstance(m, dict))
    expected = {"schemaVersion": 1, "contract": CONTRACT, "upstreamCommit": UPSTREAM,
                "featurePolicy": "grassy_unavailable_features_disabled_v1",
                "inputContract": INPUT_CONTRACT, "proofSchemaVersion": 2,
                "identityContract": "grassy_hmac_identity_user_split_v1",
                "trainingKind": "production_exact", "observationMaskPolicy": "required_per_head_v1",
                "historyPositions": 1022, "candidatePositions": 64,
                "discreteWidth": 64, "continuousWidth": 8,
                "utilityPolicy": "trained_heads_weighted_sum_v1"}
    require(all(m.get(k) == v and type(m.get(k)) is type(v) for k, v in expected.items()))
    require(isinstance(m.get("grassyCommit"), str) and re.fullmatch(r"[a-f0-9]{40}", m["grassyCommit"]))
    validate_manifest_age_policy(m)
    require(isinstance(m.get("identityKeyId"), str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", m["identityKeyId"]))
    require(isinstance(m.get("projectId"), str) and re.fullmatch(r"[a-z][a-z0-9-]{4,62}", m["projectId"]))
    for key in ("identityKeySha256", "actionContractSha256", "embeddingSnapshotSha256",
                "sidSnapshotSha256", "featureAtlasSha256", "checkpointManifestSha256", "trainingDatasetSha256"):
        require(is_digest(m.get(key)))
    require(integer(m.get("trainingSteps"), 1, 2**53-1))
    nonzero = False
    for name, width in (("Discrete", 60), ("Continuous", 5)):
        heads = m.get(f"trained{name}Heads")
        require(isinstance(heads, list) and len(heads) <= width and
                all(integer(x, 1, width-1) for x in heads) and len(set(heads)) == len(heads))
        weights = m.get(f"{name.lower()}Weights")
        require(isinstance(weights, dict))
        for key, value in weights.items():
            require(key.isdecimal() and str(int(key)) == key and int(key) in heads and
                    type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1000)
            nonzero |= value != 0
    require(nonzero)
    return m


V6_POLICY_FIELDS = ("sourceContract", "featureSchemaVersion", "proofSchemaVersion", "agePolicy",
    "maxCandidateAgeMs", "ageEncoding", "postAgeGranularityMins", "postAgeMaxMins", "contentContract", "renderPolicy")


def validate_manifest_age_policy(manifest: dict) -> None:
    if manifest.get("sourceContract") == "grassy_phoenix_dataset_v5":
        expected = {"modelConfigName": "grassy_home_direct_packed_nano", "featureSchemaVersion": 3,
                    "agePolicy": "legacy_48h_v1", "maxCandidateAgeMs": MAX_AGE_MS}
        require("contentContract" not in manifest and "renderPolicy" not in manifest)
        if any(k in manifest for k in ("ageEncoding", "postAgeGranularityMins", "postAgeMaxMins")):
            expected.update(ageEncoding="phoenix_post_age_1h_80bins_v1", postAgeGranularityMins=60, postAgeMaxMins=4800)
    else:
        expected = {"sourceContract": "grassy_phoenix_dataset_v6", "modelConfigName": "grassy_home_direct_packed_nano_90d",
            "featureSchemaVersion": 4, "agePolicy": "grassy_candidate_age_90d_v1", "maxCandidateAgeMs": 7776000000,
            "ageEncoding": "grassy_post_age_30h_80bins_v1", "postAgeGranularityMins": 1800,
            "postAgeMaxMins": 144000, "contentContract": "grassy_served_content_v1"}
        require(manifest.get("renderPolicy") in ("grassy_text_only_v1", "grassy_caption_only_v1"))
    require(all(manifest.get(k) == v and type(manifest.get(k)) is type(v) for k, v in expected.items()))


def validate_runtime_model_age_policy(model: object, manifest: dict) -> None:
    """Restore only the exact architecture/preprocessing that trained this inventory."""
    expected_encoding, granularity, maximum = (("grassy_post_age_30h_80bins_v1", 1800, 144000)
        if manifest["sourceContract"] == "grassy_phoenix_dataset_v6" else ("phoenix_post_age_1h_80bins_v1", 60, 4800))
    require(getattr(model, "grassy_candidate_age_policy", None) == manifest["agePolicy"] and
            getattr(model, "grassy_post_age_encoding", None) == expected_encoding and
            getattr(model, "post_age_granularity_mins", None) == granularity and
            getattr(model, "post_age_max_mins", None) == maximum and
            getattr(getattr(model, "feature_prep", None), "post_age_granularity_mins", None) == granularity and
            getattr(getattr(model, "feature_prep", None), "post_age_max_mins", None) == maximum,
            "checkpoint_age_policy_mismatch")


def validate_restored_training_state(step: object, elapsed_samples: object, manifest: dict) -> None:
    # The admitted Grassy recipe is one device with32 physical examples per update.
    # Inference initialization may allocate random arrays before restore; those never qualify.
    require(integer(step, 1, 2**53-1) and step == manifest["trainingSteps"] and
            integer(elapsed_samples, 1, 2**53-1) and elapsed_samples == step * 32,
            "checkpoint_training_progress_mismatch")


def safe_file(root: Path, name: object) -> Path:
    require(isinstance(name, str) and len(name) <= 512 and name not in ("", "."))
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts)
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()))
    return path


def hash_file(path: Path, maximum: int = 256 * 1024**3) -> str:
    require(path.stat().st_size <= maximum)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(root: Path, manifest: dict, inventory_path: Path) -> tuple[Path, dict]:
    """Read-only startup verification; inventory binds every local checkpoint file."""
    data = inventory_path.read_bytes()
    require(len(data) <= 8 * 1024**2 and sha256(data) == manifest["checkpointManifestSha256"])
    inventory = load_json_bytes(data)
    require(isinstance(inventory, dict) and inventory.get("schemaVersion") == 1 and
            inventory.get("trainingKind") == "production_exact" and
            inventory.get("trainingSteps") == manifest["trainingSteps"] and
            inventory.get("trainingDatasetSha256") == manifest["trainingDatasetSha256"] and
            inventory.get("modelConfigName") == manifest["modelConfigName"] and
            inventory.get("grassyCommit") == manifest["grassyCommit"] and
            inventory.get("featurePolicy") == manifest["featurePolicy"])
    if manifest["sourceContract"] == "grassy_phoenix_dataset_v6":
        require(all(inventory.get(k) == manifest[k] and type(inventory.get(k)) is type(manifest[k])
                    for k in V6_POLICY_FIELDS), "checkpoint_policy_mismatch")
    else:
        legacy = {"sourceContract": "grassy_phoenix_dataset_v5", "featureSchemaVersion": 3,
            "proofSchemaVersion": 2, "agePolicy": "legacy_48h_v1", "maxCandidateAgeMs": MAX_AGE_MS,
            "ageEncoding": "phoenix_post_age_1h_80bins_v1", "postAgeGranularityMins": 60, "postAgeMaxMins": 4800}
        require("contentContract" not in inventory and "renderPolicy" not in inventory and
                all(k not in inventory or (inventory[k] == v and type(inventory[k]) is type(v))
                    for k, v in legacy.items()), "checkpoint_policy_mismatch")
    files = inventory.get("files")
    require(isinstance(files, list) and 2 <= len(files) <= 50000)
    names = set()
    for row in files:
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"})
        path = safe_file(root, row["path"])
        require(row["path"] not in names and integer(row["bytes"], 0, 256 * 1024**3) and
                path.stat().st_size == row["bytes"] and is_digest(row["sha256"]) and
                hash_file(path) == row["sha256"])
        names.add(row["path"])
    # Exact directory is fixed, mounted read-only; no latest-prefix discovery or hot reload.
    require("completed" in names and any(name.startswith("orbax-ckpt/") for name in names))
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == names and not any(p.is_symlink() for p in root.rglob("*")))
    return root.resolve(), inventory


def validate_identity(item: dict) -> None:
    require(is_digest(item.get("generationKey")) and decimal_id(item.get("modelPostId")) and
            decimal_id(item.get("modelAuthorId")) and
            integer(item.get("createTimeSeconds"), 0, 2**53-1) and
            integer(item.get("createTimeNanoseconds"), 0, 999999999) and
            integer(item.get("createdAtMs"), 1, 2**53-1) and
            item["createdAtMs"] == item["createTimeSeconds"] * 1000 + item["createTimeNanoseconds"] // 1000000 and
            (int(item["modelPostId"]) >> 22) + EPOCH_MS == item["createdAtMs"] and
            is_digest(item.get("embeddingSha256")))
    sid = item.get("semanticIds")
    require(isinstance(sid, list) and len(sid) == 6 and all(integer(v, 0, 255) for v in sid))


def load_feature_atlas(atlas_path: Path, mm_path: Path, sid_path: Path, manifest: dict) -> dict[str, dict]:
    """Verify actual engine snapshot rows, not merely self-declared per-post digests."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for path, field in ((atlas_path, "featureAtlasSha256"), (mm_path, "embeddingSnapshotSha256"),
                        (sid_path, "sidSnapshotSha256")):
        require(hash_file(path) == manifest[field])
    require(atlas_path.stat().st_size <= 256 * 1024**2 and sid_path.stat().st_size <= 256 * 1024**2)
    atlas = load_json_bytes(atlas_path.read_bytes())
    sids = load_json_bytes(sid_path.read_bytes())
    require(isinstance(atlas, dict) and atlas.get("schemaVersion") == 1 and
            atlas.get("inputContract") == INPUT_CONTRACT and
            atlas.get("identityKeyId") == manifest["identityKeyId"] and
            atlas.get("embeddingSnapshotSha256") == manifest["embeddingSnapshotSha256"] and
            atlas.get("sidSnapshotSha256") == manifest["sidSnapshotSha256"] and
            isinstance(atlas.get("entries"), list) and 0 < len(atlas["entries"]) <= 1000000)
    require(isinstance(sids, dict) and sids.get("schemaVersion") == 1 and isinstance(sids.get("entries"), list))
    by_id, sid_by_id, keys = {}, {}, set()
    for row in sids["entries"]:
        require(isinstance(row, dict) and set(row) == {"modelPostId", "semanticIds"} and
                decimal_id(row["modelPostId"]) and row["modelPostId"] not in sid_by_id)
        sid_by_id[row["modelPostId"]] = row["semanticIds"]
    for item in atlas["entries"]:
        require(isinstance(item, dict) and set(item) == IDENTITY_KEYS)
        validate_identity(item)
        require(item["modelPostId"] not in by_id and item["generationKey"] not in keys and
                sid_by_id.get(item["modelPostId"]) == item["semanticIds"])
        by_id[item["modelPostId"]] = item
        keys.add(item["generationKey"])
    require(set(by_id) == set(sid_by_id))
    parquet = pq.ParquetFile(mm_path)
    schema = parquet.schema_arrow
    require(len(schema) == 3 and schema.field(0).type == pa.uint64() and
            schema.field(2).type == pa.large_list(pa.float32()))
    found = set()
    for batch in parquet.iter_batches(batch_size=1024):
        for post_id, vector in zip(batch.column(0).to_pylist(), batch.column(2).to_pylist()):
            key = str(post_id)
            require(key in by_id and key not in found and isinstance(vector, list) and len(vector) == 1024 and
                    all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 65504 for v in vector) and
                    any(v != 0 for v in vector) and
                    sha256(struct.pack("<1024f", *vector)) == by_id[key]["embeddingSha256"])
            found.add(key)
    require(found == set(by_id))
    return by_id


def validate_request(data: bytes, digest: str, manifest: dict, manifest_sha: str,
                     atlas: dict[str, dict], now_ms: int) -> dict:
    require(0 < len(data) <= MAX_BODY and is_digest(digest) and sha256(data) == digest)
    request = load_json_bytes(data)
    require(isinstance(request, dict) and set(request) ==
            {"schemaVersion", "contract", "nonce", "manifestSha256", "inputs"} and
            request["schemaVersion"] == 1 and request["contract"] == CONTRACT and
            request["manifestSha256"] == manifest_sha and isinstance(request["nonce"], str) and
            re.fullmatch(r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}", request["nonce"]))
    value = request["inputs"]
    require(isinstance(value, dict) and set(value) == {"inputContract", "manifestSha256", "identityKeyId",
            "modelUserId", "asOfMs", "capturedAtMs", "history", "candidates"} and
            value["inputContract"] == INPUT_CONTRACT and value["manifestSha256"] == manifest_sha and
            value["identityKeyId"] == manifest["identityKeyId"] and decimal_id(value["modelUserId"]) and
            integer(value["asOfMs"], now_ms - 10000, now_ms + 10000) and
            integer(value["capturedAtMs"], now_ms - 300000, now_ms + 5000) and
            isinstance(value["candidates"], list) and 1 <= len(value["candidates"]) <= 64 and
            isinstance(value["history"], list) and len(value["history"]) <= 1022)
    ids, previous = set(), 0
    for is_history, entries in ((False, value["candidates"]), (True, value["history"])):
        for item in entries:
            require(isinstance(item, dict) and set(item) == IDENTITY_KEYS |
                    ({"impressedAtMs", "surface", "actionOrdinals", "dwellMs"} if is_history else set()))
            validate_identity(item)
            identity = {k: item[k] for k in IDENTITY_KEYS}
            require(item["modelPostId"] not in ids and atlas.get(item["modelPostId"]) == identity)
            ids.add(item["modelPostId"])
            if not is_history:
                require(0 <= value["asOfMs"] - item["createdAtMs"] <= manifest["maxCandidateAgeMs"])
            else:
                require(integer(item["impressedAtMs"], max(previous + 1, item["createdAtMs"]), value["asOfMs"]) and
                        item["surface"] in ("for_you", "profile") and isinstance(item["actionOrdinals"], list) and
                        0 < len(item["actionOrdinals"]) <= 60 and
                        all(type(v) is int and v in manifest["trainedDiscreteHeads"] for v in item["actionOrdinals"]) and
                        len(set(item["actionOrdinals"])) == len(item["actionOrdinals"]) and
                        (item["dwellMs"] is None or (1 in manifest["trainedContinuousHeads"] and
                         integer(item["dwellMs"], 0, 86400000))))
                previous = item["impressedAtMs"]
    return request
