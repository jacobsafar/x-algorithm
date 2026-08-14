# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""Fail-closed loader for Grassy's immutable label-observability sidecars.

The Grassy exporter deliberately leaves Phoenix's 36-column Parquet schema
unchanged.  This module validates the two-level dataset/sidecar manifest
binding, then exposes exact row windows for the existing in-memory Arrow
bridge.  It has no JAX dependency so the storage contract can be tested in
isolation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy import typing as npt

from xrex.data.recsys.observability import append_observation_mask_columns


GRASSY_UPSTREAM_COMMIT = "a389166f6cf5da70a286b568c87695d4dcdce3a1"
GRASSY_DATASET_MANIFEST = "grassy_phoenix_manifest.json"
GRASSY_DATASET_MANIFEST_VERSION = 4
GRASSY_SIDECAR_MANIFEST_VERSION = 1
GRASSY_EXPORTER_NAME = "grassy-feed-event-plane-to-phoenix"
GRASSY_EXPORTER_VERSION = 2
GRASSY_INPUT_SCHEMA_VERSION = 1
GRASSY_SEQUENCE_LENGTH = 1086
GRASSY_DISCRETE_WIDTH = 64
GRASSY_CONTINUOUS_WIDTH = 8
GRASSY_BASE_COLUMNS = 36
GRASSY_REQUIRED_MODEL_GATES = {
    "require_label_observation_masks": True,
    "enable_engagement_counts": False,
    "compute_post_unexplored_label": False,
}


class ObservationSidecarError(ValueError):
    """A Grassy artifact is not safe to use as training evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationSidecarError(message)


def _object(value: object, where: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{where}: expected object")
    return value


def _array(value: object, where: str) -> list[object]:
    _require(isinstance(value, list), f"{where}: expected array")
    return value


def _string(value: object, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where}: expected non-empty string")
    return value


def _integer(value: object, where: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected integer")
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{where}: must be >= {minimum}")
    return result


def _read_json(path: Path, where: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ObservationSidecarError(f"{where}: cannot read {path}: {exc}") from exc
    try:
        parsed: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationSidecarError(f"{where}: invalid JSON in {path}: {exc}") from exc
    return _object(parsed, where), raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ObservationSidecarError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_dataset_path(root: Path, value: object, where: str) -> Path:
    relative = Path(_string(value, where))
    _require(not relative.is_absolute(), f"{where}: absolute paths are forbidden")
    _require(".." not in relative.parts, f"{where}: parent traversal is forbidden")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ObservationSidecarError(f"{where}: path escapes dataset root") from exc
    return candidate


def _require_exact_keys(value: dict[str, object], expected: set[str], where: str) -> None:
    actual = set(value)
    _require(actual == expected, f"{where}: fields {sorted(actual)} != expected {sorted(expected)}")


@dataclass(frozen=True)
class ObservationMaskSidecar:
    """Validated, immutable full-file masks with row-window attachment."""

    base_path: Path
    rows: int
    outcome_cutoff_ms: int
    action_mask: npt.NDArray[np.bool_]
    continuous_action_mask: npt.NDArray[np.bool_]

    def attach(self, record_batch: pa.RecordBatch, row_start: int) -> pa.RecordBatch:
        """Append the exact sidecar window corresponding to a Parquet batch."""

        _require(row_start >= 0, f"negative sidecar row offset {row_start}")
        row_end = row_start + record_batch.num_rows
        _require(
            row_end <= self.rows,
            f"sidecar row window [{row_start}, {row_end}) exceeds {self.rows} rows",
        )
        action_window = self.action_mask[row_start:row_end]
        continuous_window = self.continuous_action_mask[row_start:row_end]
        _require(
            action_window.shape
            == (record_batch.num_rows, GRASSY_SEQUENCE_LENGTH, GRASSY_DISCRETE_WIDTH),
            "action-observation row window is truncated",
        )
        _require(
            continuous_window.shape
            == (
                record_batch.num_rows,
                GRASSY_SEQUENCE_LENGTH,
                GRASSY_CONTINUOUS_WIDTH,
            ),
            "continuous-observation row window is truncated",
        )
        return append_observation_mask_columns(
            record_batch,
            action_mask=action_window,
            continuous_action_mask=continuous_window,
        )


def _validate_dataset_manifest(
    *,
    root: Path,
    base_path: Path,
    base_sha256: str,
    sidecar_manifest_path: Path,
    sidecar_manifest_sha256: str,
    sidecar: dict[str, object],
    rows: int,
) -> None:
    manifest_path = (root / GRASSY_DATASET_MANIFEST).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ObservationSidecarError("dataset manifest escapes dataset root") from exc
    manifest, _ = _read_json(manifest_path, "dataset manifest")
    _require(
        manifest.get("manifest_version") == GRASSY_DATASET_MANIFEST_VERSION,
        "dataset manifest version mismatch",
    )
    _require(
        manifest.get("upstream_commit") == GRASSY_UPSTREAM_COMMIT,
        "dataset upstream commit mismatch",
    )
    exporter = _object(manifest.get("exporter"), "dataset.exporter")
    _require_exact_keys(
        exporter,
        {
            "name",
            "version",
            "input_schema_version",
            "source_sha256",
            "base_adapter_sha256",
        },
        "dataset.exporter",
    )
    _require(
        exporter.get("name") == GRASSY_EXPORTER_NAME
        and exporter.get("version") == GRASSY_EXPORTER_VERSION
        and exporter.get("input_schema_version") == GRASSY_INPUT_SCHEMA_VERSION,
        "dataset exporter contract mismatch",
    )
    for field in ("source_sha256", "base_adapter_sha256"):
        digest = _string(exporter.get(field), f"dataset.exporter.{field}")
        _require(
            len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            f"dataset.exporter.{field}: invalid SHA-256",
        )
    _require(manifest.get("training_eligible") is True, "dataset is not training eligible")
    _require(
        manifest.get("required_model_gates") == GRASSY_REQUIRED_MODEL_GATES,
        "dataset does not pin the exact Grassy model gates",
    )
    _require(manifest.get("rows") == rows, "dataset manifest row count mismatch")
    _require(
        _safe_dataset_path(root, manifest.get("parquet"), "dataset.parquet") == base_path,
        "dataset manifest points to a different Parquet file",
    )
    _require(manifest.get("parquet_sha256") == base_sha256, "dataset Parquet hash mismatch")
    _require(
        _safe_dataset_path(
            root,
            manifest.get("sidecar_manifest"),
            "dataset.sidecar_manifest",
        )
        == sidecar_manifest_path,
        "dataset manifest points to a different sidecar manifest",
    )
    _require(
        manifest.get("sidecar_manifest_sha256") == sidecar_manifest_sha256,
        "sidecar manifest hash does not match dataset manifest",
    )
    identity = _object(sidecar.get("rowIdentity"), "sidecar.rowIdentity")
    _require(
        manifest.get("row_identity_sha256") == identity.get("sha256"),
        "row-identity hash does not match dataset manifest",
    )
    _require(
        manifest.get("outcome_cutoff_ms") == sidecar.get("outcomeCutoffMs"),
        "outcome cutoff does not match dataset manifest",
    )
    _require(
        manifest.get("action_contract_version") == sidecar.get("labelContractVersion"),
        "label contract version does not match dataset manifest",
    )
    shape = _object(manifest.get("shape"), "dataset.shape")
    expected_shape = {
        "history_positions": 1022,
        "candidate_positions": 64,
        "sequence_positions": GRASSY_SEQUENCE_LENGTH,
        "discrete_action_width": GRASSY_DISCRETE_WIDTH,
        "continuous_action_width": GRASSY_CONTINUOUS_WIDTH,
        "base_columns": GRASSY_BASE_COLUMNS,
    }
    _require(shape == expected_shape, "dataset shape contract mismatch")


def _load_mask(
    *,
    root: Path,
    sidecar: dict[str, object],
    section_name: str,
    expected_shape: tuple[int, int, int],
    expected_forced_false_heads: list[int],
) -> npt.NDArray[np.bool_]:
    section = _object(sidecar.get(section_name), f"sidecar.{section_name}")
    _require_exact_keys(
        section,
        {"path", "sha256", "dtype", "shape", "forcedFalseHeads"},
        f"sidecar.{section_name}",
    )
    _require(section.get("dtype") == "bool", f"{section_name}: declared dtype is not bool")
    _require(
        section.get("shape") == list(expected_shape),
        f"{section_name}: declared shape mismatch",
    )
    _require(
        section.get("forcedFalseHeads") == expected_forced_false_heads,
        f"{section_name}: forced-false head metadata mismatch",
    )
    path = _safe_dataset_path(root, section.get("path"), f"sidecar.{section_name}.path")
    _require(path.is_file(), f"{section_name}: mask file is missing")
    _require(_sha256_file(path) == section.get("sha256"), f"{section_name}: hash mismatch")
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ObservationSidecarError(f"{section_name}: invalid NumPy file: {exc}") from exc
    _require(value.dtype == np.dtype(np.bool_), f"{section_name}: physical dtype is not bool")
    _require(value.shape == expected_shape, f"{section_name}: physical shape mismatch")
    _require(
        not any(bool(value[..., head].any()) for head in expected_forced_false_heads),
        f"{section_name}: reserved head is observable",
    )
    return value


def _validate_row_identities(
    *,
    root: Path,
    sidecar: dict[str, object],
    rows: int,
) -> None:
    identity = _object(sidecar.get("rowIdentity"), "sidecar.rowIdentity")
    _require_exact_keys(identity, {"path", "sha256", "encoding"}, "sidecar.rowIdentity")
    _require(
        identity.get("encoding")
        == "jsonl-opaque-keyed-blake2b-128-plus-rollout-provenance-v1",
        "row-identity encoding mismatch",
    )
    identity_path = _safe_dataset_path(root, identity.get("path"), "sidecar.rowIdentity.path")
    _require(identity_path.is_file(), "row-identity file is missing")
    _require(_sha256_file(identity_path) == identity.get("sha256"), "row-identity hash mismatch")

    rollout_values = _array(sidecar.get("rolloutsUsed"), "sidecar.rolloutsUsed")
    _require(bool(rollout_values), "sidecar has no rollout provenance")
    known_rollouts: dict[tuple[str, str, str], set[str]] = {}
    for index, raw_rollout in enumerate(rollout_values):
        rollout = _object(raw_rollout, f"sidecar.rolloutsUsed[{index}]")
        _require_exact_keys(
            rollout,
            {
                "releaseId",
                "platform",
                "appVersion",
                "effectiveFromMs",
                "effectiveToMs",
                "surfaces",
            },
            f"sidecar.rolloutsUsed[{index}]",
        )
        key = (
            _string(rollout.get("releaseId"), f"rollout[{index}].releaseId"),
            _string(rollout.get("platform"), f"rollout[{index}].platform"),
            _string(rollout.get("appVersion"), f"rollout[{index}].appVersion"),
        )
        start = _integer(
            rollout.get("effectiveFromMs"),
            f"rollout[{index}].effectiveFromMs",
            minimum=0,
        )
        end = _integer(
            rollout.get("effectiveToMs"),
            f"rollout[{index}].effectiveToMs",
            minimum=0,
        )
        _require(start < end, f"rollout[{index}]: invalid effective interval")
        surfaces = {
            _string(value, f"rollout[{index}].surfaces")
            for value in _array(rollout.get("surfaces"), f"rollout[{index}].surfaces")
        }
        _require(bool(surfaces), f"rollout[{index}]: surfaces must not be empty")
        _require(key not in known_rollouts, f"rollout[{index}]: duplicate provenance")
        known_rollouts[key] = surfaces

    seen_ids: set[str] = set()
    expected_fields = {"exampleId", "rolloutReleaseId", "platform", "appVersion", "surface"}
    identity_count = 0
    try:
        with identity_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                identity_count += 1
                try:
                    raw_identity: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ObservationSidecarError(
                        f"rowIdentity[{index}]: invalid JSON"
                    ) from exc
                item = _object(raw_identity, f"rowIdentity[{index}]")
                _require_exact_keys(item, expected_fields, f"rowIdentity[{index}]")
                example_id = _string(
                    item.get("exampleId"), f"rowIdentity[{index}].exampleId"
                )
                _require(
                    len(example_id) == 32
                    and all(char in "0123456789abcdef" for char in example_id),
                    f"rowIdentity[{index}]: invalid opaque example ID",
                )
                _require(
                    example_id not in seen_ids,
                    f"rowIdentity[{index}]: duplicate example ID",
                )
                seen_ids.add(example_id)
                rollout_key = (
                    _string(
                        item.get("rolloutReleaseId"),
                        f"rowIdentity[{index}].rolloutReleaseId",
                    ),
                    _string(item.get("platform"), f"rowIdentity[{index}].platform"),
                    _string(item.get("appVersion"), f"rowIdentity[{index}].appVersion"),
                )
                surface = _string(item.get("surface"), f"rowIdentity[{index}].surface")
                _require(
                    rollout_key in known_rollouts and surface in known_rollouts[rollout_key],
                    f"rowIdentity[{index}]: rollout provenance is not declared",
                )
    except (OSError, UnicodeDecodeError) as exc:
        raise ObservationSidecarError(f"cannot read row identities: {exc}") from exc
    _require(identity_count == rows, "row-identity count mismatch")


def load_observation_mask_sidecar(
    *,
    base_path: str | Path,
    dataset_root: str | Path,
    parquet_num_rows: int,
    parquet_num_columns: int,
) -> ObservationMaskSidecar:
    """Load and fully verify the sidecar adjacent to ``base_path``.

    Every manifest path is interpreted relative to ``dataset_root`` and is
    constrained to remain below that root.  Missing artifacts and all binding,
    shape, dtype, identity, or provenance mismatches are fatal.
    """

    root = Path(dataset_root).resolve()
    base = Path(base_path).resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise ObservationSidecarError("base Parquet is outside dataset root") from exc
    _require(base.is_file(), "base Parquet is missing")
    _require(parquet_num_rows >= 1, "base Parquet has no rows")
    _require(
        parquet_num_columns == GRASSY_BASE_COLUMNS,
        f"base Parquet has {parquet_num_columns} columns, expected {GRASSY_BASE_COLUMNS}",
    )

    sidecar_manifest_path = base.with_suffix(".observability.json").resolve()
    try:
        sidecar_manifest_path.relative_to(root)
    except ValueError as exc:
        raise ObservationSidecarError("sidecar manifest escapes dataset root") from exc
    _require(
        sidecar_manifest_path.is_file(),
        f"sidecar manifest is missing: {sidecar_manifest_path}",
    )
    sidecar, sidecar_raw = _read_json(sidecar_manifest_path, "sidecar manifest")
    _require_exact_keys(
        sidecar,
        {
            "manifestVersion",
            "exporterVersion",
            "upstreamCommit",
            "labelContractVersion",
            "outcomeCutoffMs",
            "rows",
            "base",
            "rowIdentity",
            "actionObservation",
            "continuousActionObservation",
            "rolloutsUsed",
        },
        "sidecar manifest",
    )
    _require(
        sidecar.get("manifestVersion") == GRASSY_SIDECAR_MANIFEST_VERSION,
        "sidecar manifest version mismatch",
    )
    _require(
        sidecar.get("exporterVersion") == GRASSY_EXPORTER_VERSION,
        "sidecar exporter version mismatch",
    )
    _require(
        sidecar.get("upstreamCommit") == GRASSY_UPSTREAM_COMMIT,
        "sidecar upstream commit mismatch",
    )
    _integer(sidecar.get("labelContractVersion"), "sidecar.labelContractVersion", minimum=1)
    cutoff = _integer(sidecar.get("outcomeCutoffMs"), "sidecar.outcomeCutoffMs", minimum=1)
    rows = _integer(sidecar.get("rows"), "sidecar.rows", minimum=1)
    _require(rows == parquet_num_rows, "sidecar/base row count mismatch")

    base_binding = _object(sidecar.get("base"), "sidecar.base")
    _require_exact_keys(base_binding, {"path", "sha256", "columns"}, "sidecar.base")
    _require(
        base_binding.get("columns") == GRASSY_BASE_COLUMNS,
        "sidecar base-column count mismatch",
    )
    _require(
        _safe_dataset_path(root, base_binding.get("path"), "sidecar.base.path") == base,
        "sidecar manifest points to a different Parquet file",
    )
    base_sha256 = _sha256_file(base)
    _require(base_binding.get("sha256") == base_sha256, "sidecar/base hash mismatch")

    _validate_dataset_manifest(
        root=root,
        base_path=base,
        base_sha256=base_sha256,
        sidecar_manifest_path=sidecar_manifest_path,
        sidecar_manifest_sha256=_sha256_bytes(sidecar_raw),
        sidecar=sidecar,
        rows=rows,
    )
    _validate_row_identities(root=root, sidecar=sidecar, rows=rows)
    action_mask = _load_mask(
        root=root,
        sidecar=sidecar,
        section_name="actionObservation",
        expected_shape=(rows, GRASSY_SEQUENCE_LENGTH, GRASSY_DISCRETE_WIDTH),
        expected_forced_false_heads=[60, 61, 62, 63],
    )
    continuous_mask = _load_mask(
        root=root,
        sidecar=sidecar,
        section_name="continuousActionObservation",
        expected_shape=(rows, GRASSY_SEQUENCE_LENGTH, GRASSY_CONTINUOUS_WIDTH),
        expected_forced_false_heads=[5, 6, 7],
    )
    return ObservationMaskSidecar(
        base_path=base,
        rows=rows,
        outcome_cutoff_ms=cutoff,
        action_mask=action_mask,
        continuous_action_mask=continuous_mask,
    )
