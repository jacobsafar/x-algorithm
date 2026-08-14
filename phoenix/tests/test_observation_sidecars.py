# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xrex.data.recsys.observability import (
    ACTION_OBSERVATION_MASK_COLUMN,
    CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
    extract_observation_mask_column,
)
from xrex.data.recsys.observation_sidecars import (
    GRASSY_DATASET_MANIFEST,
    GRASSY_DATASET_MANIFEST_VERSION,
    GRASSY_EXPORTER_NAME,
    GRASSY_EXPORTER_VERSION,
    GRASSY_INPUT_SCHEMA_VERSION,
    GRASSY_REQUIRED_MODEL_GATES,
    GRASSY_UPSTREAM_COMMIT,
    ObservationSidecarError,
    load_observation_mask_sidecar,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixed_size_array(values: np.ndarray) -> pa.FixedSizeListArray:
    _, sequence_length, width = values.shape
    scalar_type = pa.bool_() if values.dtype == np.bool_ else pa.float32()
    scalars = pa.array(values.reshape(-1), type=scalar_type)
    per_position = pa.FixedSizeListArray.from_arrays(scalars, width)
    return pa.FixedSizeListArray.from_arrays(per_position, sequence_length)


class _Artifact:
    def __init__(self, root: Path, rows: int = 4):
        self.root = root
        self.rows = rows
        self.batch_dir = root / "partition=0" / "0"
        self.batch_dir.mkdir(parents=True)
        self.base = self.batch_dir / "batch_0.parquet"
        self.action = self.batch_dir / "batch_0.action_observation.npy"
        self.continuous = self.batch_dir / "batch_0.continuous_action_observation.npy"
        self.identities = self.batch_dir / "batch_0.example_ids.jsonl"
        self.sidecar = self.batch_dir / "batch_0.observability.json"

        actions = np.zeros((rows, 1086, 64), dtype=np.bool_)
        continuous_values = np.zeros((rows, 1086, 8), dtype=np.float32)
        arrays: list[pa.Array] = [
            _fixed_size_array(actions),
            _fixed_size_array(continuous_values),
        ]
        names = ["actionNameMultiHotSeqSeq", "continuousActionValuesSeqSeq"]
        for index in range(34):
            arrays.append(pa.array(np.full(rows, index, dtype=np.int32)))
            names.append(f"field_{index}")
        pq.write_table(pa.Table.from_arrays(arrays, names=names), self.base)

        action_mask = np.zeros((rows, 1086, 64), dtype=np.bool_)
        continuous_mask = np.zeros((rows, 1086, 8), dtype=np.bool_)
        for row in range(rows):
            action_mask[row, 0, row + 1] = True
            continuous_mask[row, 0, 1] = row % 2 == 0
        np.save(self.action, action_mask, allow_pickle=False)
        np.save(self.continuous, continuous_mask, allow_pickle=False)

        self.identities.write_text(
            "".join(
                json.dumps(
                    {
                        "exampleId": f"{row + 1:032x}",
                        "rolloutReleaseId": "release-1",
                        "platform": "ios",
                        "appVersion": "1.0",
                        "surface": "for_you",
                    },
                    sort_keys=True,
                )
                + "\n"
                for row in range(rows)
            ),
            encoding="utf-8",
        )
        self.sidecar_value = {
            "manifestVersion": 1,
            "exporterVersion": GRASSY_EXPORTER_VERSION,
            "upstreamCommit": GRASSY_UPSTREAM_COMMIT,
            "labelContractVersion": 2,
            "outcomeCutoffMs": 1_786_672_800_000,
            "rows": rows,
            "base": {
                "path": "partition=0/0/batch_0.parquet",
                "sha256": _sha256(self.base),
                "columns": 36,
            },
            "rowIdentity": {
                "path": "partition=0/0/batch_0.example_ids.jsonl",
                "sha256": _sha256(self.identities),
                "encoding": "jsonl-opaque-keyed-blake2b-128-plus-rollout-provenance-v1",
            },
            "actionObservation": {
                "path": "partition=0/0/batch_0.action_observation.npy",
                "sha256": _sha256(self.action),
                "dtype": "bool",
                "shape": [rows, 1086, 64],
                "forcedFalseHeads": [60, 61, 62, 63],
            },
            "continuousActionObservation": {
                "path": "partition=0/0/batch_0.continuous_action_observation.npy",
                "sha256": _sha256(self.continuous),
                "dtype": "bool",
                "shape": [rows, 1086, 8],
                "forcedFalseHeads": [5, 6, 7],
            },
            "rolloutsUsed": [
                {
                    "releaseId": "release-1",
                    "platform": "ios",
                    "appVersion": "1.0",
                    "effectiveFromMs": 1_786_000_000_000,
                    "effectiveToMs": 1_787_000_000_000,
                    "surfaces": ["for_you"],
                }
            ],
        }
        self.rewrite_manifests()

    def rewrite_manifests(self) -> None:
        _write_json(self.sidecar, self.sidecar_value)
        dataset = {
            "manifest_version": GRASSY_DATASET_MANIFEST_VERSION,
            "exporter": {
                "name": GRASSY_EXPORTER_NAME,
                "version": GRASSY_EXPORTER_VERSION,
                "input_schema_version": GRASSY_INPUT_SCHEMA_VERSION,
                "source_sha256": "a" * 64,
                "base_adapter_sha256": "b" * 64,
            },
            "upstream_commit": GRASSY_UPSTREAM_COMMIT,
            "action_contract_version": 2,
            "outcome_cutoff_ms": self.sidecar_value["outcomeCutoffMs"],
            "rows": self.rows,
            "parquet": "partition=0/0/batch_0.parquet",
            "parquet_sha256": self.sidecar_value["base"]["sha256"],
            "sidecar_manifest": "partition=0/0/batch_0.observability.json",
            "sidecar_manifest_sha256": _sha256(self.sidecar),
            "row_identity_sha256": self.sidecar_value["rowIdentity"]["sha256"],
            "shape": {
                "history_positions": 1022,
                "candidate_positions": 64,
                "sequence_positions": 1086,
                "discrete_action_width": 64,
                "continuous_action_width": 8,
                "base_columns": 36,
            },
            "training_eligible": True,
            "required_model_gates": GRASSY_REQUIRED_MODEL_GATES,
        }
        _write_json(self.root / GRASSY_DATASET_MANIFEST, dataset)

    def load(self):
        parquet = pq.ParquetFile(self.base)
        return load_observation_mask_sidecar(
            base_path=self.base,
            dataset_root=self.root,
            parquet_num_rows=parquet.metadata.num_rows,
            parquet_num_columns=len(parquet.schema_arrow),
        )


class ObservationSidecarTest(unittest.TestCase):
    def test_valid_artifact_attaches_exact_windows_out_of_order(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            sidecar = artifact.load()
            batches = list(pq.ParquetFile(artifact.base).iter_batches(batch_size=2))

            # A prefetched/interleaved consumer may attach later windows first;
            # offsets are explicit and never inferred from delivery order.
            later = sidecar.attach(batches[1], 2)
            earlier = sidecar.attach(batches[0], 0)
            later_action = extract_observation_mask_column(
                later, ACTION_OBSERVATION_MASK_COLUMN, 2, 1086, 64
            )
            earlier_continuous = extract_observation_mask_column(
                earlier, CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN, 2, 1086, 8
            )
            assert later_action is not None
            assert earlier_continuous is not None
            self.assertTrue(later_action[0, 0, 3])
            self.assertTrue(later_action[1, 0, 4])
            self.assertFalse(later_action[..., 60:64].any())
            self.assertTrue(earlier_continuous[0, 0, 1])
            self.assertFalse(earlier_continuous[1, 0, 1])
            self.assertFalse(earlier_continuous[..., 5:8].any())

    def test_missing_sidecar_or_dataset_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.sidecar.unlink()
            with self.assertRaisesRegex(ObservationSidecarError, "sidecar manifest is missing"):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            (artifact.root / GRASSY_DATASET_MANIFEST).unlink()
            with self.assertRaisesRegex(ObservationSidecarError, "dataset manifest: cannot read"):
                artifact.load()

    def test_every_grassy_model_gate_is_manifest_pinned(self):
        for gate, unsafe_value in (
            ("require_label_observation_masks", False),
            ("enable_engagement_counts", True),
            ("compute_post_unexplored_label", True),
        ):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                artifact = _Artifact(Path(directory))
                manifest_path = artifact.root / GRASSY_DATASET_MANIFEST
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["required_model_gates"][gate] = unsafe_value
                _write_json(manifest_path, manifest)
                with self.assertRaisesRegex(
                    ObservationSidecarError,
                    "exact Grassy model gates",
                ):
                    artifact.load()

    def test_backend_manifest_and_exporter_versions_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            manifest_path = artifact.root / GRASSY_DATASET_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["manifest_version"] = GRASSY_DATASET_MANIFEST_VERSION - 1
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ObservationSidecarError,
                "dataset manifest version mismatch",
            ):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            manifest_path = artifact.root / GRASSY_DATASET_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["exporter"]["version"] = GRASSY_EXPORTER_VERSION - 1
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ObservationSidecarError,
                "dataset exporter contract mismatch",
            ):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            manifest_path = artifact.root / GRASSY_DATASET_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["exporter"]["source_sha256"] = "not-a-sha"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ObservationSidecarError,
                "invalid SHA-256",
            ):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.sidecar_value["exporterVersion"] = GRASSY_EXPORTER_VERSION - 1
            artifact.rewrite_manifests()
            with self.assertRaisesRegex(
                ObservationSidecarError,
                "sidecar exporter version mismatch",
            ):
                artifact.load()

    def test_base_mask_and_identity_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.sidecar_value["base"]["sha256"] = "0" * 64
            artifact.rewrite_manifests()
            with self.assertRaisesRegex(ObservationSidecarError, "sidecar/base hash mismatch"):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.action.write_bytes(artifact.action.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                ObservationSidecarError, "actionObservation: hash mismatch"
            ):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.identities.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ObservationSidecarError, "row-identity hash mismatch"):
                artifact.load()

    def test_declared_and_physical_shape_dtype_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            artifact.sidecar_value["actionObservation"]["shape"] = [artifact.rows, 1085, 64]
            artifact.rewrite_manifests()
            with self.assertRaisesRegex(ObservationSidecarError, "declared shape mismatch"):
                artifact.load()

        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            np.save(
                artifact.action,
                np.zeros((artifact.rows, 1086, 64), dtype=np.uint8),
                allow_pickle=False,
            )
            artifact.sidecar_value["actionObservation"]["sha256"] = _sha256(artifact.action)
            artifact.rewrite_manifests()
            with self.assertRaisesRegex(ObservationSidecarError, "physical dtype is not bool"):
                artifact.load()

    def test_wrong_row_offset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _Artifact(Path(directory))
            sidecar = artifact.load()
            batch = next(pq.ParquetFile(artifact.base).iter_batches(batch_size=2))
            with self.assertRaisesRegex(ObservationSidecarError, "exceeds"):
                sidecar.attach(batch, artifact.rows - 1)


if __name__ == "__main__":
    unittest.main()
