# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from xrex.data.recsys.observability import (
    ACTION_OBSERVATION_MASK_COLUMN,
    CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
    GRASSY_CONTINUOUS_PADDING_HEADS,
    GRASSY_DISCRETE_PADDING_HEADS,
    append_observation_mask_columns,
    extract_observation_mask_column,
    validate_observation_mask_array,
)


def _fixed_size_array(values: np.ndarray) -> pa.FixedSizeListArray:
    _, sequence_length, width = values.shape
    scalar_type = pa.bool_() if values.dtype == np.bool_ else pa.float32()
    scalars = pa.array(values.reshape(-1), type=scalar_type)
    per_position = pa.FixedSizeListArray.from_arrays(scalars, width)
    return pa.FixedSizeListArray.from_arrays(per_position, sequence_length)


def _base_batch(batch_size: int = 2, sequence_length: int = 3) -> pa.RecordBatch:
    actions = np.zeros((batch_size, sequence_length, 64), dtype=np.bool_)
    continuous = np.zeros((batch_size, sequence_length, 8), dtype=np.float32)
    return pa.RecordBatch.from_arrays(
        [_fixed_size_array(actions), _fixed_size_array(continuous)],
        ["actionNameMultiHotSeqSeq", "continuousActionValuesSeqSeq"],
    )


class ObservabilityMaskTest(unittest.TestCase):
    def test_absent_columns_preserve_base_schema_and_return_none(self):
        base = _base_batch()
        self.assertIs(append_observation_mask_columns(base), base)
        self.assertIsNone(
            extract_observation_mask_column(
                base,
                ACTION_OBSERVATION_MASK_COLUMN,
                2,
                3,
                64,
            )
        )
        self.assertEqual(
            base.schema.names,
            ["actionNameMultiHotSeqSeq", "continuousActionValuesSeqSeq"],
        )

    def test_sidecar_bridge_is_per_row_position_and_head(self):
        base = _base_batch()
        action_mask = np.ones((2, 3, 64), dtype=np.bool_)
        continuous_mask = np.ones((2, 3, 8), dtype=np.bool_)
        action_mask[1, 2, 7] = False
        continuous_mask[0, 1, 2] = False

        augmented = append_observation_mask_columns(
            base,
            action_mask=action_mask,
            continuous_action_mask=continuous_mask,
        )

        self.assertEqual(
            base.schema.names,
            ["actionNameMultiHotSeqSeq", "continuousActionValuesSeqSeq"],
        )
        self.assertEqual(
            augmented.schema.names[-2:],
            [ACTION_OBSERVATION_MASK_COLUMN, CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN],
        )
        actual_action = extract_observation_mask_column(
            augmented,
            ACTION_OBSERVATION_MASK_COLUMN,
            2,
            3,
            64,
        )
        actual_continuous = extract_observation_mask_column(
            augmented,
            CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
            2,
            3,
            8,
        )
        assert actual_action is not None
        assert actual_continuous is not None
        self.assertFalse(actual_action[1, 2, 7])
        self.assertFalse(actual_continuous[0, 1, 2])
        self.assertTrue(actual_action[0, 0, 6])
        self.assertTrue(actual_continuous[1, 2, 4])

    def test_reserved_heads_are_forced_unobserved(self):
        augmented = append_observation_mask_columns(
            _base_batch(),
            action_mask=np.ones((2, 3, 64), dtype=np.bool_),
            continuous_action_mask=np.ones((2, 3, 8), dtype=np.bool_),
        )
        action_mask = extract_observation_mask_column(
            augmented,
            ACTION_OBSERVATION_MASK_COLUMN,
            2,
            3,
            64,
        )
        continuous_mask = extract_observation_mask_column(
            augmented,
            CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
            2,
            3,
            8,
        )
        assert action_mask is not None
        assert continuous_mask is not None
        self.assertFalse(action_mask[..., GRASSY_DISCRETE_PADDING_HEADS].any())
        self.assertFalse(continuous_mask[..., GRASSY_CONTINUOUS_PADDING_HEADS].any())

    def test_bridge_rejects_wrong_row_sequence_head_and_dtype(self):
        base = _base_batch()
        bad_masks = (
            np.ones((1, 3, 64), dtype=np.bool_),
            np.ones((2, 2, 64), dtype=np.bool_),
            np.ones((2, 3, 63), dtype=np.bool_),
            np.ones((2, 3, 64), dtype=np.float32),
        )
        for mask in bad_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    append_observation_mask_columns(base, action_mask=mask)

    def test_reader_rejects_wrong_declared_head_count(self):
        base = _base_batch()
        wrong = base.append_column(
            ACTION_OBSERVATION_MASK_COLUMN,
            _fixed_size_array(np.ones((2, 3, 63), dtype=np.bool_)),
        )
        with self.assertRaisesRegex(ValueError, "head count 63 != expected 64"):
            extract_observation_mask_column(
                wrong,
                ACTION_OBSERVATION_MASK_COLUMN,
                2,
                3,
                64,
            )

    def test_reader_rejects_null_mask_values(self):
        base = _base_batch()
        values = [True] * (2 * 3 * 64)
        values[17] = None
        scalars = pa.array(values, type=pa.bool_())
        per_position = pa.FixedSizeListArray.from_arrays(scalars, 64)
        with_null = pa.FixedSizeListArray.from_arrays(per_position, 3)
        malformed = base.append_column(ACTION_OBSERVATION_MASK_COLUMN, with_null)
        with self.assertRaisesRegex(ValueError, "must not contain nulls"):
            extract_observation_mask_column(
                malformed,
                ACTION_OBSERVATION_MASK_COLUMN,
                2,
                3,
                64,
            )

    def test_array_validator_rejects_shape_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_observation_mask_array(
                np.ones((2, 3, 4), dtype=np.bool_),
                (2, 3, 5),
                "mask",
            )
        with self.assertRaisesRegex(ValueError, "boolean dtype"):
            validate_observation_mask_array(
                np.ones((2, 3, 4), dtype=np.float32),
                (2, 3, 4),
                "mask",
            )


if __name__ == "__main__":
    unittest.main()
