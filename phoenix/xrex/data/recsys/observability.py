# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""Optional per-head label observability for recommendation examples.

An action value of ``False`` is only a negative label when its corresponding
observability value is ``True``.  A ``False`` observability value means the
label is unknown and must not contribute to loss or action metrics.

The Parquet columns are optional so existing X data keeps its legacy
all-heads-observed behavior without a schema migration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from numpy import typing as npt


ACTION_OBSERVATION_MASK_COLUMN = "actionObservationMaskSeqSeq"
CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN = "continuousActionObservationMaskSeqSeq"

# Grassy's current 64-discrete/8-continuous export contract reserves these
# aligned output slots.  They must never become implicit negative labels.
GRASSY_DISCRETE_PADDING_HEADS = (60, 61, 62, 63)
GRASSY_CONTINUOUS_PADDING_HEADS = (5, 6, 7)


def validate_observation_mask_array(
    mask: Any,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    """Reject masks that could silently reinterpret unknown labels as negatives."""

    actual_shape = tuple(mask.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"{name} shape {actual_shape} != expected {expected_shape}")
    if np.dtype(mask.dtype) != np.dtype(np.bool_):
        raise ValueError(f"{name} must have boolean dtype, got {mask.dtype}")


def force_reserved_heads_unobserved(
    mask: npt.NDArray[np.bool_],
    reserved_heads: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return a mask with reserved output slots defensively disabled."""

    applicable = [head for head in reserved_heads if head < mask.shape[-1]]
    if not applicable:
        return mask
    result = mask.copy()
    result[..., applicable] = False
    return result


def extract_observation_mask_column(
    record_batch: pa.RecordBatch,
    column_name: str,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
) -> npt.NDArray[np.bool_] | None:
    """Read an optional ``[batch, sequence, head]`` fixed-list boolean mask.

    Every structural or nullability mismatch is rejected.  Falling back to an
    all-ones mask on malformed input would turn unknown labels into negatives,
    so this path intentionally fails closed.
    """

    if column_name not in (
        ACTION_OBSERVATION_MASK_COLUMN,
        CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
    ):
        raise ValueError(f"unsupported observation-mask column {column_name}")

    if column_name not in record_batch.schema.names:
        return None
    if record_batch.num_rows != batch_size:
        raise ValueError(
            f"{column_name} batch size {batch_size} != record batch rows "
            f"{record_batch.num_rows}"
        )

    column = record_batch.column(column_name)
    if not pa.types.is_fixed_size_list(column.type):
        raise ValueError(
            f"{column_name} must be fixed_size_list<fixed_size_list<bool>>, got {column.type}"
        )
    if column.type.list_size != sequence_length:
        raise ValueError(
            f"{column_name} sequence size {column.type.list_size} != expected {sequence_length}"
        )

    inner_type = column.type.value_type
    if not pa.types.is_fixed_size_list(inner_type):
        raise ValueError(
            f"{column_name} values must be fixed_size_list<bool>, got {inner_type}"
        )
    if inner_type.list_size != num_heads:
        raise ValueError(
            f"{column_name} head count {inner_type.list_size} != expected {num_heads}"
        )
    if not pa.types.is_boolean(inner_type.value_type):
        raise ValueError(f"{column_name} values must be boolean, got {inner_type.value_type}")

    inner_values = column.values
    scalar_values = inner_values.values
    if column.null_count or inner_values.null_count or scalar_values.null_count:
        raise ValueError(f"{column_name} must not contain nulls")

    flat = scalar_values.to_numpy(zero_copy_only=False)
    mask = flat.reshape(batch_size, sequence_length, num_heads).astype(np.bool_, copy=False)
    validate_observation_mask_array(
        mask,
        (batch_size, sequence_length, num_heads),
        column_name,
    )
    reserved_heads = (
        GRASSY_DISCRETE_PADDING_HEADS
        if column_name == ACTION_OBSERVATION_MASK_COLUMN
        else GRASSY_CONTINUOUS_PADDING_HEADS
    )
    return force_reserved_heads_unobserved(mask, reserved_heads)


def _fixed_size_mask_array(mask: npt.NDArray[np.bool_]) -> pa.FixedSizeListArray:
    _, sequence_length, num_heads = mask.shape
    scalars = pa.array(mask.reshape(-1), type=pa.bool_())
    per_position = pa.FixedSizeListArray.from_arrays(scalars, num_heads)
    return pa.FixedSizeListArray.from_arrays(per_position, sequence_length)


def append_observation_mask_columns(
    record_batch: pa.RecordBatch,
    action_mask: npt.NDArray[np.bool_] | None = None,
    continuous_action_mask: npt.NDArray[np.bool_] | None = None,
) -> pa.RecordBatch:
    """Attach aligned sidecar masks to an in-memory base RecordBatch.

    This bridge lets a producer retain the exact upstream Parquet schema on
    disk.  The sidecar must use identical row ordering and sequence length;
    malformed or duplicate input is rejected rather than silently aligned.
    """

    masks = (
        (
            ACTION_OBSERVATION_MASK_COLUMN,
            action_mask,
            GRASSY_DISCRETE_PADDING_HEADS,
        ),
        (
            CONTINUOUS_ACTION_OBSERVATION_MASK_COLUMN,
            continuous_action_mask,
            GRASSY_CONTINUOUS_PADDING_HEADS,
        ),
    )
    result = record_batch
    if "actionNameMultiHotSeqSeq" not in record_batch.schema.names:
        raise ValueError("base record batch is missing actionNameMultiHotSeqSeq")
    action_type = record_batch.schema.field("actionNameMultiHotSeqSeq").type
    if not (
        pa.types.is_fixed_size_list(action_type)
        and pa.types.is_fixed_size_list(action_type.value_type)
        and pa.types.is_boolean(action_type.value_type.value_type)
    ):
        raise ValueError(
            "base actionNameMultiHotSeqSeq must be "
            f"fixed_size_list<fixed_size_list<bool>>, got {action_type}"
        )
    expected_sequence_length = action_type.list_size
    expected_action_heads = action_type.value_type.list_size
    expected_continuous_heads: int | None = None
    if "continuousActionValuesSeqSeq" in record_batch.schema.names:
        continuous_type = record_batch.schema.field("continuousActionValuesSeqSeq").type
        if not (
            pa.types.is_fixed_size_list(continuous_type)
            and pa.types.is_fixed_size_list(continuous_type.value_type)
            and pa.types.is_floating(continuous_type.value_type.value_type)
        ):
            raise ValueError(
                "base continuousActionValuesSeqSeq must be "
                f"fixed_size_list<fixed_size_list<float>>, got {continuous_type}"
            )
        if continuous_type.list_size != expected_sequence_length:
            raise ValueError(
                "base continuousActionValuesSeqSeq sequence size "
                f"{continuous_type.list_size} != action sequence size "
                f"{expected_sequence_length}"
            )
        expected_continuous_heads = continuous_type.value_type.list_size
    elif continuous_action_mask is not None:
        raise ValueError("base record batch is missing continuousActionValuesSeqSeq")

    for column_name, raw_mask, reserved_heads in masks:
        if raw_mask is None:
            continue
        if column_name in result.schema.names:
            raise ValueError(f"record batch already contains {column_name}")
        if raw_mask.ndim != 3:
            raise ValueError(f"{column_name} must be rank 3, got shape {raw_mask.shape}")
        expected_shape = tuple(raw_mask.shape)
        validate_observation_mask_array(raw_mask, expected_shape, column_name)
        if raw_mask.shape[0] != record_batch.num_rows:
            raise ValueError(
                f"{column_name} row count {raw_mask.shape[0]} != base rows "
                f"{record_batch.num_rows}"
            )
        if (
            expected_sequence_length is not None
            and raw_mask.shape[1] != expected_sequence_length
        ):
            raise ValueError(
                f"{column_name} sequence size {raw_mask.shape[1]} != base sequence size "
                f"{expected_sequence_length}"
            )
        expected_heads = (
            expected_action_heads
            if column_name == ACTION_OBSERVATION_MASK_COLUMN
            else expected_continuous_heads
        )
        if expected_heads is not None and raw_mask.shape[2] != expected_heads:
            raise ValueError(
                f"{column_name} head count {raw_mask.shape[2]} != base head count "
                f"{expected_heads}"
            )
        mask = force_reserved_heads_unobserved(raw_mask, reserved_heads)
        result = result.append_column(column_name, _fixed_size_mask_array(mask))
    return result
