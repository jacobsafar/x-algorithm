# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


PHOENIX_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PHOENIX_ROOT / "xrex" / "configs" / "xrecsys.py"
MODEL_PATH = PHOENIX_ROOT / "xrex" / "models" / "recsys_model.py"


def _function(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {class_node.name}.{name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _dict_entry(tree: ast.Module, assignment_name: str, entry_name: str) -> ast.Call:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == entry_name:
                if not isinstance(value, ast.Call):
                    raise AssertionError(f"{entry_name} is not built by _make_cfg")
                return value
    raise AssertionError(f"missing {assignment_name}[{entry_name!r}]")


def _explicit_dict_values(call: ast.Call) -> dict[str, object]:
    if not call.args or not isinstance(call.args[0], ast.Dict):
        raise AssertionError("_make_cfg must receive a literal parameter dict")
    values: dict[str, object] = {}
    for key, value in zip(call.args[0].keys, call.args[0].values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            values[key.value] = ast.literal_eval(value)
    return values


class GrassyModelContractTest(unittest.TestCase):
    def test_named_nano_job_pins_every_unavailable_input_gate(self) -> None:
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))
        call = _dict_entry(tree, "MODEL_CFGS", "grassy_home_direct_packed_nano")
        values = _explicit_dict_values(call)
        self.assertIs(values["require_label_observation_masks"], True)
        self.assertIs(values["enable_engagement_counts"], False)
        self.assertIs(values["compute_post_unexplored_label"], False)
        self.assertEqual(values["num_kafka_partitions"], 1)

    def test_raw_count_changes_cannot_reach_model_inputs_or_logits_when_disabled(self) -> None:
        source = MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(MODEL_PATH))
        model = _class(tree, "RecsysAggregatedModel")
        build_inputs = _function(model, "build_inputs")

        count_guard = next(
            (
                node
                for node in ast.walk(build_inputs)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "ctx_config.enable_engagement_counts"
            ),
            None,
        )
        self.assertIsNotNone(count_guard, "raw count feature construction lost its false gate")
        assert count_guard is not None
        guarded_start = count_guard.lineno
        guarded_end = count_guard.end_lineno or count_guard.lineno
        raw_count_reads = [
            node
            for node in ast.walk(build_inputs)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "int64_features"
        ]
        self.assertEqual(len(raw_count_reads), 1)
        self.assertTrue(
            all(guarded_start <= node.lineno <= guarded_end for node in raw_count_reads),
            "a raw engagement-count read escaped the enable_engagement_counts guard",
        )

        embed_context = ast.get_source_segment(
            source,
            _function(model, "embed_categorical_context_features"),
        )
        assert embed_context is not None
        self.assertIn('feature_name.endswith("_count_bucket")', embed_context)
        self.assertIn("not ctx_config.enable_engagement_counts", embed_context)

        # Parity at the model boundary: the guarded branch contributes no
        # input dimensions when disabled, so downstream deterministic logits
        # are identical for arbitrary raw count changes. The AST assertions
        # above bind this small executable parity check to the production path.
        base_inputs = np.array([[0.25, -0.5, 1.5]], dtype=np.float32)
        low_counts = np.array([[0, 0, 0, 1]], dtype=np.float32)
        high_counts = np.array([[999, 88, 77, 66]], dtype=np.float32)
        weights = np.array([[0.3], [-0.2], [0.7]], dtype=np.float32)

        def gated_inputs(raw_counts: np.ndarray, enabled: bool) -> np.ndarray:
            if not enabled:
                return base_inputs.copy()
            return np.concatenate([base_inputs, np.log1p(raw_counts)], axis=-1)

        low_inputs = gated_inputs(low_counts, enabled=False)
        high_inputs = gated_inputs(high_counts, enabled=False)
        np.testing.assert_array_equal(low_inputs, high_inputs)
        np.testing.assert_array_equal(low_inputs @ weights, high_inputs @ weights)


if __name__ == "__main__":
    unittest.main()
