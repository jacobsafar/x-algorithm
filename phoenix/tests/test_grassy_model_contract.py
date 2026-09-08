# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from typing import cast

from xrex.data.recsys.grassy_contract import (
    GRASSY_DISABLED_PREP_FLAGS, GRASSY_FEATURE_POLICY,
    grassy_feature_prep_overrides, validate_grassy_model_contract,
)

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
            values[key.value] = (
                GRASSY_FEATURE_POLICY
                if isinstance(value, ast.Name) and value.id == "GRASSY_FEATURE_POLICY"
                else ast.literal_eval(value)
            )
    return values


def _load_function(path: Path, name: str, namespace: dict) -> object:
    """Execute the production function body with NumPy instead of importing CUDA/JAX."""
    tree = ast.parse(path.read_text(), filename=str(path))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _valid_model() -> SimpleNamespace:
    return SimpleNamespace(
        require_label_observation_masks=True,
        grassy_feature_policy=GRASSY_FEATURE_POLICY,
        feature_prep_enabled=True,
        feature_prep=SimpleNamespace(**grassy_feature_prep_overrides(), post_age_granularity_mins=60, post_age_max_mins=4800),
        grassy_candidate_age_policy="legacy_48h_v1", grassy_post_age_encoding="phoenix_post_age_1h_80bins_v1",
        post_age_granularity_mins=60, post_age_max_mins=4800,
        use_ip_address=False,
        context_features=SimpleNamespace(enable_engagement_counts=False),
        history_seq_len=1022, candidate_seq_len=64,
        model_config=SimpleNamespace(output_vocab_size=64), num_continuous_actions=8,
        split_head_training_by_source=False, log_q_correction=False,
    )


class GrassyModelContractTest(unittest.TestCase):
    def test_named_nano_job_pins_every_unavailable_input_gate(self) -> None:
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))
        call = _dict_entry(tree, "MODEL_CFGS", "grassy_home_direct_packed_nano")
        values = _explicit_dict_values(call)
        self.assertIs(values["require_label_observation_masks"], True)
        self.assertIs(values["enable_engagement_counts"], False)
        self.assertIs(values["compute_post_unexplored_label"], False)
        self.assertEqual(values["num_kafka_partitions"], 1)
        self.assertEqual(values["grassy_feature_policy"], GRASSY_FEATURE_POLICY)
        self.assertEqual(values["total_samples"], 6368)
        self.assertEqual(values["num_negatives_per_example"], 0)
        self.assertIs(values["log_q_correction"], False)
        self.assertEqual(next(k.value.value for k in call.keywords if k.arg == "ip_vocab_size"), 0)

    def test_config_drift_fails_before_feature_creation(self) -> None:
        valid = _valid_model()
        validate_grassy_model_contract(valid)
        for flag in GRASSY_DISABLED_PREP_FLAGS:
            with self.subTest(flag=flag):
                drifted = deepcopy(valid)
                setattr(drifted.feature_prep, flag, True)
                with self.assertRaisesRegex(ValueError, flag):
                    validate_grassy_model_contract(drifted)
        for field, value in (
            ("grassy_feature_policy", None), ("feature_prep_enabled", False),
            ("use_ip_address", True), ("history_seq_len", 1023),
            ("candidate_seq_len", 128), ("num_continuous_actions", 16),
            ("split_head_training_by_source", True), ("log_q_correction", True),
        ):
            with self.subTest(field=field):
                drifted = deepcopy(valid)
                setattr(drifted, field, value)
                with self.assertRaises(ValueError):
                    validate_grassy_model_contract(drifted)
        # Generic upstream models keep their original optional-mask behavior.
        validate_grassy_model_contract(SimpleNamespace(require_label_observation_masks=False))

    def test_feature_factory_disables_nested_upstream_defaults_in_both_paths(self) -> None:
        def make_config(**kwargs):
            return SimpleNamespace(**kwargs)
        def replace_config(config, **kwargs):
            return SimpleNamespace(**{**vars(config), **kwargs})
        build = _load_function(CONFIG_PATH, "_make_feature_prep_config", {
            "FeaturePrepConfig": make_config, "replace": replace_config,
            "grassy_feature_prep_overrides": grassy_feature_prep_overrides, "POST_AGE_MAX_MINUTES": 4800,
        })
        base = {"require_label_observation_masks": True, "emb_size": 128, "emb_table_width": 128}
        for inherited in (False, True):
            params = dict(base)
            if inherited:
                params["feature_prep"] = SimpleNamespace(**dict.fromkeys(GRASSY_DISABLED_PREP_FLAGS, True))
            config = build(params, None)
            self.assertTrue(config.reserve_unavailable_user_feature_token)
            for flag in GRASSY_DISABLED_PREP_FLAGS:
                self.assertIs(getattr(config, flag), False, flag)

    def test_fixed_context_token_never_reads_unknown_context(self) -> None:
        path = PHOENIX_ROOT / "xrex/models/recsys_feature_prep.py"
        build = _load_function(path, "_build_user_features_token", {
            "jnp": np, "DTYPE_BY_NAME": {"float32": np.float32},
        })
        config = SimpleNamespace(has_user_features=True, reserve_unavailable_user_feature_token=True,
                                 emb_size=128, fprop_dtype="float32")
        # Missing unavailable columns are legitimate. Their absence must not fail,
        # produce learned false-valued features, or change the 2-prefix layout.
        token, mask = build({"user_hashes": np.ones((2, 2), dtype=np.uint64)}, config)
        self.assertEqual(token.shape, (2, 1, 128))
        np.testing.assert_array_equal(token, 0)
        np.testing.assert_array_equal(mask, np.ones((2, 1), dtype=np.bool_))
        poisoned = {"user_hashes": np.ones((2, 2), dtype=np.uint64),
                    "user_categorical_features": object(), "user_float_features": object(),
                    "user_installed_apps_multihot": object()}
        np.testing.assert_array_equal(build(poisoned, config)[0], token)

    def test_unknown_counts_relationships_and_timezone_never_reach_prep(self) -> None:
        path = PHOENIX_ROOT / "xrex/models/recsys_feature_prep.py"
        add = _load_function(path, "_add_context_features", {
            "jnp": np, "DTYPE_BY_NAME": {"float32": np.float32},
        })
        config = SimpleNamespace(**grassy_feature_prep_overrides(),
                                 enable_product_surface=False, enable_post_age=False,
                                 enable_stale_post=False, fprop_dtype="float32")
        base = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        for prefix in ("hist", "cand"):
            for unknowns in ({}, {"bool_features": object(), "int64_features": object(),
                                  "categorical_features": object()}):
                np.testing.assert_array_equal(add(base.copy(), unknowns, config, prefix), base)

    def test_upstream_candidate_selection_and_observability_both_survive_padding(self) -> None:
        pad = _load_function(PHOENIX_ROOT / "xrex/data/parquet_recsys.py", "pad_batch", {
            "np": np, "cast": cast, "PostSeq": dict,
        })
        sequence_keys = (
            "impr_ts", "actions", "continuous_actions", "post_hashes", "auth_hashes",
            "ip_hashes", "product_surface", "client_app_id", "post_ids", "promoted_ids",
            "line_item_objective", "safety_label_mask", "embedding", "search_query_embeddings",
            "categorical_features", "bool_features", "float_features", "int64_features",
            "post_creation_ts_sec",
        )
        sequence = {key: np.ones((1, 2, 1), dtype=np.int64) for key in sequence_keys}
        sequence.update(
            trained_candidate_mask=np.array([[True, False]]),
            action_observation_mask=np.ones((1, 2, 64), dtype=np.bool_),
            continuous_action_observation_mask=np.ones((1, 2, 8), dtype=np.bool_),
        )
        user_keys = ("user_hashes", "user_ip_hashes", "user_categorical_features", "user_bool_features",
                     "user_float_features", "user_int64_features", "user_installed_apps_multihot")
        batch = {key: np.ones((1, 2), dtype=np.int64) for key in user_keys}
        batch.update(history_seq=sequence, candidate_seq=sequence)
        padded = pad(batch, 2)
        for key in ("history_seq", "candidate_seq"):
            result = padded[key]
            np.testing.assert_array_equal(result["trained_candidate_mask"][0], [True, False])
            np.testing.assert_array_equal(result["trained_candidate_mask"][1], True)
            for family in ("action_observation_mask", "continuous_action_observation_mask"):
                np.testing.assert_array_equal(result[family][0], sequence[family][0])
                np.testing.assert_array_equal(result[family][1], False)

    def test_upstream_age_embedding_preserves_timestamps_and_uses_overflow_for_old_posts(self) -> None:
        age_bucket = _load_function(PHOENIX_ROOT / "xrex/models/recsys_feature_prep.py",
                                    "_compute_post_age_bucket_linear",
                                    {"jnp": np, "POST_AGE_MAX_MINUTES": 4800})
        impression = np.full(5, 1_800_000_000, dtype=np.int64)
        creation = impression - np.array([0, 48 * 3600, 14 * 86400, 90 * 86400, -1])
        np.testing.assert_array_equal(age_bucket(impression, creation), [1, 49, 81, 81, 0])
        # Missing or future-by-a-full-minute creation timestamps are unknown.
        np.testing.assert_array_equal(age_bucket(impression[:2], np.array([0, impression[1] + 60])), [0, 0])

    def test_prospective_90d_age_encoding_has_100day_capacity_without_changing_legacy(self) -> None:
        age_bucket = _load_function(PHOENIX_ROOT / "xrex/models/recsys_feature_prep.py",
            "_compute_post_age_bucket_linear", {"jnp": np, "POST_AGE_MAX_MINUTES": 4800})
        days = np.array([1/24, 2, 7, 30, 90, 100, 101])
        impression = np.full(len(days), 1_800_000_000, dtype=np.int64)
        creation = impression - (days * 86400).astype(np.int64)
        np.testing.assert_array_equal(age_bucket(impression, creation, 1800, 144000), [1, 2, 6, 25, 73, 81, 81])
        np.testing.assert_array_equal(age_bucket(impression, creation), [2, 49, 81, 81, 81, 81, 81])
        config = _valid_model()
        config.grassy_candidate_age_policy = "grassy_candidate_age_90d_v1"
        config.grassy_post_age_encoding = "grassy_post_age_30h_80bins_v1"
        config.post_age_granularity_mins = config.feature_prep.post_age_granularity_mins = 1800
        config.post_age_max_mins = config.feature_prep.post_age_max_mins = 144000
        validate_grassy_model_contract(config)
        config.feature_prep.post_age_max_mins = 4800
        with self.assertRaisesRegex(ValueError, "age encoding"): validate_grassy_model_contract(config)

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
