# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
import typing
import zlib
from typing import Any, NamedTuple

import jax
import numpy as np

from xai_checkpointing.tree_util import tree_to_dict
from xrex.models.model_utils import unwrap_tree
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.sharding_context import make_legacy_sharding_context

if typing.TYPE_CHECKING:
    from xrex.train.trainer_recsys import RecsysTrainer

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_DIR = "export"
MANIFEST_NAME = f"{BUNDLE_DIR}/MANIFEST.json"


def restamp_manifest(data: bytes) -> bytes:
    manifest = json.loads(data)
    manifest["created_timestamp"] = time.time()
    return json.dumps(manifest, indent=2).encode()


class EmbeddingSlices(NamedTuple):
    hist_post_end: int
    hist_auth_end: int
    cand_post_end: int
    cand_auth_end: int
    user_end: int
    user_ip_end: int


class PackedGeometry(NamedTuple):
    packed_history_len: int
    packed_candidate_len: int
    bs_per_device: int
    merged_batch: int


@dataclasses.dataclass(frozen=True)
class BundleFile:
    name: str
    data: bytes


_parameter_serialization_registered = False


def _pspec_to_json(pspec: Any) -> list[Any]:
    return [None if e is None else (e if isinstance(e, str) else list(e)) for e in tuple(pspec)]


def _pspec_from_json(entries: list[Any]) -> Any:
    from jax.sharding import PartitionSpec

    return PartitionSpec(
        *[None if e is None else (e if isinstance(e, str) else tuple(e)) for e in entries]
    )


def ensure_parameter_serialization_registered() -> None:
    global _parameter_serialization_registered
    if _parameter_serialization_registered:
        return
    from jax import export as jax_export

    from xrex.models.model_utils import Parameter

    def serialize_auxdata(aux: dict[str, Any]) -> bytes:
        aux = dict(aux)
        aux["pspec"] = _pspec_to_json(aux["pspec"])
        return json.dumps(aux, default=list).encode()

    def deserialize_auxdata(data: bytes) -> dict[str, Any]:
        aux = json.loads(data)
        aux["pspec"] = _pspec_from_json(aux["pspec"])
        if isinstance(aux.get("rms_clip_axes"), list):
            aux["rms_clip_axes"] = tuple(aux["rms_clip_axes"])
        return aux

    jax_export.register_pytree_node_serialization(
        Parameter,
        serialized_name="xrex.models.model_utils.Parameter",
        serialize_auxdata=serialize_auxdata,
        deserialize_auxdata=deserialize_auxdata,
    )
    _parameter_serialization_registered = True


_packing_layout_serialization_registered = False


def ensure_packing_layout_serialization_registered() -> None:
    global _packing_layout_serialization_registered
    if _packing_layout_serialization_registered:
        return
    from jax import export as jax_export

    from xrex.data.recsys.sequence_packing import SequencePackedLayout

    jax_export.register_pytree_node_serialization(
        SequencePackedLayout,
        serialized_name="xrex.data.recsys.sequence_packing.SequencePackedLayout",
        serialize_auxdata=lambda aux: json.dumps(list(aux)).encode(),
        deserialize_auxdata=lambda data: tuple(json.loads(data)),
    )
    _packing_layout_serialization_registered = True


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def _aval_entry(leaf: Any) -> dict[str, Any]:
    return {"shape": [int(d) for d in leaf.shape], "dtype": _dtype_name(leaf.dtype)}


def _to_shape_dtype_struct(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, jax.dtypes.canonicalize_dtype(leaf.dtype)),
        tree,
    )


def _make_export_config(trainer: RecsysTrainer, history_seq_len: int, candidate_seq_len: int):
    from xrex.configs.config_loader import replace_cli_subs

    init_params = trainer.to_dict()
    init_params.pop("__class")
    export_cfg = type(trainer).from_dict(init_params, ensure_class=type(trainer))

    overrides = [
        "num_devices_per_process=1",
        "ep=1",
        "dp=1",
        "num_negatives_per_example=0",
        "num_global_negatives_per_example=0",
        f"history_seq_len={history_seq_len}",
        f"candidate_seq_len={candidate_seq_len}",
    ]
    export_cfg, used = replace_cli_subs(export_cfg, overrides)
    unused = [k for k, v in used.items() if not v]
    if unused:
        raise ValueError(f"StableHLO bundle export overrides did not match config: {unused}")
    return export_cfg


def _batch_avals(export_cfg: Any, bs: int, *, packed: bool) -> Any:
    model_config = export_cfg.model_config
    batch = export_cfg.dataset.example_data(bs)

    if packed:
        from xrex.data.recsys.sequence_packing import pack_batch

        batch = pack_batch(
            batch=batch,
            num_devices_per_process=export_cfg.parallel_config.num_devices_per_process,
            num_user_prefix_tokens=model_config.num_user_prefix_tokens,
            dist=None,
            rng=None,
            block_size=export_cfg._seqpack_block_size,
        )

    batch = _to_shape_dtype_struct(batch)

    if model_config.multimodal_embedding_type is not None:
        cand_post = batch["candidate_seq"]["post_hashes"]
        batch["candidate_seq"]["embedding"] = jax.ShapeDtypeStruct(
            (cand_post.shape[0], cand_post.shape[1], model_config.multimodal_embedding_dim),
            np.float32,
        )

    if model_config.use_post_sid and model_config.sid_num_levels > 0:
        for seq_name in ("history_seq", "candidate_seq"):
            post_hashes = batch[seq_name]["post_hashes"]
            batch[seq_name]["post_sids"] = jax.ShapeDtypeStruct(
                (post_hashes.shape[0], post_hashes.shape[1], model_config.sid_num_levels),
                np.uint16,
            )

    return batch


def _embedding_slices(export_cfg: Any, history_seq_len: int, candidate_seq_len: int):
    ht = export_cfg.dataset.hash_table
    hist_post_seq = ht.num_item_hashes * history_seq_len
    hist_auth_seq = ht.num_author_hashes * history_seq_len
    cand_post_seq = ht.num_item_hashes * candidate_seq_len
    cand_auth_seq = ht.num_author_hashes * candidate_seq_len
    user_seq = ht.num_user_hashes
    ip_seq = ht.num_ip_hashes if export_cfg.model_config.use_ip_address else 0

    user_end = hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq
    return EmbeddingSlices(
        hist_post_end=hist_post_seq,
        hist_auth_end=hist_post_seq + hist_auth_seq,
        cand_post_end=hist_post_seq + hist_auth_seq + cand_post_seq,
        cand_auth_end=hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq,
        user_end=user_end,
        user_ip_end=user_end + ip_seq,
    )


def _packed_embedding_slices(
    export_cfg: Any, batch_avals: Any
) -> tuple[EmbeddingSlices, PackedGeometry]:
    hist = batch_avals["history_seq"]
    cand = batch_avals["candidate_seq"]
    hist_post_seq = int(np.prod(hist["post_hashes"].shape[1:]))
    hist_auth_seq = int(np.prod(hist["auth_hashes"].shape[1:]))
    cand_post_seq = int(np.prod(cand["post_hashes"].shape[1:]))
    cand_auth_seq = int(np.prod(cand["auth_hashes"].shape[1:]))
    user_seq = int(np.prod(batch_avals["user_hashes"].shape[1:]))
    ip_seq = (
        int(np.prod(batch_avals["user_ip_hashes"].shape[1:]))
        if export_cfg.model_config.use_ip_address
        else 0
    )

    user_end = hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq
    slices = EmbeddingSlices(
        hist_post_end=hist_post_seq,
        hist_auth_end=hist_post_seq + hist_auth_seq,
        cand_post_end=hist_post_seq + hist_auth_seq + cand_post_seq,
        cand_auth_end=hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq,
        user_end=user_end,
        user_ip_end=user_end + ip_seq,
    )
    geometry = PackedGeometry(
        packed_history_len=int(hist["post_hashes"].shape[1]),
        packed_candidate_len=int(cand["post_hashes"].shape[1]),
        bs_per_device=int(batch_avals["user_hashes"].shape[1]),
        merged_batch=int(batch_avals["user_hashes"].shape[0]),
    )
    return slices, geometry


def _make_forward_fn(
    export_cfg: Any,
    embedding_slices: EmbeddingSlices,
    mesh: jax.sharding.Mesh,
    packed_geometry: PackedGeometry | None = None,
):
    import haiku as hk
    import jax.numpy as jnp

    model_config = export_cfg.model_config

    def _packed_recsys_embeddings(merged: jax.Array, sl: EmbeddingSlices) -> RecsysEmbeddings:
        assert packed_geometry is not None
        g = packed_geometry

        def _section(start: int, end: int, rows: int) -> jax.Array:
            return merged[:, start:end, :].reshape(merged.shape[0], rows, -1, merged.shape[-1])

        return RecsysEmbeddings(
            history_post_embeddings=_section(0, sl.hist_post_end, g.packed_history_len),
            history_author_embeddings=_section(
                sl.hist_post_end, sl.hist_auth_end, g.packed_history_len
            ),
            candidate_post_embeddings=_section(
                sl.hist_auth_end, sl.cand_post_end, g.packed_candidate_len
            ),
            candidate_author_embeddings=_section(
                sl.cand_post_end, sl.cand_auth_end, g.packed_candidate_len
            ),
            user_embeddings=_section(sl.cand_auth_end, sl.user_end, g.bs_per_device),
            user_ip_embeddings=(
                _section(sl.user_end, sl.user_ip_end, g.bs_per_device)
                if sl.user_ip_end > sl.user_end
                else None
            ),
        )

    @hk.transform
    def forward_fn(batch: Any, merged_embeddings: jax.Array):
        sl = embedding_slices
        if packed_geometry is not None:
            recsys_embeddings = _packed_recsys_embeddings(merged_embeddings, sl)
        else:
            recsys_embeddings = RecsysEmbeddings(
                history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :],
                history_author_embeddings=merged_embeddings[
                    :, sl.hist_post_end : sl.hist_auth_end, :
                ],
                candidate_post_embeddings=merged_embeddings[
                    :, sl.hist_auth_end : sl.cand_post_end, :
                ],
                candidate_author_embeddings=merged_embeddings[
                    :, sl.cand_post_end : sl.cand_auth_end, :
                ],
                user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :],
                user_ip_embeddings=(
                    merged_embeddings[:, sl.user_end : sl.user_ip_end, :]
                    if sl.user_ip_end > sl.user_end
                    else None
                ),
            )
        model = model_config.make(sharding_context=make_legacy_sharding_context(mesh))
        logits, candidate_continuous_predictions = model.forward(batch, recsys_embeddings)
        log_probs = jax.nn.log_sigmoid(logits).astype(jnp.bfloat16).astype(jnp.float32)
        cont_preds = candidate_continuous_predictions.astype(jnp.bfloat16).astype(jnp.float32)
        has_nan = jnp.any(jnp.isnan(log_probs), axis=tuple(range(1, log_probs.ndim)))
        return log_probs, cont_preds, has_nan

    return forward_fn


def _scan_custom_call_targets(lowered_text: str) -> list[str]:
    targets = set(re.findall(r"stablehlo\.custom_call\s*@([\w.$-]+)", lowered_text))
    targets |= set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', lowered_text))
    return sorted(targets)


def _input_spec(
    params_avals: Any, rng_aval: Any, batch_avals: Any, merged_aval: Any
) -> list[dict[str, Any]]:
    from xai_checkpointing.tree_util import keystr

    spec: list[dict[str, Any]] = []

    param_leaves = jax.tree.leaves(params_avals)
    param_keys = list(tree_to_dict(unwrap_tree(params_avals), keep_none=False).keys())
    if len(param_keys) != len(param_leaves):
        raise AssertionError(
            f"params flatten mismatch: {len(param_keys)} keys vs {len(param_leaves)} leaves"
        )
    for key, leaf in zip(param_keys, param_leaves):
        spec.append({"kind": "weight", "key": f"params.{key}", **_aval_entry(leaf)})

    spec.append({"kind": "rng", "key": "rng", **_aval_entry(rng_aval)})

    for path, leaf in jax.tree_util.tree_flatten_with_path(batch_avals)[0]:
        key = keystr(path)
        kind = "packing_layout" if key.startswith("packing_layout.") else "batch"
        spec.append({"kind": kind, "key": key, **_aval_entry(leaf)})

    spec.append(
        {"kind": "merged_embeddings", "key": "merged_embeddings", **_aval_entry(merged_aval)}
    )
    return spec


def build_bundle(trainer: RecsysTrainer) -> list[BundleFile]:
    import flatbuffers
    from jax import export as jax_export

    from xrex.models.recsys_gen_recs_model import RecsysGenRecsModelConfig
    from xrex.models.recsys_model import RecsysAggregatedModelConfig

    model_config = trainer.model_config
    if not isinstance(model_config, RecsysAggregatedModelConfig) or isinstance(
        model_config, RecsysGenRecsModelConfig
    ):
        raise NotImplementedError(
            "StableHLO bundle export supports ranking (RecsysAggregatedModelConfig) only, "
            f"got {type(model_config).__name__}"
        )
    if trainer.using_seqpack and trainer.using_fa4:
        raise NotImplementedError(
            "StableHLO bundle export supports seqpack with pallas_ranker_varlen_attn only; "
            "FA4 (cutedsl_ranker_varlen_attn) block-sparse layouts are not supported yet"
        )
    if not trainer.checkpoint_config.copy_port:
        raise ValueError("export_stablehlo_bundle requires checkpoint_config.copy_port")

    buckets = sorted(
        {int(b) for b in str(trainer.export_bundle_bs_per_device).split(",") if b.strip()}
    )
    if not buckets or buckets[0] < 1:
        raise ValueError(
            f"invalid export_bundle_bs_per_device: {trainer.export_bundle_bs_per_device!r}"
        )

    history_seq_len = trainer.export_bundle_history_seq_len or trainer.dataset.history_seq_len
    candidate_seq_len = trainer.export_bundle_candidate_seq_len or trainer.dataset.candidate_seq_len

    if trainer.using_seqpack:
        block = int(trainer._seqpack_block_size)
        prefix = int(model_config.num_user_prefix_tokens)
        total = prefix + candidate_seq_len + history_seq_len
        if total % block:
            raise ValueError(
                f"seqpack export needs (num_user_prefix_tokens + candidate_seq_len + "
                f"history_seq_len) to be a multiple of the attention block size "
                f"{block}, got {prefix} + {candidate_seq_len} + {history_seq_len} = {total}; "
                "set export_bundle_candidate_seq_len / export_bundle_history_seq_len "
                "to block-aligned serving lengths"
            )

    ensure_parameter_serialization_registered()
    if trainer.using_seqpack:
        ensure_packing_layout_serialization_registered()
    export_cfg = _make_export_config(trainer, history_seq_len, candidate_seq_len)

    axis_names = export_cfg.parallel_config.mesh_axis_names()
    mesh_shape = export_cfg.parallel_config.mesh_shape()
    if int(np.prod(mesh_shape)) != 1:
        raise AssertionError(f"export mesh must be single-device, got {mesh_shape}")
    device = jax.local_devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]).reshape(mesh_shape), axis_names)

    packed = bool(trainer.using_seqpack)
    dense_slices = (
        None if packed else _embedding_slices(export_cfg, history_seq_len, candidate_seq_len)
    )

    params_avals = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), trainer.state_shape.params
    )
    rng_aval = jax.ShapeDtypeStruct((2,), np.uint32)

    files: list[BundleFile] = []
    programs: dict[str, Any] = {}
    all_custom_call_targets: set[str] = set()

    for bs in buckets:
        start = time.perf_counter()
        batch_avals = _batch_avals(export_cfg, bs, packed=packed)
        if packed:
            embedding_slices, packed_geometry = _packed_embedding_slices(export_cfg, batch_avals)
            merged_batch = packed_geometry.merged_batch
        else:
            assert dense_slices is not None
            embedding_slices, packed_geometry = dense_slices, None
            merged_batch = bs
        forward_fn = _make_forward_fn(export_cfg, embedding_slices, mesh, packed_geometry)
        merged_aval = jax.ShapeDtypeStruct(
            (merged_batch, embedding_slices.user_ip_end, model_config.emb_table_width),
            model_config.embedding_dtype,
        )
        args = (params_avals, rng_aval, batch_avals, merged_aval)

        with mesh:
            jitted = jax.jit(forward_fn.apply)
            lowered_text = jitted.lower(*args).as_text(dialect="stablehlo")
            custom_call_targets = _scan_custom_call_targets(lowered_text)
            exported = jax_export.export(
                jitted,
                platforms=("cuda",),
                disabled_checks=[
                    jax_export.DisabledSafetyCheck.custom_call(t) for t in custom_call_targets
                ],
            )(*args)

        spec = _input_spec(params_avals, rng_aval, batch_avals, merged_aval)
        if len(spec) != len(exported.in_avals):
            raise AssertionError(
                f"input spec mismatch for bs={bs}: {len(spec)} != {len(exported.in_avals)}"
            )
        for entry, aval in zip(spec, exported.in_avals):
            expected = _aval_entry(aval)
            if entry["shape"] != expected["shape"] or entry["dtype"] != expected["dtype"]:
                raise AssertionError(
                    f"input spec mismatch for bs={bs} {entry['key']}: {entry} vs {expected}"
                )

        mlir_name = f"{BUNDLE_DIR}/forward_bs{bs}.mlirbc"
        jax_export_name = f"{BUNDLE_DIR}/forward_bs{bs}.jax_export"
        mlir_bytes = bytes(exported.mlir_module_serialized)
        jax_export_bytes = bytes(exported.serialize())
        files.append(BundleFile(mlir_name, mlir_bytes))
        files.append(BundleFile(jax_export_name, jax_export_bytes))
        all_custom_call_targets.update(custom_call_targets)

        outputs = [
            {"name": name, **_aval_entry(aval)}
            for name, aval in zip(("log_probs", "cont_preds", "has_nan"), exported.out_avals)
        ]
        programs[str(bs)] = {
            "batch_size": bs,
            "mlir_module": {"file": mlir_name, "adler32": zlib.adler32(mlir_bytes)},
            "jax_export": {"file": jax_export_name, "adler32": zlib.adler32(jax_export_bytes)},
            "calling_convention_version": int(exported.calling_convention_version),
            "module_kept_var_idx": [int(i) for i in exported.module_kept_var_idx],
            "custom_call_targets": custom_call_targets,
            "input_spec": spec,
            "output_spec": outputs,
            "seqpack": (
                {**packed_geometry._asdict(), "merged_slices": embedding_slices._asdict()}
                if packed_geometry is not None
                else None
            ),
        }
        logger.info(
            "Exported StableHLO ranking forward bs=%d in %.1fs (%d inputs, %d kept, "
            "custom_calls=%s)",
            bs,
            time.perf_counter() - start,
            len(spec),
            len(exported.module_kept_var_idx),
            custom_call_targets,
        )

    manifest = _build_manifest(
        trainer,
        export_cfg,
        device,
        programs,
        dense_slices,
        history_seq_len,
        candidate_seq_len,
        sorted(all_custom_call_targets),
        packed=packed,
    )
    files.insert(0, BundleFile(MANIFEST_NAME, json.dumps(manifest, indent=2).encode()))
    return files


def _leaf_width(tree: Any, key: str, axis: int) -> int:
    leaf = tree.get(key)
    if leaf is None or len(leaf.shape) <= axis:
        return 0
    return int(leaf.shape[axis])


def _prep_spec(
    export_cfg: Any, history_seq_len: int, candidate_seq_len: int, *, packed: bool
) -> dict[str, Any]:
    dataset = export_cfg.dataset
    model_config = export_cfg.model_config
    ht = dataset.hash_table
    hk = ht.hash_keys
    batch = _batch_avals(export_cfg, 1, packed=False)
    hist = batch["history_seq"]
    cand = batch["candidate_seq"]

    return {
        "user_id_table_size": int(ht.user_id_table_size),
        "user_hash_scales": [int(x) for x in hk.user_hash_scales],
        "user_biases": [int(x) for x in hk.user_biases],
        "user_modulus": int(hk.user_modulus),
        "item_id_table_size": int(ht.item_id_table_size),
        "item_hash_vocab_size": int(getattr(hk, "item_hash_vocab_size", 0) or 0),
        "item_hash_scales": [int(x) for x in hk.item_hash_scales],
        "item_biases": [int(x) for x in hk.item_biases],
        "item_modulus": int(hk.item_modulus),
        "author_id_table_size": int(ht.author_id_table_size),
        "author_hash_scales": [int(x) for x in hk.author_hash_scales],
        "author_biases": [int(x) for x in hk.author_biases],
        "author_modulus": int(hk.author_modulus),
        "ip_id_table_size": int(ht.ip_id_table_size),
        "ip_hash_scales": [int(x) for x in hk.ip_hash_scales],
        "ip_biases": [int(x) for x in hk.ip_biases],
        "ip_modulus": int(hk.ip_modulus),
        "output_vocab_size": int(dataset.output_vocab_size),
        "num_continuous_actions": int(model_config.num_continuous_actions),
        "search_query_embedding_dim": _leaf_width(cand, "search_query_embeddings", 2),
        "num_user_categorical_features": _leaf_width(batch, "user_categorical_features", 1),
        "num_user_bool_features": _leaf_width(batch, "user_bool_features", 1),
        "num_user_float_features": _leaf_width(batch, "user_float_features", 1),
        "num_user_int64_features": _leaf_width(batch, "user_int64_features", 1),
        "num_user_installed_apps": _leaf_width(batch, "user_installed_apps_multihot", 1),
        "num_post_categorical_features": _leaf_width(hist, "categorical_features", 2),
        "num_post_bool_features": _leaf_width(hist, "bool_features", 2),
        "num_post_float_features": _leaf_width(hist, "float_features", 2),
        "num_post_int64_features": _leaf_width(hist, "int64_features", 2),
        "enable_stale_post": bool(
            getattr(getattr(model_config, "feature_prep", None), "enable_stale_post", False)
        ),
        "history_seq_len": history_seq_len,
        "candidate_seq_len": candidate_seq_len,
        "sid_num_levels": _leaf_width(hist, "post_sids", 2),
        "multimodal_embedding_dim": _leaf_width(cand, "embedding", 2),
        "num_categorical_features": 0,
        "use_ip": bool(model_config.use_ip_address),
        "use_seqpack": packed,
        "seqpack_block_size": int(export_cfg._seqpack_block_size) if packed else 0,
        "num_user_prefix_tokens": (int(model_config.num_user_prefix_tokens) if packed else 0),
        "transformer_candidate_seq_len": (
            (0 if cand.get("post_ids") is not None else candidate_seq_len) if packed else 0
        ),
    }


def _build_manifest(
    trainer: RecsysTrainer,
    export_cfg: Any,
    device: Any,
    programs: dict[str, Any],
    dense_slices: EmbeddingSlices | None,
    history_seq_len: int,
    candidate_seq_len: int,
    custom_call_targets: list[str],
    *,
    packed: bool,
) -> dict[str, Any]:
    import jaxlib

    model_config = export_cfg.model_config
    dataset = export_cfg.dataset

    ep = int(trainer.mesh.shape["expert"])
    emb_rows = int(trainer.state_shape.emb_table.x.shape[0])
    emb_rows_padded = -(-emb_rows // ep) * ep

    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": "recsys_ranking_forward",
        "name": trainer.name,
        "model_config_class": type(trainer.model_config).__name__,
        "created_timestamp": time.time(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platforms": ["cuda"],
        "device_kind": str(device.device_kind),
        "compute_capability": str(getattr(device, "compute_capability", "")),
        "custom_call_targets": custom_call_targets,
        "history_seq_len": history_seq_len,
        "candidate_seq_len": candidate_seq_len,
        "output_vocab_size": int(dataset.output_vocab_size),
        "num_continuous_actions": int(model_config.num_continuous_actions),
        "use_seqpack": packed,
        "prep_spec": _prep_spec(export_cfg, history_seq_len, candidate_seq_len, packed=packed),
        "programs": programs,
        "embedding": {
            "table_key": "emb_table",
            "rows": emb_rows,
            "rows_padded": emb_rows_padded,
            "num_shards": ep,
            "width": int(model_config.emb_table_width),
            "dtype": _dtype_name(model_config.embedding_dtype),
            "merged_slices": dense_slices._asdict() if dense_slices is not None else None,
            "hash_table": dataset.hash_table.to_dict(),
        },
    }
