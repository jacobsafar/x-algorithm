"""Start actual upstream inference with one verified immutable checkpoint. GPU required.

This module is invoked only by the explicit service --serve path. It does not
initialize model weights or implement a surrogate. Patches are narrow admission
hooks around upstream restore/publication, with immutable reload policy.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

from .contracts import load_manifest, require, verify_checkpoint, validate_runtime_model_age_policy, validate_restored_training_state


def install_checkpoint_guards(manifest: dict, manifest_sha: str, checkpoint: Path) -> None:
    from xrex.inference.model_runner import BaseModelRunner
    from xrex.data.recsys.grassy_contract import validate_grassy_model_contract
    original_create = BaseModelRunner.create_state
    original_publish = BaseModelRunner._publish_checkpoint_timestamp

    def create_state(self, ctx):
        require(os.environ.get("DEBUG_ALLOW_RANDOM_INIT") != "1" and
                getattr(ctx, "checkpoint", None) is not None and
                Path(ctx.checkpoint.path).resolve() == checkpoint and
                not self.fake_mm_embeddings and not self.enable_hotswap and not self.copy_url,
                "checkpoint_admission_failed")
        validate_grassy_model_contract(self.model_config)
        validate_runtime_model_age_policy(self.model_config, manifest)
        original_create(self, ctx)
        require(self.elapsed_samples > 0 and Path(ctx.checkpoint.path).resolve() == checkpoint,
                "checkpoint_admission_failed")
        validate_restored_training_state(self.state.step.item(), self.elapsed_samples, manifest)
        self._grassy_verified_checkpoint = str(checkpoint)

    def publish(self, server, timestamp):
        require(getattr(self, "_grassy_verified_checkpoint", None) == str(checkpoint) and
                self.elapsed_samples > 0 and not self.enable_hotswap and timestamp > 0,
                "checkpoint_admission_failed")
        validate_restored_training_state(self.state.step.item(), self.elapsed_samples, manifest)
        original_publish(self, server, timestamp)
        # Upstream omits checkpointPath from individual replies; this admission marker is
        # checked before AND after every request by the private HTTP adapter.
        server.set_serving_prefix(manifest_sha)

    BaseModelRunner.create_state = create_state
    BaseModelRunner._publish_checkpoint_timestamp = publish
    BaseModelRunner._reload_trigger_check_enabled = lambda *args, **kwargs: False
    BaseModelRunner._has_newer_checkpoint = lambda *args, **kwargs: False


def main() -> None:
    require(os.environ.get("DEBUG_ALLOW_RANDOM_INIT", "") not in ("1", "true", "True"))
    manifest_sha = os.environ["PHOENIX_MANIFEST_SHA256"]
    manifest = load_manifest(Path(os.environ["PHOENIX_MANIFEST"]), manifest_sha)
    checkpoint, _ = verify_checkpoint(Path(os.environ["PHOENIX_CHECKPOINT"]), manifest,
                                     Path(os.environ["PHOENIX_CHECKPOINT_INVENTORY"]))
    require(Path(os.environ["MM_LOCAL_SNAPSHOT"]).is_file())
    install_checkpoint_guards(manifest, manifest_sha, checkpoint)
    # No user overrides, synthetic providers, random-init or fake MM switches.
    sys.argv = ["launch_inference", "--driver", "local", "--service_type", "ranking",
        "--config_name", manifest["modelConfigName"], "--checkpoint_path", str(checkpoint),
        "--num_devices_per_process", "1", "--bs_per_device", "1", "--bs_per_device_buckets", "1",
        "--history_seq_len", "1022", "--candidate_seq_len", "64", "--grpc_port", "9988",
        "--metrics_port", "9090", "--max_inflight_requests", "4", "--max_inflight_backend_requests", "1",
        "--allow_random_init", "false", "--fake_mm_embeddings", "false", "--enable_hotswap", "false",
        "--channel_size", "4", "--enqueue_timeout_ms", "50", "--queue_max_staleness_ms", "350"]
    runpy.run_module("xrex.inference.launch_inference", run_name="__main__")


if __name__ == "__main__":
    main()
