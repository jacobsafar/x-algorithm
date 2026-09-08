"""Default preflight verifies artifacts without importing JAX or allocating GPU resources."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .contracts import ContractError, load_feature_atlas, load_manifest, require, verify_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Explicitly launch the real CUDA runner")
    args = parser.parse_args()
    child = None
    try:
        manifest_sha = os.environ["PHOENIX_MANIFEST_SHA256"]
        manifest = load_manifest(Path(os.environ["PHOENIX_MANIFEST"]), manifest_sha)
        require(os.environ.get("PHOENIX_SOURCE_COMMIT") == manifest["grassyCommit"], "source_mismatch")
        require(os.environ.get("DEBUG_ALLOW_RANDOM_INIT", "") not in ("1", "true", "True"))
        verify_checkpoint(Path(os.environ["PHOENIX_CHECKPOINT"]), manifest,
                          Path(os.environ["PHOENIX_CHECKPOINT_INVENTORY"]))
        atlas = load_feature_atlas(Path(os.environ["PHOENIX_FEATURE_ATLAS"]),
            Path(os.environ["MM_LOCAL_SNAPSHOT"]), Path(os.environ["PHOENIX_SID_SNAPSHOT"]), manifest)
        if not args.serve:
            print('{"artifactPreflightPassed":true,"actualCheckpointPredictionPassed":false,"liveServingEnabled":false}')
            return 0
        from .native import NativePhoenix
        from .server import GoogleIdentityVerifier, serve
        verifier = GoogleIdentityVerifier(os.environ["PHOENIX_IAM_AUDIENCE"], os.environ["PHOENIX_CALLER_SERVICE_ACCOUNT"])
        native = NativePhoenix(manifest, manifest_sha, atlas)
        child = subprocess.Popen([sys.executable, "-m", "grassy_service.runner"], env=os.environ.copy())
        serve(native, verifier, manifest, manifest_sha, atlas, lambda: child.poll() is None)
    except (ContractError, KeyError, OSError):
        print('{"artifactPreflightPassed":false,"error":"service_admission_failed"}', file=sys.stderr)
        return 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
