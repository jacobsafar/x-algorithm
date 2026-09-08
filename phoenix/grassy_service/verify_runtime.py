"""Validate installed runtime versions without initializing JAX/CUDA."""
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def main():
    for line in Path(__file__).with_name('runtime-requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, pinned = line.split('==')
            if version(name) != pinned:
                raise SystemExit('runtime_dependency_mismatch')
    for module in ('xai_recsys_engine', 'xai_proto', 'xai_checkpointing', 'xrex'):
        if find_spec(module) is None:
            raise SystemExit('runtime_dependency_missing')
    print('{"runtimePackagesVerified":true,"actualCheckpointPredictionPassed":false}')


if __name__ == '__main__':
    main()
