# SPDX-License-Identifier: Apache-2.0
"""CPU prospective policy admission, independent of legacy artifact parsing."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from test_observation_sidecars import _sha256, _write_json
from test_observation_sidecars_v5 import ArtifactV5
from xrex.data.recsys.observation_sidecars import GRASSY_DATASET_MANIFEST, ObservationSidecarError
from xrex.data.recsys.observation_sidecars_v6 import AUTHORITY, COLLECTIONS, MODEL_GATES, SOURCE_CONTRACT, expected_model_policy


class ArtifactV6(ArtifactV5):
    def __init__(self,root,render='grassy_text_only_v1'):
        self.is_v6=False
        super().__init__(root)
        self.is_v6=True
        self.capture=dict(featureSchemaVersion=4,candidateAgePolicy='grassy_candidate_age_90d_v1',renderPolicy=render)
        self.source.update(contract='grassy_feed_training_snapshot_v2',capturePolicy=self.capture)
        for kind in ('raw','selected'):
            for collection in COLLECTIONS:
                self.source['objects'][f'{kind}/{collection}.jsonl']={'generation':'1','sha256':'a'*64}
        for generation in self.generation_values:
            generation.update(renderPolicy=render,encoderInputSha256='f'*64,contentSnapshotId='e'*64,contentSha256='d'*64,
                contentReferences=[dict(contentSnapshotId='e'*64,contentSha256='d'*64,
                    sourceUpdateTimeSeconds=generation['createTimeSeconds'],sourceUpdateTimeNanoseconds=generation['createTimeNanoseconds'])])
        self.rewrite_manifests()

    def rewrite_manifests(self):
        if not self.is_v6: return super().rewrite_manifests()
        self.sidecar_value.update(manifestVersion=3,exporterVersion=4,observationAuthority=AUTHORITY)
        super().rewrite_manifests()
        path=self.root/GRASSY_DATASET_MANIFEST;manifest=json.loads(path.read_text())
        manifest.update(manifest_version=6,source_contract=SOURCE_CONTRACT,observation_authority=AUTHORITY,
            required_model_gates=MODEL_GATES,candidate_age_policy='grassy_candidate_age_90d_v1',capture_policy=self.capture,
            model_policy=expected_model_policy(self.capture['renderPolicy']),age_encoding='grassy_post_age_30h_80bins_v1',
            source_snapshots={c:self.source['objects'][f'selected/{c}.jsonl'] for c in COLLECTIONS})
        manifest['exporter'].update(name='grassy-feed-ten-source-to-phoenix',version=4,
            prospective_source_sha256='b'*64,authority_validator_sha256='b'*64)
        _write_json(path,manifest)


class ObservationSidecarV6Test(unittest.TestCase):
    def test_explicit_text_and_caption_policies_load_same_base_geometry(self):
        for render in ('grassy_text_only_v1','grassy_caption_only_v1'):
            with self.subTest(render=render),tempfile.TemporaryDirectory() as temp:
                artifact=ArtifactV6(Path(temp),render);before=_sha256(artifact.base)
                self.assertEqual(artifact.load().action_mask.shape,(4,1086,64))
                self.assertEqual(_sha256(artifact.base),before)

    def test_rehashed_policy_missing_content_and_reference_mismatch_fail(self):
        for mutation in ('age','source','representative','references','render'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                artifact=ArtifactV6(Path(temp))
                if mutation=='source':del artifact.source['objects']['raw/FeedServedContentSnapshots.jsonl']
                if mutation=='representative':artifact.generation_values[0]['contentSnapshotId']='a'*64
                if mutation=='references':artifact.generation_values[0]['contentReferences']*=2
                if mutation=='render':artifact.generation_values[0]['renderPolicy']='grassy_caption_only_v1'
                artifact.rewrite_manifests()
                if mutation=='age':
                    path=artifact.root/GRASSY_DATASET_MANIFEST;data=json.loads(path.read_text());data['model_policy']['postAgeMaxMins']=4800;_write_json(path,data)
                with self.assertRaises(ObservationSidecarError):artifact.load()

if __name__=='__main__':unittest.main()
