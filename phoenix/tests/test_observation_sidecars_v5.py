# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures exercise sealed v5 structure and semantic rejection independently."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from test_observation_sidecars import _Artifact, _sha256, _write_json
from xrex.data.recsys.observation_sidecars import GRASSY_DATASET_MANIFEST, ObservationSidecarError
from xrex.data.recsys.observation_sidecars_v5 import AUTHORITY, COLLECTIONS, MODEL_GATES, SOURCE_CONTRACT
from xrex.data.recsys.grassy_contract import GRASSY_RUNTIME_UPSTREAM_COMMIT


class ArtifactV5(_Artifact):
    def __init__(self, root):
        self.is_v5 = False
        super().__init__(root)
        self.is_v5 = True
        table = pq.ParquetFile(self.base).read()
        stamp = 1_786_500_000_000
        created = stamp - 3_600_000
        post = ((created - 1288834974657) << 22) | 21
        for index, (name, values, dtype) in enumerate([
            ('length', [1] * self.rows, pa.int32()),
            ('impressedTimeMsSeq', [[stamp] + [0] * 1085] * self.rows, pa.list_(pa.int64(), 1086)),
            ('tweetIdSeq', [[post] + [0] * 1085] * self.rows, pa.list_(pa.int64(), 1086)),
            ('authorIdSeq', [[22] + [0] * 1085] * self.rows, pa.list_(pa.int64(), 1086)),
            ('paddingMask', [[True] + [False] * 1085] * self.rows, pa.list_(pa.bool_(), 1086)),
        ]):
            table = table.set_column(index + 2, name, pa.array(values, type=dtype))
        pq.write_table(table, self.base)
        action = np.zeros((self.rows, 1086, 64), dtype=np.bool_)
        action[:, 0, 13] = True
        np.save(self.action, action)
        self.identity_values = [dict(exampleId=f'{n+1:064x}', userId=f'{n+101:064x}', split='train',
            rolloutReleaseId='release-1', protectedReleaseId=None, platform='ios', appVersion='1.0',
            surface='for_you') for n in range(self.rows)]
        self.generation_values = [dict(modelPostId=str(post), generationId='c'*64, modelAuthorId='22',
            createdAtMs=created, createTimeSeconds=created//1000, createTimeNanoseconds=0)]
        self.registry = dict(contract='grassy_hmac_identity_user_split_v1', projectId='fixture', keyId='fixture-key',
            keySha256='b'*64, splitNamespace='global_user_8000_1000_1000_v1')
        self.sidecar_value.update(manifestVersion=2, exporterVersion=3,
            upstreamCommit=GRASSY_RUNTIME_UPSTREAM_COMMIT, split='train', observationAuthority=AUTHORITY,
            identityRegistry=self.registry, protectedActionRollouts=[])
        self.sidecar_value['rowIdentity']['encoding'] = 'jsonl-hmac-sha256-user-split-producer-v1'
        self.source = dict(contract='grassy_feed_training_snapshot_v1',complete=True,trainingEligible=False,
            snapshotId='fixture',outcomeCutoffMs=self.sidecar_value['outcomeCutoffMs'],identity=self.registry,
            rollout={'sha256':'d'*64,'evidenceOrigins':[{'origin':{'sha256':'e'*64}}]}, objects={})
        for kind in ('raw','selected'):
            for collection in COLLECTIONS:
                self.source['objects'][f'{kind}/{collection}.jsonl']={'generation':'1','sha256':'a'*64}
        for name in ('ObservabilityRollouts.json','rollout-attestation.json','selection-report.json',
                     'user-split-assignments.jsonl','exposure-identities.jsonl'):
            self.source['objects'][name]={'generation':'1','sha256':'a'*64}
        self.rewrite_manifests()

    def rewrite_manifests(self):
        if not self.is_v5:
            return super().rewrite_manifests()
        self.identities.write_text(''.join(json.dumps(v)+'\n' for v in self.identity_values))
        generation_path=self.root/'post-generation-identities.jsonl'
        generation_path.write_text(''.join(json.dumps(v)+'\n' for v in self.generation_values))
        binding={'path':generation_path.name,'sha256':_sha256(generation_path),
            'encoding':'jsonl-model-post-full-generation-v1','rows':len(self.generation_values)}
        self.sidecar_value['postGenerationIdentities']=binding
        for key,path in (('base',self.base),('actionObservation',self.action),
                         ('continuousActionObservation',self.continuous),('rowIdentity',self.identities)):
            self.sidecar_value[key]['sha256']=_sha256(path)
        super().rewrite_manifests()
        manifest_path=self.root/GRASSY_DATASET_MANIFEST
        manifest=json.loads(manifest_path.read_text())
        source_path=self.root/'source-snapshot-manifest.json'
        _write_json(source_path,self.source)
        manifest.update(manifest_version=5,source_contract=SOURCE_CONTRACT,
            upstream_commit=GRASSY_RUNTIME_UPSTREAM_COMMIT,required_model_gates=MODEL_GATES,
            observation_authority=AUTHORITY,candidate_age_policy='legacy_48h_v1',split='train',
            identity_registry=self.registry,post_generation_identities=binding,
            source_snapshot={'path':source_path.name,'sha256':_sha256(source_path),'generation':'1','snapshotId':'fixture'},
            source_snapshots={c:self.source['objects'][f'selected/{c}.jsonl'] for c in COLLECTIONS},
            observability_rollout_sha256='d'*64)
        manifest['exporter'].update(name='grassy-feed-eight-source-to-phoenix',version=3,input_schema_version=2,
            legacy_validator_sha256='b'*64,identity_mapper_sha256='b'*64)
        _write_json(manifest_path,manifest)


class ObservationSidecarV5Test(unittest.TestCase):
    def test_valid_v5_loads_without_changing_base(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact=ArtifactV5(Path(directory)); digest=_sha256(artifact.base)
            result=artifact.load()
            self.assertEqual(result.action_mask.shape,(4,1086,64))
            self.assertEqual(_sha256(artifact.base),digest)

    def test_rehashed_old_cohort_cannot_gain_protected_head(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact=ArtifactV5(Path(directory)); action=np.load(artifact.action)
            action[0,0,1]=True;np.save(artifact.action,action);artifact.rewrite_manifests()
            with self.assertRaisesRegex(ObservationSidecarError,'old cohort'):
                artifact.load()

    def test_split_and_rollout_time_are_semantic_not_only_hash_checks(self):
        for mutation in ('split','time','generation'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                artifact=ArtifactV5(Path(directory))
                if mutation=='split': artifact.identity_values[0]['split']='test'
                if mutation=='time': artifact.sidecar_value['rolloutsUsed'][0]['effectiveFromMs']=1_786_600_000_000
                if mutation=='generation': artifact.generation_values[0]['createTimeNanoseconds']=1_000_000
                artifact.rewrite_manifests()
                with self.assertRaises(ObservationSidecarError): artifact.load()

    def test_raw_original_authority_and_identity_registry_are_required(self):
        for mutation in ('original','key'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                artifact=ArtifactV5(Path(directory))
                if mutation=='original': del artifact.source['objects']['raw/FeedPositiveActionOperations.jsonl']
                if mutation=='key': artifact.source['identity']={**artifact.registry,'keyId':'other'}
                artifact.rewrite_manifests()
                with self.assertRaises(ObservationSidecarError): artifact.load()

    def test_valid_protected_cohort_and_unattested_evidence(self):
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as directory:
                artifact=ArtifactV5(Path(directory))
                rollout=artifact.sidecar_value['rolloutsUsed'][0]
                producer={**rollout,'releaseId':'protected-1','producerAuthority':AUTHORITY['protectedActionAuthority'],
                    'producerSchemaVersion':1,'discreteOrdinals':[1], 'evidenceSha256':['f'*64 if corrupt else 'e'*64]}
                artifact.sidecar_value['protectedActionRollouts']=[producer]
                artifact.identity_values[0]['protectedReleaseId']='protected-1'
                action=np.load(artifact.action);action[0,0,1]=True;np.save(artifact.action,action)
                artifact.rewrite_manifests()
                if corrupt:
                    with self.assertRaisesRegex(ObservationSidecarError,'evidence'): artifact.load()
                else: self.assertTrue(artifact.load().action_mask[0,0,1])

if __name__=='__main__': unittest.main()
