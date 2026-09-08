# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU admission fixtures; none of these values are production models."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from test_observation_sidecars import _sha256, _write_json
from test_observation_sidecars_v5 import ArtifactV5
from xrex.data.recsys.observation_sidecars import GRASSY_DATASET_MANIFEST, ObservationSidecarError
from xrex.data.recsys.grassy_representations import CONTRACT, load_grassy_representations
from xrex.data.recsys.grassy_contract import GRASSY_RUNTIME_UPSTREAM_COMMIT


class RepresentationFixture:
    def __init__(self, root, artifact_class=ArtifactV5):
        self.artifact=artifact_class(root/'train')
        self.root=root/'representations';self.root.mkdir()
        dataset_path=self.artifact.root/GRASSY_DATASET_MANIFEST
        dataset=json.loads(dataset_path.read_text())
        index=dict(sourceContract=dataset['source_contract'],sourceSnapshotSha256=dataset['source_snapshot']['sha256'],
            identityRegistry=self.artifact.registry, splits={
            'train':dict(rows=4,users=2,manifest='train/'+GRASSY_DATASET_MANIFEST,sha256=_sha256(dataset_path)),
            'validation':dict(rows=0,users=0,manifest=None,sha256=None),
            'test':dict(rows=0,users=0,manifest=None,sha256=None)})
        if dataset['source_contract']=='grassy_phoenix_dataset_v6':
            index.update(capturePolicy=dataset['capture_policy'],model_policy=dataset['model_policy'])
        _write_json(root/'dataset-index.json',index)
        source=self.artifact.generation_values[0]
        embedding=np.zeros((1,1024),dtype=np.float32);embedding[0,0]=1
        np.save(self.root/'embeddings.npy',embedding)
        np.save(self.root/'post_ids.npy',np.array([int(source['modelPostId'])],dtype=np.int64))
        np.save(self.root/'codebook.npy',np.zeros((6,256,1024),dtype=np.float32))
        _write_json(self.root/'train-membership.json',[source['modelPostId']])
        self.record={**{k:source[k] for k in ('modelPostId','generationId','modelAuthorId','createdAtMs','createTimeSeconds','createTimeNanoseconds')},'contentSha256':'d'*64,'embeddingSha256':hashlib.sha256(embedding[0].tobytes()).hexdigest(),
            'sourceUpdateTimeSeconds':source['createTimeSeconds'],'sourceUpdateTimeNanoseconds':source['createTimeNanoseconds'],
            'semanticIds':[0,1,2,253,254,255]}
        self.manifest=dict(contract=CONTRACT,schemaVersion=1,synthetic=False,artifactPurpose="training",
            sourceDatasetIndexSha256=_sha256(root/'dataset-index.json'),
            sourceSnapshotSha256=dataset['source_snapshot']['sha256'],sourceSnapshotGeneration='1',
            identityRegistry=self.artifact.registry,upstreamCommit=GRASSY_RUNTIME_UPSTREAM_COMMIT,
            encoder=dict(modelId='isolated-contract-fixture',revision='a'*40,inventorySha256='b'*64,
                referenceFiles={str(n):'a'*64 for n in range(4)},contract='qwen3_vl_text_truncate_l2_1024_v1',modality='text_only'),
            contentSnapshotManifestSha256='c'*64,
            generationMaps={'train':{**dataset['post_generation_identities'],'datasetManifestSha256':_sha256(dataset_path)}},
            codebook=dict(kind='rq_kmeans',levels=6,centroids=256,fitPartition='train',
                trainMembershipSha256=_sha256(self.root/'train-membership.json'),sha256=_sha256(self.root/'codebook.npy'),iterations=20,seed=42),
            counts=dict(posts=1,trainPosts=1),artifacts={})
        filenames={'records':'representation-records.jsonl','mmSnapshot':'mm-snapshot.parquet','sidSnapshot':'sid-snapshot.json',
            'featureAtlas':'feature-atlas.json','nativeSid':'sid-native.parquet','postIds':'post_ids.npy','embeddings':'embeddings.npy'}
        for key,name in filenames.items():
            path=self.root/name
            if not path.exists(): path.write_text('{}\n')
            self.manifest['artifacts'][key]=dict(path=name,sha256=_sha256(path),count=1)
        self.rewrite()

    def rewrite(self):
        path=self.root/'representation-records.jsonl';path.write_text(json.dumps(self.record)+'\n')
        self.manifest['artifacts']['records']['sha256']=_sha256(path)
        _write_json(self.root/'representation-manifest.json',self.manifest)

    def load(self):
        path=self.root/'representation-manifest.json'
        return load_grassy_representations(dataset_root=self.artifact.root,manifest_path=path,expected_sha256=_sha256(path))


class GrassyRepresentationTest(unittest.TestCase):
    def test_exact_sid_codes_attach_to_every_nonpadding_position(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=RepresentationFixture(Path(directory));provider=fixture.load()
            digest=_sha256(fixture.artifact.base)
            batch=next(pq.ParquetFile(fixture.artifact.base).iter_batches())
            attached=provider.attach(batch)
            self.assertEqual(attached.num_columns,37)
            self.assertEqual(attached.column('semanticIdSeq').to_pylist()[0][0],[0,1,2,253,254,255])
            self.assertEqual(attached.column('semanticIdSeq').to_pylist()[0][1],[0]*6)
            self.assertEqual(_sha256(fixture.artifact.base),digest)
            with self.assertRaisesRegex(ObservationSidecarError,'preexisting'): provider.attach(attached)

    def test_no_missing_generation_or_author_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=RepresentationFixture(Path(directory));provider=fixture.load()
            batch=next(pq.ParquetFile(fixture.artifact.base).iter_batches())
            col=batch.column('authorIdSeq').to_pylist();col[0][0]=23
            batch=batch.set_column(batch.schema.get_field_index('authorIdSeq'),'authorIdSeq',pa.array(col,type=pa.list_(pa.int64(),1086)))
            with self.assertRaisesRegex(ObservationSidecarError,'missing exact'):provider.attach(batch)

    def test_hashes_synthetic_marker_sid_range_and_full_nanoseconds_fail(self):
        for mutation in ('hash','synthetic','sid','generation','split','content','extension'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                fixture=RepresentationFixture(Path(directory))
                if mutation=='hash':fixture.manifest['artifacts']['embeddings']['sha256']='f'*64
                if mutation=='synthetic':fixture.manifest['synthetic']=True
                if mutation=='extension':fixture.manifest['artifactPurpose']='serving_extension'
                if mutation=='sid':fixture.record['semanticIds'][0]=256
                if mutation=='generation':fixture.record['createTimeNanoseconds']=1
                if mutation=='split':fixture.manifest['codebook']['fitPartition']='validation'
                if mutation=='content':fixture.record['contentSha256']=None
                fixture.rewrite()
                with self.assertRaises(ObservationSidecarError):fixture.load()

    def test_zero_mm_missing_vector_cannot_become_real_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=RepresentationFixture(Path(directory))
            path=fixture.root/'embeddings.npy';np.save(path,np.zeros((1,1024),dtype=np.float32))
            fixture.manifest['artifacts']['embeddings']['sha256']=_sha256(path)
            fixture.rewrite()
            with self.assertRaisesRegex(ObservationSidecarError,'L2-normalized'):fixture.load()

if __name__=='__main__':unittest.main()
