"""CPU artifact/transport tests. No real checkpoint or model execution is claimed."""
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grassy_service.contracts import (CONTRACT, INPUT_CONTRACT, UPSTREAM, ContractError,
    load_feature_atlas, load_manifest, sha256, validate_request, verify_checkpoint, validate_runtime_model_age_policy, validate_restored_training_state, V6_POLICY_FIELDS)
from grassy_service.native import NativePhoenix

NOW = 1788818850000
H = 'a' * 64


def manifest():
    return {"schemaVersion": 1, "contract": CONTRACT, "upstreamCommit": UPSTREAM,
        "grassyCommit": "b" * 40, "modelConfigName": "grassy_home_direct_packed_nano",
        "featurePolicy": "grassy_unavailable_features_disabled_v1", "inputContract": INPUT_CONTRACT,
        "sourceContract": "grassy_phoenix_dataset_v5", "featureSchemaVersion": 3, "proofSchemaVersion": 2,
        "agePolicy": "legacy_48h_v1", "maxCandidateAgeMs": 172800000,
        "identityContract": "grassy_hmac_identity_user_split_v1", "identityKeyId": "test-key",
        "identityKeySha256": H, "projectId": "test-project", "actionContractSha256": H,
        "embeddingSnapshotSha256": H, "sidSnapshotSha256": H, "featureAtlasSha256": H,
        "checkpointManifestSha256": H, "trainingDatasetSha256": H, "trainingKind": "production_exact",
        "trainingSteps": 5, "observationMaskPolicy": "required_per_head_v1", "historyPositions": 1022,
        "candidatePositions": 64, "discreteWidth": 64, "continuousWidth": 8,
        "trainedDiscreteHeads": [1, 4], "trainedContinuousHeads": [1],
        "utilityPolicy": "trained_heads_weighted_sum_v1", "discreteWeights": {"1": .5, "4": 5},
        "continuousWeights": {"1": .004}}


def candidate(index=1):
    created = NOW - 3600000
    return {"generationKey": str(index) * 64, "modelPostId": str(((created - 1288834974657) << 22) + index),
        "modelAuthorId": str(index), "createTimeSeconds": created // 1000, "createTimeNanoseconds": 0,
        "createdAtMs": created, "embeddingSha256": sha256(struct.pack('<1024f', *([.5] * 1024))),
        "semanticIds": [0, 1, 2, 3, 4, 5]}


def request():
    return {"schemaVersion": 1, "contract": CONTRACT,
        "nonce": "12345678-1234-4234-8234-123456789abc", "manifestSha256": H,
        "inputs": {"inputContract": INPUT_CONTRACT, "manifestSha256": H, "identityKeyId": "test-key",
            "modelUserId": "123", "asOfMs": NOW, "capturedAtMs": NOW,
            "history": [], "candidates": [candidate()]}}


def write_json(path, value):
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return sha256(data)


class ContractTests(unittest.TestCase):
    def test_manifest_rejects_untrained_synthetic_mask_and_policy_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            original = manifest()
            digest = write_json(path, original)
            self.assertEqual(load_manifest(path, digest), original)
            for key, value in [('trainingSteps', 0), ('trainingKind', 'synthetic'),
                    ('featurePolicy', 'defaults'), ('trainedDiscreteHeads', [63]), ('trainedDiscreteHeads',[0]),
                    ('trainedContinuousHeads',[0]),
                    ('continuousWeights', {'7': 1}), ('sourceContract', 'grassy_phoenix_dataset_v4'),
                    ('maxCandidateAgeMs', 90 * 86400000), ('discreteWidth', True)]:
                changed = {**original, key: value}
                digest = write_json(path, changed)
                with self.assertRaises(ContractError, msg=key):
                    load_manifest(path, digest)

    def test_v6_manifest_runtime_age_and_candidate_admission_are_one_tuple(self):
        v6 = {**manifest(), "sourceContract":"grassy_phoenix_dataset_v6", "modelConfigName":"grassy_home_direct_packed_nano_90d",
            "featureSchemaVersion":4, "agePolicy":"grassy_candidate_age_90d_v1", "maxCandidateAgeMs":7776000000,
            "ageEncoding":"grassy_post_age_30h_80bins_v1", "postAgeGranularityMins":1800, "postAgeMaxMins":144000,
            "contentContract":"grassy_served_content_v1", "renderPolicy":"grassy_caption_only_v1"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            self.assertEqual(load_manifest(path, write_json(path,v6)),v6)
            for key,value in [("sourceContract","grassy_phoenix_dataset_v5"), ("modelConfigName","grassy_home_direct_packed_nano"),
                    ("featureSchemaVersion",3), ("agePolicy","legacy_48h_v1"), ("postAgeGranularityMins",60),
                    ("postAgeMaxMins",4800), ("ageEncoding","phoenix_post_age_1h_80bins_v1"), ("renderPolicy",None)]:
                invalid = {**v6,key:value}
                with self.assertRaises(ContractError,msg=key): load_manifest(path,write_json(path,invalid))
        model = SimpleNamespace(grassy_candidate_age_policy=v6['agePolicy'], grassy_post_age_encoding=v6['ageEncoding'],
            post_age_granularity_mins=1800, post_age_max_mins=144000,
            feature_prep=SimpleNamespace(post_age_granularity_mins=1800,post_age_max_mins=144000))
        validate_runtime_model_age_policy(model,v6)
        model.feature_prep.post_age_granularity_mins=60
        with self.assertRaises(ContractError): validate_runtime_model_age_policy(model,v6)
        original=request()
        c=original['inputs']['candidates'][0]
        c.update(createdAtMs=NOW-30*86400000, createTimeSeconds=(NOW-30*86400000)//1000,
                 modelPostId=str(((NOW-30*86400000-1288834974657)<<22)+1))
        atlas={c['modelPostId']:c}
        data=json.dumps(original).encode()
        validate_request(data,sha256(data),v6,H,atlas,NOW)
        with self.assertRaises(ContractError): validate_request(data,sha256(data),manifest(),H,atlas,NOW)

    def test_actual_restored_progress_must_match_manifest_and_admitted_batch32_recipe(self):
        validate_restored_training_state(5, 160, manifest())
        for step, samples in [(0,0),(4,128),(5,159),(5,320),(True,160),(5.0,160)]:
            with self.assertRaises(ContractError): validate_restored_training_state(step,samples,manifest())

    def test_request_digest_time_identity_and_atlas_mismatch(self):
        atlas = {candidate()['modelPostId']: candidate()}
        original = request()
        data = json.dumps(original).encode()
        self.assertEqual(validate_request(data, sha256(data), manifest(), H, atlas, NOW), original)
        for update in [lambda x: x.update(nonce='old'),
                lambda x: x['inputs'].update(modelUserId=123),
                lambda x: x['inputs'].update(capturedAtMs=NOW-300001),
                lambda x: x['inputs'].update(extraRawUserId='private'),
                lambda x: x['inputs']['candidates'][0].update(createTimeNanoseconds=1),
                lambda x: x['inputs']['candidates'][0].update(semanticIds=[1, 1, 2, 3, 4, 5]),
                lambda x: x['inputs'].update(candidates=[candidate(), candidate()])]:
            changed = deepcopy(original)
            update(changed)
            data = json.dumps(changed).encode()
            with self.assertRaises(ContractError):
                validate_request(data, sha256(data), manifest(), H, atlas, NOW)
        with self.assertRaises(ContractError):
            validate_request(b'{"schemaVersion":1,"schemaVersion":1}', H, manifest(), H, atlas, NOW)

    def test_checkpoint_hashes_exact_inventory_and_no_extra_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'checkpoint'
            (root / 'orbax-ckpt').mkdir(parents=True)
            (root / 'completed').write_text('complete')
            (root / 'orbax-ckpt' / 'weights').write_bytes(b'test-only-non-model-file')
            m = manifest()
            inventory = {k: m[k] for k in ['trainingKind', 'trainingSteps', 'trainingDatasetSha256',
                'modelConfigName', 'grassyCommit', 'featurePolicy']}
            inventory.update(schemaVersion=1, files=[{'path': str(p.relative_to(root)),
                'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes())}
                for p in [root / 'completed', root / 'orbax-ckpt' / 'weights']])
            inventory_path = base / 'inventory.json'
            m['checkpointManifestSha256'] = write_json(inventory_path, inventory)
            self.assertEqual(verify_checkpoint(root, m, inventory_path)[0], root.resolve())
            (root / 'injected').write_text('unlisted')
            with self.assertRaises(ContractError):
                verify_checkpoint(root, m, inventory_path)
            (root / 'injected').unlink()
            (root / 'orbax-ckpt' / 'weights').write_bytes(b'modified')
            with self.assertRaises(ContractError):
                verify_checkpoint(root, m, inventory_path)

    def test_v6_checkpoint_inventory_must_repeat_exact_policy_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); root=base/'checkpoint'; (root/'orbax-ckpt').mkdir(parents=True)
            (root/'completed').write_bytes(b'unit-test-only')
            (root/'orbax-ckpt'/'metadata').write_bytes(b'not-a-real-checkpoint')
            m={**manifest(), "sourceContract":"grassy_phoenix_dataset_v6", "modelConfigName":"grassy_home_direct_packed_nano_90d",
               "featureSchemaVersion":4,"agePolicy":"grassy_candidate_age_90d_v1","maxCandidateAgeMs":7776000000,
               "ageEncoding":"grassy_post_age_30h_80bins_v1","postAgeGranularityMins":1800,"postAgeMaxMins":144000,
               "contentContract":"grassy_served_content_v1","renderPolicy":"grassy_caption_only_v1"}
            inventory={k:m[k] for k in ('schemaVersion','trainingKind','trainingSteps','trainingDatasetSha256',
                'modelConfigName','grassyCommit','featurePolicy',*V6_POLICY_FIELDS)}
            inventory['files']=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes())}
                for p in root.rglob('*') if p.is_file()]
            path=base/'inventory.json'; m['checkpointManifestSha256']=write_json(path,inventory)
            verify_checkpoint(root,m,path)
            for field in V6_POLICY_FIELDS:
                invalid={**inventory}; invalid.pop(field)
                m['checkpointManifestSha256']=write_json(path,invalid)
                with self.assertRaises(ContractError,msg=field): verify_checkpoint(root,m,path)

    def test_native_mm_parquet_geometry_and_values_bind_actual_atlas(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            m, item = manifest(), candidate()
            mm, sid, atlas = base / 'mm.parquet', base / 'sid.json', base / 'atlas.json'
            table = pa.table({'postId': pa.array([int(item['modelPostId'])], type=pa.uint64()),
                'version': ['v5'], 'embedding': pa.array([[.5] * 1024], type=pa.large_list(pa.float32()))})
            pq.write_table(table, mm)
            m['embeddingSnapshotSha256'] = sha256(mm.read_bytes())
            m['sidSnapshotSha256'] = write_json(sid, {'schemaVersion': 1,
                'entries': [{'modelPostId': item['modelPostId'], 'semanticIds': item['semanticIds']}]})
            atlas_doc = {'schemaVersion': 1, 'inputContract': INPUT_CONTRACT, 'identityKeyId': 'test-key',
                'embeddingSnapshotSha256': m['embeddingSnapshotSha256'],
                'sidSnapshotSha256': m['sidSnapshotSha256'], 'entries': [item]}
            m['featureAtlasSha256'] = write_json(atlas, atlas_doc)
            self.assertEqual(load_feature_atlas(atlas, mm, sid, m), {item['modelPostId']: item})
            atlas_doc['entries'][0]['embeddingSha256'] = H
            m['featureAtlasSha256'] = write_json(atlas, atlas_doc)
            with self.assertRaises(ContractError):
                load_feature_atlas(atlas, mm, sid, m)


class ProtoFactory:
    def __getattr__(self, name):
        return lambda **kwargs: SimpleNamespace(**kwargs)


class Stub:
    def __init__(self):
        self.sent = None
        self.bad_status = False
        self.status_calls = 0
        self.bad_output = False
    def ReloadModel(self, request, timeout):
        self.status_calls += 1
        assert request.do_reload is False
        return SimpleNamespace(reload_in_progress=False,
            serving_prefix='wrong' if self.bad_status else H, current_checkpoint_timestamp=123.0)
    def PredictNextActions(self, request, timeout):
        self.sent = request
        values = [math.log(.8)] * 64
        if self.bad_output:
            values[0] = .1
        return SimpleNamespace(distributionSets=[SimpleNamespace(userId=123,
            candidateDistributions=[SimpleNamespace(candidate=SimpleNamespace(tweetId=int(candidate()['modelPostId'])),
                topLogProbs=values, continuousActionsValues=[0.] * 8)])])


class NativeTransportTests(unittest.TestCase):
    def engine(self):
        native = NativePhoenix.__new__(NativePhoenix)
        native.pb, native.stub = ProtoFactory(), Stub()
        native.manifest, native.manifest_sha = manifest(), H
        return native
    def test_actual_grpc_method_wiring_uses_log_probabilities_and_no_reload(self):
        native = self.engine()
        result = native.predict(request(), H)
        self.assertTrue(native.stub.sent.returnLogprob)
        self.assertFalse(native.stub.sent.returnLogMap)
        self.assertEqual(native.stub.sent.candidateSets[0].candidates[0].semanticIds, [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['outputSemantics'], 'log_sigmoid_v1')
        self.assertAlmostEqual(math.exp(result['predictions'][0]['logProbabilities'][0]), .8)
        self.assertEqual(native.stub.status_calls, 2)
    def test_wrong_checkpoint_or_invalid_distribution_cannot_return_scores(self):
        native = self.engine()
        native.stub.bad_status = True
        with self.assertRaises(ContractError):
            native.predict(request(), H)
        self.assertIsNone(native.stub.sent)
        native = self.engine()
        native.stub.bad_output = True
        with self.assertRaises(ContractError):
            native.predict(request(), H)
    def test_history_dwell_is_attached_exactly_once_with_real_surface(self):
        native = self.engine()
        value = request()
        value['inputs']['history'] = [{**candidate(2), 'impressedAtMs': NOW-1000,
            'surface': 'profile', 'actionOrdinals': [1, 4], 'dwellMs': 2500}]
        native.predict(value, H)
        actions = native.stub.sent.sequences[0].userActionsData.orderedAggregatedUserActionsList.aggregatedUserActions[0].actions
        self.assertEqual([a.userActionMeta.dwellTime for a in actions], [2500, 0])
        self.assertEqual(actions[0].userActionMeta.productSurface, 9)


class IdentityAuthenticationTests(unittest.TestCase):
    def test_google_token_requires_exact_audience_and_verified_single_caller(self):
        from grassy_service.server import GoogleIdentityVerifier
        verifier = GoogleIdentityVerifier('https://test.a.run.app', 'backend@test.iam.gserviceaccount.com')
        good = {'email': 'backend@test.iam.gserviceaccount.com', 'email_verified': True, 'sub': '123'}
        with patch('google.oauth2.id_token.verify_oauth2_token', return_value=good) as verify:
            verifier.verify('Bearer unit-test-token')
            self.assertEqual(verify.call_args.args[2], 'https://test.a.run.app')
        for bad in [{**good, 'email': 'other@test.iam.gserviceaccount.com'},
                    {**good, 'email_verified': False}, {**good, 'sub': ''}]:
            with patch('google.oauth2.id_token.verify_oauth2_token', return_value=bad):
                with self.assertRaises(ContractError):
                    verifier.verify('Bearer unit-test-token')
        with patch('google.oauth2.id_token.verify_oauth2_token', side_effect=ValueError('expired private-token')):
            with self.assertRaisesRegex(ContractError, '^unauthenticated$'):
                verifier.verify('Bearer unit-test-token')


if __name__ == '__main__':
    unittest.main()
