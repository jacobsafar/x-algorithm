# SPDX-License-Identifier: Apache-2.0
"""Execute production NumPy samplers without importing CUDA/JAX extensions."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_grassy_model_contract import _load_function

SOURCE=Path(__file__).resolve().parents[1]/'xrex/data/recsys/recsys_batch.py'


def sequence(masked=True):
    shape=(2,2)
    value=dict(impr_ts=np.full(shape,100,dtype=np.int32),
        actions=np.ones((*shape,64),dtype=np.bool_),continuous_actions=np.ones((*shape,8),dtype=np.float32),
        post_hashes=np.ones((*shape,1),dtype=np.int64),auth_hashes=np.ones((*shape,1),dtype=np.int64),
        ip_hashes=np.zeros((*shape,1),dtype=np.int64),product_surface=np.ones(shape,dtype=np.int32),
        client_app_id=np.ones(shape,dtype=np.int32),post_creation_ts_sec=np.ones(shape,dtype=np.int32),
        post_ids=np.array([[11,12],[21,22]],dtype=np.int64),trained_candidate_mask=np.ones(shape,dtype=np.bool_),
        promoted_ids=np.zeros(shape,dtype=np.int64),line_item_objective=np.zeros(shape,dtype=np.int16),
        safety_label_mask=np.zeros(shape,dtype=np.int64),embedding=None,search_query_embeddings=None,
        categorical_features=np.zeros((*shape,0),dtype=np.int16),bool_features=np.zeros((*shape,0),dtype=np.bool_),
        float_features=np.zeros((*shape,0),dtype=np.float32),int64_features=np.zeros((*shape,0),dtype=np.int64),
        post_sids=np.ones((*shape,6),dtype=np.uint16))
    if masked:
        value.update(action_observation_mask=np.ones((*shape,64),dtype=np.bool_),
            continuous_action_observation_mask=np.ones((*shape,8),dtype=np.bool_))
    return value


class NegativeAuthorityTest(unittest.TestCase):
    def run_sampler(self,kind,masked=True):
        sampler=_load_function(SOURCE,kind,dict(np=np,PostSeq=dict,
            action_type_map={'ClientTweetRecapNotDwelled':13}))
        source=sequence(masked)
        if kind=='apply_negative_sampling':
            return source,sampler(np.array([1,2]),source,1,2,2)
        table=SimpleNamespace(get_item_hash=lambda ids:ids[...,None],get_author_hash=lambda ids:ids[...,None])
        return source,sampler(np.array([31,32]),np.array([41,42]),table,source,2,2,
            global_post_sids=np.zeros((2,6),dtype=np.int32))

    def test_donor_and_global_samples_never_inherit_exact_label_authority(self):
        for kind in ('apply_negative_sampling','apply_global_negative_sampling'):
            with self.subTest(kind=kind):
                source,result=self.run_sampler(kind)
                for key in ('action_observation_mask','continuous_action_observation_mask'):
                    np.testing.assert_array_equal(result[key][:,:2],source[key])
                    self.assertFalse(result[key][:,2:].any())
                self.assertTrue(result['trained_candidate_mask'][:,:2].all())
                self.assertFalse(result['trained_candidate_mask'][:,2:].any())
                # The generic sampler retains its explicit artificial label marker,
                # but no Grassy head can train or report metrics on that marker.
                self.assertTrue(result['actions'][:,2:,13].all())

    def test_unmasked_upstream_sampling_behavior_is_preserved(self):
        for kind in ('apply_negative_sampling','apply_global_negative_sampling'):
            with self.subTest(kind=kind):
                source,result=self.run_sampler(kind,False)
                self.assertNotIn('action_observation_mask',result)
                self.assertTrue(result['trained_candidate_mask'][:,2:].all())
                np.testing.assert_array_equal(result['actions'][:,:2],source['actions'])

    def test_zero_sampling_does_not_create_or_drop_candidates(self):
        sampler=_load_function(SOURCE,'apply_negative_sampling',dict(np=np,PostSeq=dict))
        source=sequence()
        self.assertIs(sampler(np.array([1,1]),source,0,2,2),source)

if __name__=='__main__':unittest.main()
