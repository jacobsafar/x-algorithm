# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path
from test_grassy_representations import RepresentationFixture
from test_observation_sidecars_v6 import ArtifactV6
from xrex.data.recsys.observation_sidecars import ObservationSidecarError

class RepresentationV6Test(unittest.TestCase):
    def test_only_matching_proof_bound_content_and_encoder_policy_admitted(self):
        for caption in (False,True):
            for corrupt in (None,'revision','content','encoder'):
                with self.subTest(caption=caption,corrupt=corrupt),tempfile.TemporaryDirectory() as temp:
                    render='grassy_caption_only_v1' if caption else 'grassy_text_only_v1'
                    fixture=RepresentationFixture(Path(temp),lambda root:ArtifactV6(root,render))
                    if caption:fixture.manifest['encoder'].update(contract='qwen3_vl_caption_truncate_l2_1024_v1',modality='caption_only')
                    if corrupt=='revision':fixture.record['sourceUpdateTimeNanoseconds']+=1
                    if corrupt=='content':fixture.record['contentSha256']='a'*64
                    if corrupt=='encoder':fixture.manifest['encoder']['modality']='other'
                    fixture.rewrite()
                    if corrupt:
                        with self.assertRaises(ObservationSidecarError):fixture.load()
                    else:self.assertEqual(len(fixture.load().records),1)

if __name__=='__main__':unittest.main()
