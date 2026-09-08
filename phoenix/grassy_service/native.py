"""Only production backend: the current upstream gRPC RankingModelRunner."""
from __future__ import annotations

import math
import time

from .contracts import CONTRACT, MAX_RESPONSE, require


class NativePhoenix:
    def __init__(self, manifest: dict, manifest_sha: str, atlas: dict, target: str = "127.0.0.1:9988"):
        # Deliberately no arbitrary remote target or fake/test fallback in the executable.
        require(target == "127.0.0.1:9988")
        import grpc
        from xai_proto import recsys_pb2, recsys_pb2_grpc
        self.pb = recsys_pb2
        self.channel = grpc.insecure_channel(target, options=[
            ("grpc.max_send_message_length", 1024 * 1024),
            ("grpc.max_receive_message_length", MAX_RESPONSE)])
        self.stub = recsys_pb2_grpc.RecsysPredictorStub(self.channel)
        self.manifest, self.manifest_sha, self.atlas = manifest, manifest_sha, atlas

    def status(self, timeout: float) -> tuple[str, float]:
        status = self.stub.ReloadModel(self.pb.ReloadModelRequest(do_reload=False), timeout=timeout)
        # peer_ready is the P2P copy server, not ranking readiness. The protected runner
        # publishes this manifest digest only after verifying and restoring its exact checkpoint.
        require(not status.reload_in_progress and status.serving_prefix == self.manifest_sha and
                math.isfinite(status.current_checkpoint_timestamp) and status.current_checkpoint_timestamp > 0,
                "model_unavailable")
        return status.serving_prefix, status.current_checkpoint_timestamp

    def _tweet(self, item: dict):
        return self.pb.TweetInfo(tweetId=int(item["modelPostId"]), authorId=int(item["modelAuthorId"]),
                                 semanticIds=item["semanticIds"])

    def predict(self, request: dict, request_sha: str, timeout: float = .35) -> dict:
        end = time.monotonic() + timeout
        def remaining():
            value = end - time.monotonic()
            require(value > 0, "model_timeout")
            return value
        before = self.status(remaining())
        value, pb = request["inputs"], self.pb
        user_id = int(value["modelUserId"])
        history = []
        for index, item in enumerate(value["history"]):
            # Native engine sums action-meta dwell; attach the real value exactly once.
            actions = [pb.ActionInfo(engagementTimeMs=item["impressedAtMs"], actionName=head,
                userActionMeta=pb.UserActionMeta(indexInSequence=index,
                    productSurface=1 if item["surface"] == "for_you" else 9,
                    dwellTime=item["dwellMs"] if ordinal == 0 and item["dwellMs"] is not None else 0))
                for ordinal, head in enumerate(item["actionOrdinals"])]
            history.append(pb.AggregatedUserAction(userId=user_id, tweetInfo=self._tweet(item),
                impressedTimeMs=item["impressedAtMs"], actions=actions))
        sequence = pb.UserActionSequence(userId=user_id,
            userActionsData=pb.UserActionSequenceDataContainer(
                orderedAggregatedUserActionsList=pb.AggregatedUserActionList(aggregatedUserActions=history)),
            metadata=pb.UserActionSequenceMeta(length=len(history),
                firstSequenceTime=value["history"][0]["impressedAtMs"] if history else 0,
                lastSequenceTime=value["history"][-1]["impressedAtMs"] if history else 0))
        reply = self.stub.PredictNextActions(pb.PredictNextActionsRequest(
            sequences=[sequence], candidateSets=[pb.CandidateSet(userId=user_id, productSurface=1,
                candidates=[self._tweet(item) for item in value["candidates"]])],
            returnLogprob=True, returnLogMap=False, returnCandidateTweetIdOnly=True), timeout=remaining())
        after = self.status(remaining())
        require(before == after and len(reply.distributionSets) == 1, "model_identity_mismatch")
        result = reply.distributionSets[0]
        require(result.userId == user_id and len(result.candidateDistributions) == len(value["candidates"]),
                "model_identity_mismatch")
        predictions = []
        for item, distribution in zip(value["candidates"], result.candidateDistributions):
            logp, continuous = list(distribution.topLogProbs), list(distribution.continuousActionsValues)
            require(distribution.candidate.tweetId == int(item["modelPostId"]) and
                    len(logp) == 64 and len(continuous) == 8 and
                    all(math.isfinite(v) and v <= 0 for v in logp) and
                    all(math.isfinite(v) for v in continuous), "model_output_invalid")
            predictions.append({"generationKey": item["generationKey"], "modelPostId": item["modelPostId"],
                                "logProbabilities": logp, "continuousValues": continuous})
        return {"schemaVersion": 1, "contract": CONTRACT, "nonce": request["nonce"],
                "requestSha256": request_sha, "manifestSha256": self.manifest_sha,
                "checkpointManifestSha256": self.manifest["checkpointManifestSha256"],
                "outputSemantics": "log_sigmoid_v1", "predictions": predictions}
