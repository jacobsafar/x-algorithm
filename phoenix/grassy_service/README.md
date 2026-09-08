# Grassy Phoenix shadow service

This adapter executes the current fork's `RankingModelRunner` through its native gRPC `PredictNextActions` method. It is separate from Firebase's four-head ONNX ranker. The executable contains no synthetic model, random initialization fallback, fake embeddings or default prediction scores. It does not enable live ranking.

The current source is X upstream `902a06fd616ed815f660e5546d16d492fa1ca825`, plus a separately pinned Grassy commit with `grassy_unavailable_features_disabled_v1`, per-head observation masks, `grassy_phoenix_dataset_v5`, and `legacy_48h_v1`. Runtime geometry is 1022 history positions, 64 candidates, 64 discrete and eight continuous outputs. New age/data/model policies require a different admitted contract.

## Artifacts and admission

The backend pins the SHA256 of `manifest.json` bytes. Its TypeScript interface is `PhoenixManifest` in `functions/src/phoenix-readiness.ts`. The service independently checks the same contracts. Pin the full source/config/preprocessing/mask/age/identity-key/action-contract/MM/SID/checkpoint tuple, not a friendly version name. `trainingKind` must be `production_exact`, training steps positive, and utility can use only explicitly trained, observed non-padding heads. These operator-attested fields do not themselves prove model quality.

`checkpoint-inventory.json` is outside the checkpoint directory. It contains `schemaVersion:1`, `trainingKind`, `trainingSteps`, `trainingDatasetSha256`, `modelConfigName`, `grassyCommit`, `featurePolicy`, and `files:[{path,bytes,sha256}]`. Inventory every checkpoint file, including `completed` and all `orbax-ckpt/` files. Extra files, symlinks, changed hashes, incomplete checkpoints and broad/latest paths fail admission. Mount the exact directory read-only. The runtime restore hook also checks the actual resolved checkpoint, positive restored elapsed samples and Grassy model feature guards. The native loader retains its checksum checks. Reload and hotswap are disabled; each model promotion requires a new revision.

The MM snapshot uses the native Rust schema: column0 `uint64` post ID, column1 version, column2 `large_list<float32>` containing exactly1024 finite nonzero-vector values. The service validates every row and its little-endian float32 digest against the atlas. Native inference reads this exact `MM_LOCAL_SNAPSHOT`; inline protobuf MM embeddings are not authoritative in the current upstream engine.

SID JSON is `{schemaVersion:1,entries:[{modelPostId:"decimal int63",semanticIds:[six unshifted codes0..255]}]}`. Atlas JSON is `{schemaVersion:1,inputContract:"grassy_phoenix_online_features_v1",identityKeyId,embeddingSnapshotSha256,sidSnapshotSha256,entries:[PhoenixCandidateIdentity]}`. The atlas has an exact one-to-one post inventory with MM and SID snapshots. Candidate identities are generation HMACs, exporter-v5 decimal model post/author IDs, full Firestore creation seconds/nanoseconds, creation milliseconds, embedding digest and SID codes. Actual content/generation provenance belongs to the representation materializer and online provider; never label fixture/random vectors as production artifacts.

The online provider must use `identity_v1.py` / the same stable HMAC registry as exporter v5. No remapping or key creation occurs in this service. JSON model IDs are decimal strings to preserve int64 precision. History requires strictly increasing times, distinct post generations disjoint from candidates, observed positive action ordinals and actual surface. Dwell is attached once, because native preprocessing sums action metadata. Null dwell remains explicitly unavailable at the provider boundary; the upstream numeric history placeholder follows the existing training encoding.

## Wire and authentication

`POST /v1/predict-shadow` accepts the strict bounded request in `contracts.py`, at most1MiB, <=64 candidates and <=1022 history events. Cloud Run IAM must admit only the backend service account. The HTTP adapter additionally verifies the Google-signed ID token's exact audience, issuer, expiry and verified caller email. Configure `PHOENIX_IAM_AUDIENCE` to the exact service HTTPS origin, and `PHOENIX_CALLER_SERVICE_ACCOUNT` to that single service account. Only HTTP port8080 is exposed; native gRPC9988 and metrics9090 must remain private. The backend uses `Authorization`, preserving the signed token for container verification; Google documents signature removal for the alternate `X-Serverless-Authorization` header ([Cloud Run service authentication](https://docs.cloud.google.com/run/docs/authenticating/service-to-service)).

A UUID nonce and SHA256 of exact HTTP body bytes bind every response. The current native engine does not populate response `checkpointPath`. The protected restore hook therefore publishes the admitted manifest digest through the existing private status RPC after loading; the adapter checks it and the checkpoint timestamp before and after prediction. A mismatch/reload/unavailable model produces no predictions.

Despite its legacy field names, current `RankingModelRunner` returns **log-sigmoid probabilities**, not raw logits. The service explicitly requests `returnLogprob=true`, returns `topLogProbs`, and Firebase computes `exp(logProbability)`. It never applies sigmoid a second time. Nonfinite outputs, wrong widths, IDs, nonce, input digest or model identity fail closed. Logs contain only fixed statuses, counts and elapsed time.

## Build and preflight

No production trained checkpoint, full Linux/CUDA runtime image, GPU allocation or real-model smoke result has been created by this change. CPU contract tests and artifact hashing do not satisfy `actualCheckpointPredictionPassed`.

`Dockerfile` is an adapter layer over a **required digest-pinned Phoenix CUDA runtime image**. That image must contain this exact fork at `/opt/phoenix`, its `SOURCE_COMMIT` file, compiled `xai_recsys_engine`, generated `xai_proto`, CUDA-compatible JAX and a reviewed full transitive dependency lock. `runtime-requirements.txt` pins this adapter's directly checked packages; it is not falsely presented as a full transitive lock. Build rejects source/version drift; it performs no unpinned package install. A real approved runtime image is an explicit remaining deployment dependency.

Set the environment paths from `cloud-run.template.yaml`, plus `PHOENIX_MANIFEST_SHA256` and `PHOENIX_SOURCE_COMMIT`, then run this read-only, CPU-safe preflight:

```sh
python -m grassy_service.verify_runtime
python -m grassy_service
python -m unittest discover -s tests -p 'test_grassy_service.py' -v
```

Preflight verifies immutable artifacts and reports `actualCheckpointPredictionPassed:false`. On separately approved compatible compute, `python -m grassy_service --serve` starts the actual upstream runner. A real inference smoke must use an admitted production checkpoint and real representation inventory, then preserve the request/model/input digests and bounded aggregate output-quality/latency evidence. Do not use upstream `oss_bench --smoke` or synthetic examples as rollout evidence.

The Cloud Run file is an unresolved deployment review template. It creates nothing; resolve the immutable image, read-only artifact mount, IAM/VPC routing, GPU/CPU/memory quota and budget before deployment. Keep backend Phoenix shadow disabled until runtime/feature readiness is proven. Live feed ordering remains V2 throughout this lane.

The v6 service policy is a separate tuple: model `grassy_home_direct_packed_nano_90d`, dataset `grassy_phoenix_dataset_v6`, feature4/proof2, candidate age `grassy_candidate_age_90d_v1`/7776000000ms, age encoding `grassy_post_age_30h_80bins_v1`/1800-minute buckets/144000-minute maximum, and explicit served-content render policy. Its inventory must repeat those pins. Native restore checks actual model age configuration. V5 stays on its original48h/feature3/base model tuple; replacing only an age number never admits an incompatible checkpoint. Head0 sentinels and padding dimensions never receive trained utility. These guards do not substitute for GPU inference verification, independent checkpoint restoration or held-out evaluation.

Readiness publication additionally checks actual restored state.step == manifest.trainingSteps and actual elapsed_samples == trainingSteps *32, matching the single-device Grassy training recipe. Randomly initialized or older state cannot publish a newer run's manifest digest.
