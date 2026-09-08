# Build context: phoenix/ at the exact Grassy source commit.
# Required base: an operator-built Phoenix CUDA runtime, pinned by SHA256 digest,
# containing this fork at /opt/phoenix and its complete locked Python/Rust environment.
# No floating image, dependency resolution or package download occurs in this layer.
ARG PHOENIX_RUNTIME_IMAGE
FROM ${PHOENIX_RUNTIME_IMAGE}
ARG PHOENIX_RUNTIME_IMAGE
ARG GRASSY_SOURCE_COMMIT
WORKDIR /opt/phoenix
COPY grassy_service /opt/phoenix/grassy_service
RUN python -c "import re, pathlib; image='${PHOENIX_RUNTIME_IMAGE}'; commit='${GRASSY_SOURCE_COMMIT}'; assert re.fullmatch(r'[^ @]+@sha256:[a-f0-9]{64}', image); assert re.fullmatch(r'[a-f0-9]{40}', commit); assert pathlib.Path('/opt/phoenix/SOURCE_COMMIT').read_text().strip() == commit"
RUN python -m grassy_service.verify_runtime
ENV PHOENIX_SOURCE_COMMIT=${GRASSY_SOURCE_COMMIT}
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
# Default is artifact-only preflight, which does not import JAX or initialize CUDA.
ENTRYPOINT ["python", "-m", "grassy_service"]
