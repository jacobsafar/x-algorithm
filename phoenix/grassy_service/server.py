"""IAM-authenticated bounded JSON adapter; native gRPC remains container-private."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import time

from .contracts import ContractError, MAX_BODY, MAX_RESPONSE, require, validate_request


class GoogleIdentityVerifier:
    def __init__(self, audience: str, allowed_email: str):
        from google.auth.transport.requests import Request
        import requests
        require(audience.startswith("https://") and audience.endswith(".run.app") and
                allowed_email.endswith(".iam.gserviceaccount.com"))
        class BoundedSession(requests.Session):
            def request(self, *args, **kwargs):
                kwargs["timeout"] = min(float(kwargs.get("timeout", 2)), 2)
                return super().request(*args, **kwargs)
        self.transport = Request(session=BoundedSession())
        self.audience, self.allowed_email = audience, allowed_email

    def verify(self, authorization: str) -> None:
        from google.oauth2 import id_token
        require(authorization.startswith("Bearer ") and len(authorization) <= 16384, "unauthenticated")
        try:
            claim = id_token.verify_oauth2_token(authorization[7:], self.transport, self.audience)
            require(claim.get("email") == self.allowed_email and claim.get("email_verified") is True and
                    isinstance(claim.get("sub"), str) and claim["sub"], "unauthenticated")
        except Exception as exc:
            raise ContractError("unauthenticated") from exc


def handler(native, verifier, manifest, manifest_sha, atlas, process_alive):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PhoenixShadow"
        def log_message(self, *args):
            pass  # No URL, token, post/user IDs, request payload or model output logging.

        def respond(self, status, value):
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            require(len(body) <= MAX_RESPONSE)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Health exposes only a readiness bit; no manifest/model metadata or IDs.
            if self.path != "/healthz":
                self.respond(404, {"ready": False})
                return
            try:
                require(process_alive(), "model_unavailable")
                native.status(.2)
                self.respond(200, {"ready": True})
            except Exception:
                self.respond(503, {"ready": False})

        def do_POST(self):
            start, status = time.monotonic(), "failed"
            try:
                require(self.path == "/v1/predict-shadow", "route_invalid")
                self.connection.settimeout(.6)
                verifier.verify(self.headers.get("authorization", ""))
                require(self.headers.get("content-type", "").split(";")[0] == "application/json")
                require(self.headers.get("transfer-encoding") is None)
                length_header = self.headers.get("content-length", "")
                require(length_header.isdecimal() and 0 < int(length_header) <= MAX_BODY)
                data = self.rfile.read(int(length_header))
                require(len(data) == int(length_header))
                digest = self.headers.get("x-phoenix-request-sha256", "")
                request = validate_request(data, digest, manifest, manifest_sha, atlas, int(time.time()*1000))
                require(process_alive(), "model_unavailable")
                result = native.predict(request, digest, timeout=.35)
                require(process_alive(), "model_unavailable")
                self.respond(200, result)
                status = "predicted"
            except ContractError as exc:
                reason = str(exc)
                # ContractError messages are fixed enums from this package, never raw input.
                http_status = 401 if reason == "unauthenticated" else 503 if reason.startswith("model_") else 400
                self.respond(http_status, {"error": reason})
                status = "rejected" if http_status < 500 else "unavailable"
            except Exception:
                try:
                    self.respond(503, {"error": "model_unavailable"})
                except (BrokenPipeError, OSError):
                    pass
            finally:
                print(json.dumps({"event": "phoenix_shadow_request", "schemaVersion": 1,
                    "status": status, "elapsedMs": round((time.monotonic()-start)*1000)}), flush=True)
    return Handler


def serve(native, verifier, manifest, manifest_sha, atlas, process_alive):
    # Serial HTTP admission bounds concurrent GPU work. Cloud Run concurrency must also be1.
    server = HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))),
                        handler(native, verifier, manifest, manifest_sha, atlas, process_alive))
    server.serve_forever()
