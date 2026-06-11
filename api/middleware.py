"""Request logging and latency middleware."""

from __future__ import annotations

import time
import uuid
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with a unique request_id, method, path,
    status code, and latency. Adds X-Request-ID and X-Latency-Ms
    response headers for easy inspection in Swagger UI.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"

        logger.info(
            "%s %s %d %.0fms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response
