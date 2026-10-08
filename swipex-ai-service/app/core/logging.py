"""Structured logging and Request ID middleware context."""

import logging
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "system"
        return True


def setup_logging() -> None:
    """Configure structured logging format and context filter."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [req_id:%(request_id)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    for h in logger.handlers:
        h.addFilter(RequestIDFilter())


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to assign a unique request_id to each incoming request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        request.state.request_id = req_id
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx.reset(token)


logger = logging.getLogger("swipex.ai")
