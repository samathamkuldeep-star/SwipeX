"""Custom exception hierarchy and standard error envelope handlers."""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger("swipex.ai.exceptions")


class AIServiceError(Exception):
    """Base exception for all AI Service domain errors."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        code: str = "INTERNAL_ERROR",
        message: str = "An unexpected error occurred.",
        details: Any | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class UnsupportedFileTypeError(AIServiceError):
    def __init__(self, message: str = "Unsupported file format. Only PDF and DOCX files are allowed.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            code="UNSUPPORTED_FILE_TYPE",
            message=message,
            details=details,
        )


class FileTooLargeError(AIServiceError):
    def __init__(self, message: str = "File size exceeds maximum allowed limit.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            code="FILE_TOO_LARGE",
            message=message,
            details=details,
        )


class CorruptFileError(AIServiceError):
    def __init__(self, message: str = "The uploaded file is corrupt or unreadable.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="CORRUPT_FILE",
            message=message,
            details=details,
        )


class NoExtractableTextError(AIServiceError):
    def __init__(self, message: str = "No extractable text found in document (e.g. scanned image PDF). OCR is not supported.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="NO_EXTRACTABLE_TEXT",
            message=message,
            details=details,
        )


class InvalidInputError(AIServiceError):
    def __init__(self, message: str = "Invalid input payload.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="INVALID_INPUT",
            message=message,
            details=details,
        )


class UnauthorizedError(AIServiceError):
    def __init__(self, message: str = "Invalid or missing internal API key.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="UNAUTHORIZED",
            message=message,
            details=details,
        )


class EmbeddingDimensionError(AIServiceError):
    def __init__(self, message: str = "Generated embedding dimension does not match configured dimension.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="EMBEDDING_DIMENSION_ERROR",
            message=message,
            details=details,
        )


class VectorStoreUnavailableError(AIServiceError):
    """Internal exception when Chroma vector store is unreachable (service degrades gracefully)."""

    def __init__(self, message: str = "Vector store index service unavailable.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="VECTOR_STORE_UNAVAILABLE",
            message=message,
            details=details,
        )


class LLMUnavailableError(AIServiceError):
    """Internal exception when LLM fails or times out (falls back to rule-based engine)."""

    def __init__(self, message: str = "LLM service unavailable.", details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="LLM_UNAVAILABLE",
            message=message,
            details=details,
        )


def make_error_envelope(code: str, message: str, details: Any | None = None, request_id: str = "") -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AIServiceError)
    async def ai_service_error_handler(request: Request, exc: AIServiceError):
        req_id = getattr(request.state, "request_id", "")
        logger.warning(f"AIServiceError [{exc.code}]: {exc.message} (status: {exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error_envelope(exc.code, exc.message, exc.details, req_id),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "")
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=make_error_envelope(
                "INVALID_INPUT",
                "Request validation failed.",
                exc.errors(),
                req_id,
            ),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
        req_id = getattr(request.state, "request_id", "")
        logger.warning(f"Pydantic validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=make_error_envelope(
                "INVALID_INPUT",
                "Data validation failed.",
                exc.errors(),
                req_id,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "")
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=make_error_envelope(
                "INTERNAL_ERROR",
                "An unexpected internal server error occurred.",
                None,
                req_id,
            ),
        )
