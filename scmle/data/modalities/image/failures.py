"""Errors raised by image ingestion and feature preparation."""

from typing import Any

from scmle.contracts import errors


class ImageError(errors.DataError):
    """Base error for image ingestion and feature preparation.

    Attributes:
        code: Machine-readable failure code.
        source: Source identifier of the image concerned; may be empty.
    """

    category = "image"

    def __init__(
        self,
        message: str,
        code: str,
        source: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initializes an image error with machine-readable context.

        Args:
            message: Human-readable description of the failure.
            code: Machine-readable failure code, stored in the context.
            source: Source identifier of the image; omitted from the context
                when empty.
            details: Extra context entries merged into the error context.
        """
        context = {"code": code}
        if source:
            context["source"] = source
        if details:
            context.update(details)
        super().__init__(message, context)
        self.code = code
        self.source = source


class DependencyError(ImageError):
    """Raised when Pillow is required but is not installed."""

    category = "dependency"

    def __init__(self, operation: str) -> None:
        """Initializes a dependency error for ``operation``.

        Args:
            operation: Name of the operation that needs Pillow.
        """
        super().__init__(
            "image operation requires the optional 'Pillow' dependency; "
            "install it with 'pip install Pillow'",
            code="missing_dependency",
            details={
                "dependency": "Pillow",
                "operation": operation,
                "install": "pip install Pillow",
            },
        )


class ViolationError(ImageError):
    """Raised when an image violates a safety or validity constraint."""
