"""Home Assistant integration module."""

from .service import check_sunlight, get_sunlight_details

__all__ = [
    "check_sunlight",
    "get_sunlight_details",
]
