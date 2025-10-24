"""Utility helpers for environment checks and shared behaviour."""

from .logging import configure_logging, get_logger
from .placeholders import (
    PLACEHOLDER_NAMESPACE,
    ensure_dir,
    make_blank_json,
    make_silence_wav,
    make_text_placeholder,
)
from .preflight import check_env, ffmpeg_available, platform_summary, python_version
from .report import append_warning, record_stage, start_run, write_report

__all__ = [
    "check_env",
    "ffmpeg_available",
    "platform_summary",
    "python_version",
    "configure_logging",
    "get_logger",
    "PLACEHOLDER_NAMESPACE",
    "ensure_dir",
    "make_blank_json",
    "make_silence_wav",
    "make_text_placeholder",
    "start_run",
    "record_stage",
    "append_warning",
    "write_report",
]
