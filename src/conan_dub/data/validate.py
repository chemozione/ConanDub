"""JSON Schema validation for dubbing manifests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError

logger = logging.getLogger(__name__)

MANIFEST_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "start",
                    "end",
                    "speaker_id",
                    "character",
                    "emotion",
                    "text_ja",
                    "audio_path",
                    "speaker_embedding",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker_id": {"type": "string"},
                    "character": {"type": "string"},
                    "emotion": {"type": "string"},
                    "text_ja": {"type": "string"},
                    "text_it": {"type": ["string", "null"]},
                    "audio_path": {"type": "string"},
                    "speaker_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

_VALIDATOR = Draft202012Validator(MANIFEST_JSON_SCHEMA)


@dataclass(slots=True)
class ValidationResult:
    """Represents manifest schema validation output."""

    ok: bool
    errors: List[str]


def validate_manifest_payload(payload: Dict[str, Any]) -> ValidationResult:
    """Validate a manifest payload and report errors."""

    errors = sorted(_VALIDATOR.iter_errors(payload), key=lambda err: err.path)
    if not errors:
        return ValidationResult(ok=True, errors=[])
    messages = [f"{list(error.path)}: {error.message}" for error in errors]
    return ValidationResult(ok=False, errors=messages)


def validate_manifest_file(path: Path) -> ValidationResult:
    """Load and validate a manifest JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = validate_manifest_payload(payload)
    if not result.ok:
        for message in result.errors:
            logger.warning("Manifest validation error (%s): %s", path.name, message)
    return result

