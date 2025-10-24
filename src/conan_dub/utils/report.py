"""Helpers for recording timing and artifact summaries during pipeline runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def start_run(run_root: Path, seed: int, run_id: str | None = None) -> Dict[str, object]:
    """Initialise a report dictionary and ensure the destination exists."""

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = run_root.name
    return {
        "run_id": run_id,
        "seed": seed,
        "started_at": _utc_now(),
        "stages": [],
        "warnings": [],
    }


def record_stage(
    report: Dict[str, object],
    name: str,
    start_seconds: float,
    end_seconds: float,
    outputs: Mapping[Path | str, int],
    notes: str | None = None,
) -> Dict[str, object]:
    """Append a stage entry to the report."""

    stage_entry: Dict[str, object] = {
        "name": name,
        "seconds": round(float(end_seconds - start_seconds), 4),
        "outputs": [
            {"path": str(path), "bytes": int(size)} for path, size in outputs.items()
        ],
    }
    if notes:
        stage_entry["notes"] = notes
    report.setdefault("stages", []).append(stage_entry)
    return stage_entry


def append_warning(report: Dict[str, object], message: str) -> None:
    """Store a warning message on the report."""

    report.setdefault("warnings", []).append(message)


def write_report(report: Dict[str, object], run_root: Path) -> tuple[Path, Path]:
    """Persist JSON and Markdown versions of the test report."""

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report["completed_at"] = _utc_now()

    json_path = run_root / "report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    md_path = run_root / "report.md"
    lines: List[str] = []
    lines.append(f"# Conan Dub Test Report — {report.get('run_id', 'unknown')}")
    lines.append("")
    lines.append(f"- Seed: `{report.get('seed')}`")
    lines.append(f"- Started: `{report.get('started_at')}`")
    lines.append(f"- Completed: `{report.get('completed_at')}`")
    if report.get("warnings"):
        lines.append("- Warnings:")
        for warning in report["warnings"]:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append("| Stage | Duration (s) | Outputs |")
    lines.append("| --- | ---: | --- |")
    for stage in report.get("stages", []):
        output_descriptions = []
        for output in stage.get("outputs", []):
            output_descriptions.append(
                f"`{output['path']}` ({_human_bytes(int(output['bytes']))})"
            )
        outputs_text = "<br>".join(output_descriptions) if output_descriptions else "—"
        lines.append(
            f"| {stage.get('name')} | {stage.get('seconds')} | {outputs_text} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, md_path


__all__ = [
    "start_run",
    "record_stage",
    "append_warning",
    "write_report",
]
