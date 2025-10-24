from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from conan_dub.cli.conan_dub import app

runner = CliRunner()


def _write_sine_stub(path: Path, duration: float = 0.2, sample_rate: int = 16000) -> None:
    frame_count = int(duration * sample_rate)
    with wave.open(str(path), "w") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(b"\x00\x00" * frame_count)


def test_split_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    out_dir = tmp_path / "stems"

    result = runner.invoke(
        app,
        [
            "split",
            str(source),
            "--out",
            str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert source.exists()
    assert (out_dir / "voice.wav").exists()
    assert (out_dir / "background.wav").exists()


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_diagnostics() -> None:
    result = runner.invoke(app, ["diagnostics", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "version=" in result.output
    assert "ffmpeg=" in result.output


def test_cli_pipeline_dry_run(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    _write_sine_stub(voice)

    aligned = tmp_path / "aligned.json"
    aligned_payload = {
        "segments": [
            {
                "id": "seg-000",
                "start": 0.0,
                "end": 1.0,
                "text": "テスト",
                "speaker": "spk-001",
                "ctm": [(0.0, 0.5, "テ"), (0.5, 1.0, "スト")],
            }
        ]
    }
    aligned.write_text(json.dumps(aligned_payload), encoding="utf-8")

    diar = tmp_path / "diar.json"
    diar_payload = {
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker_id": "spk-001", "embedding": [0.0, 0.1, -0.1]}
        ]
    }
    diar.write_text(json.dumps(diar_payload), encoding="utf-8")

    ser_out = tmp_path / "ser.json"
    result = runner.invoke(
        app,
        [
            "ser",
            str(voice),
            "--aligned",
            str(aligned),
            "--out",
            str(ser_out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ser_out.exists()

    faces = tmp_path / "faces.json"
    faces.write_text(json.dumps({"detections": []}), encoding="utf-8")

    manifest_out = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "fuse",
            "--aligned",
            str(aligned),
            "--diar",
            str(diar),
            "--ser-path",
            str(ser_out),
            "--faces",
            str(faces),
            "--out",
            str(manifest_out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert payload["items"]

    translated = tmp_path / "manifest_it.json"
    result = runner.invoke(
        app,
        [
            "translate",
            str(manifest_out),
            "--out",
            str(translated),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    translated_payload = json.loads(translated.read_text(encoding="utf-8"))
    assert translated_payload["items"]

    tts_dir = tmp_path / "tts"
    result = runner.invoke(
        app,
        [
            "synth",
            str(translated),
            "--out",
            str(tts_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tts_dir / "seg-000.wav").exists()

    mix_path = tmp_path / "mix.wav"
    result = runner.invoke(
        app,
        [
            "mix",
            str(tts_dir),
            "--out",
            str(mix_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mix_path.exists()
