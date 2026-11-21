import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "True"), ("true", "True"), (" YES ", "True"), ("false", "False"),
     ("0", "False"), ("off", "False"), ("", "False")],
)
def test_log_transcripts_environment(value, expected):
    env = os.environ.copy()
    if value is None:
        env.pop("LOG_TRANSCRIPTS", None)
    else:
        env["LOG_TRANSCRIPTS"] = value
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import config; print(config.LOG_TRANSCRIPTS)",
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", None, "0.5"),
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", "1", "1.0"),
        ("WAKE_WORD_VAD_THRESHOLD", None, "0.0"),
        ("WAKE_WORD_VAD_THRESHOLD", "0.6", "0.6"),
        ("WAKE_WORD_VAD_THRESHOLD", "1", "1.0"),
        ("WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION", None, "False"),
        ("WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION", "true", "True"),
        ("WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION", "off", "False"),
    ],
)
def test_wake_word_environment(name, value, expected):
    env = os.environ.copy()
    if value is None:
        env.pop(name, None)
    else:
        env[name] = value
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import config; print(getattr(config, sys.argv[2]))",
            str(Path(__file__).resolve().parents[1] / "src"), name,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", "0"),
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", "-0.1"),
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", "1.1"),
        ("WAKE_WORD_CONFIDENCE_THRESHOLD", "nan"),
        ("WAKE_WORD_VAD_THRESHOLD", "-0.1"),
        ("WAKE_WORD_VAD_THRESHOLD", "1.1"),
        ("WAKE_WORD_VAD_THRESHOLD", "inf"),
    ],
)
def test_invalid_wake_word_threshold_is_rejected(name, value):
    env = os.environ.copy()
    env[name] = value
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import config",
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert name in result.stderr


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "0.5"), ("0", "0.0"), ("0.3", "0.3"), ("1", "1.0")],
)
def test_command_vad_threshold_environment(value, expected):
    env = os.environ.copy()
    if value is None:
        env.pop("COMMAND_VAD_THRESHOLD", None)
    else:
        env["COMMAND_VAD_THRESHOLD"] = value
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import config; print(config.COMMAND_VAD_THRESHOLD)",
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_invalid_command_vad_threshold_is_rejected(value):
    env = os.environ.copy()
    env["COMMAND_VAD_THRESHOLD"] = value
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import config",
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "COMMAND_VAD_THRESHOLD" in result.stderr
