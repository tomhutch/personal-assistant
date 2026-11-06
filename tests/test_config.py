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
