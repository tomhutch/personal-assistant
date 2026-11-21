import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_environment_setup(directory):
    script = (ROOT / "setup.sh").read_text()
    section = script.split('echo "=== 7. Preparing .env ==="', 1)[1].split(
        'echo "=== 8. Registering systemd services ==="', 1
    )[0]
    return subprocess.run(
        ["bash", "-eu", "-c", section],
        cwd=directory,
        capture_output=True,
        text=True,
    )


def test_setup_enables_speex_in_new_environment(tmp_path):
    (tmp_path / ".env.example").write_text((ROOT / ".env.example").read_text())

    result = run_environment_setup(tmp_path)

    assert result.returncode == 0, result.stderr
    env_file = tmp_path / ".env"
    assert "WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION=true\n" in env_file.read_text()
    assert os.stat(env_file).st_mode & 0o777 == 0o600


def test_setup_preserves_existing_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION=false\n")

    result = run_environment_setup(tmp_path)

    assert result.returncode == 0, result.stderr
    assert env_file.read_text() == "WAKE_WORD_ENABLE_SPEEX_NOISE_SUPPRESSION=false\n"


def test_setup_rejects_missing_speex_setting(tmp_path):
    (tmp_path / ".env.example").write_text("LOG_LEVEL=INFO\n")

    result = run_environment_setup(tmp_path)

    assert result.returncode != 0
    assert "expected Speex setting not found" in result.stderr
    assert not (tmp_path / ".env").exists()
