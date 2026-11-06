import os
import select
import signal
import subprocess
import sys

import pytest


@pytest.mark.parametrize("stuck_cleanup", [False, True])
def test_shutdown_completes_even_if_audio_cleanup_hangs(stuck_cleanup):
    script = """
import threading
from unittest.mock import MagicMock
import main

main.SHUTDOWN_GRACE_SECONDS = 0.2
main.find_input_device = lambda: 0
main.sd.query_devices = lambda *args: {"name": "test input"}
main.LED = MagicMock

class WakeWordManager:
    def listen_for_wake_word(self, buffer, shutdown_event):
        print("ready", flush=True)
        shutdown_event.wait()
        return False, 0.001

class InputStream:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if STUCK_CLEANUP:
            threading.Event().wait()

main.WakeWordManager = WakeWordManager
main.sd.InputStream = InputStream
main.run()
""".replace("STUCK_CLEANUP", repr(stuck_cleanup))
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 15)[0], "Assistant did not start"
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    if stuck_cleanup:
        assert process.returncode == 1
        assert "Shutdown timed out; forcing exit." in stderr
    else:
        assert process.returncode == 0
        assert "Assistant stopped cleanly." in stderr
        assert "Shutdown timed out" not in stderr
