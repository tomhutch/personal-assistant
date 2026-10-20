import pytest
from unittest.mock import patch
import numpy as np
from command_handler import handle_voice_command

class TestHandleVoiceCommand:
    @patch("command_handler.execute_ha_command")
    @patch("command_handler.parse_intent")
    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")  # Updated to match command_handler.py's import
    def test_scenario_successful_command_execution(
        self, mock_fetch_devices, mock_record, mock_transcribe, mock_parse, mock_exec
    ):
        """Scenario: Valid wake trigger leads to successful recording, transcription, parsing, and HA execution."""
        mock_fetch_devices.return_value = [{"entity_id": "switch.sonoff_s60zbtpg", "name": "Plug 1"}]
        mock_record.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "turn off first plug"
        mock_parse.return_value = {
            "domain": "switch",
            "service": "turn_off",
            "entity_id": "switch.sonoff_s60zbtpg"
        }
        
        handle_voice_command(noise_floor=0.001)
        
        mock_fetch_devices.assert_called_once()
        mock_record.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_parse.assert_called_once()
        mock_exec.assert_called_once_with("switch", "turn_off", "switch.sonoff_s60zbtpg")

    @patch("command_handler.execute_ha_command")
    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")  # Updated to match command_handler.py's import
    def test_scenario_empty_audio_handling(
        self, mock_fetch_devices, mock_record, mock_transcribe, mock_exec
    ):
        """Scenario: User triggers wake word but records zero audio; workflow aborts safely."""
        mock_fetch_devices.return_value = []
        mock_record.return_value = np.array([], dtype=np.float32)
        
        handle_voice_command(noise_floor=0.001)
        
        mock_transcribe.assert_not_called()
        mock_exec.assert_not_called()