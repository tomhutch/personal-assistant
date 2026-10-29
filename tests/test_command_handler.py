import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from command_handler import handle_voice_command
from led import AssistantState

class TestHandleVoiceCommand:
    @pytest.fixture(autouse=True)
    def mock_failure_announcement(self, monkeypatch):
        self.announce_problem = MagicMock()
        monkeypatch.setattr("command_handler._announce_problem", self.announce_problem)
        return self.announce_problem

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
        mock_exec.return_value = True
        mock_led = MagicMock()

        input_device = 0
        handle_voice_command(input_device, noise_floor=0.001, led=mock_led)
        
        mock_fetch_devices.assert_called_once()
        mock_record.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_parse.assert_called_once()
        mock_exec.assert_called_once_with("switch", "turn_off", "switch.sonoff_s60zbtpg")
        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.PROCESSING,
            AssistantState.SUCCESS,
        ]

    @patch("command_handler.execute_ha_command")
    @patch("command_handler.parse_intent")
    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")
    def test_failed_home_assistant_command_sets_failure_state(
        self, mock_fetch_devices, mock_record, mock_transcribe, mock_parse, mock_exec
    ):
        mock_fetch_devices.return_value = [{"entity_id": "switch.plug_1", "name": "Plug 1"}]
        mock_record.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "turn on plug 1"
        mock_parse.return_value = {
            "domain": "switch",
            "service": "turn_on",
            "entity_id": "switch.plug_1",
        }
        mock_exec.return_value = False
        mock_led = MagicMock()

        handle_voice_command(0, noise_floor=0.001, led=mock_led)

        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.PROCESSING,
            AssistantState.FAILURE,
        ]

    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")
    def test_recording_exception_logs_and_returns_with_failure_state(
        self, mock_fetch_devices, mock_record, mock_transcribe, caplog
    ):
        mock_fetch_devices.return_value = []
        mock_record.side_effect = RuntimeError("audio input failed")
        mock_led = MagicMock()

        handle_voice_command(0, noise_floor=0.001, led=mock_led)

        mock_transcribe.assert_not_called()
        assert "record_utterance failed" in caplog.text
        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.FAILURE,
        ]

    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")
    def test_device_fetch_exception_logs_and_returns_with_failure_state(
        self, mock_fetch_devices, mock_record, mock_transcribe, caplog
    ):
        mock_fetch_devices.side_effect = RuntimeError("device fetch failed")
        mock_record.return_value = np.zeros(16000, dtype=np.float32)
        mock_led = MagicMock()

        handle_voice_command(0, noise_floor=0.001, led=mock_led)

        mock_transcribe.assert_not_called()
        assert "fetch_ha_devices failed" in caplog.text
        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.FAILURE,
        ]

    @patch("command_handler.parse_intent")
    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")
    def test_device_fetch_failure_result_stops_before_transcription(
        self, mock_fetch_devices, mock_record, mock_transcribe, mock_parse,
        mock_failure_announcement,
    ):
        mock_fetch_devices.return_value = None

        def wait_until_cancelled(input_device, noise_floor, cancel_event):
            assert cancel_event.wait(timeout=1)
            return np.array([], dtype=np.float32)

        mock_record.side_effect = wait_until_cancelled
        mock_led = MagicMock()

        handle_voice_command(0, noise_floor=0.001, led=mock_led)

        mock_transcribe.assert_not_called()
        mock_parse.assert_not_called()
        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.FAILURE,
        ]
        mock_failure_announcement.assert_called_once_with(
            "I couldn't access Home Assistant. Check its connection and access token."
        )

    @patch("command_handler.execute_ha_command")
    @patch("command_handler.parse_intent")
    @patch("command_handler.transcribe_audio")
    @patch("command_handler.record_utterance")
    @patch("command_handler.fetch_ha_devices")
    def test_slm_failure_announces_service_problem(
        self, mock_fetch_devices, mock_record, mock_transcribe, mock_parse,
        mock_exec, mock_failure_announcement,
    ):
        mock_fetch_devices.return_value = [{"entity_id": "switch.plug_1", "name": "Plug 1"}]
        mock_record.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "turn on plug 1"
        mock_parse.return_value = None
        mock_led = MagicMock()

        handle_voice_command(0, noise_floor=0.001, led=mock_led)

        mock_exec.assert_not_called()
        mock_failure_announcement.assert_called_once_with(
            "Sorry, I couldn't contact the language model."
        )
        assert mock_led.set_state.call_args_list[-1].args[0] == AssistantState.FAILURE

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
        mock_led = MagicMock()

        input_device = 0
        handle_voice_command(0, noise_floor=0.001, led=mock_led)
        
        mock_transcribe.assert_not_called()
        mock_exec.assert_not_called()
        assert [call.args[0] for call in mock_led.set_state.call_args_list] == [
            AssistantState.LISTENING,
            AssistantState.FAILURE,
        ]