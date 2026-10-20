import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from transcriber import transcribe_audio

class TestTranscribeAudio:
    @patch("requests.post")
    def test_scenario_successful_transcription(self, mock_post):
        """Scenario: Whisper API successfully receives audio and returns transcript text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "turn off first plug"}
        mock_post.return_value = mock_response
        
        dummy_audio = np.zeros(16000, dtype=np.float32)
        result = transcribe_audio(dummy_audio)
        
        assert result == "turn off first plug"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_scenario_api_error_fallback(self, mock_post):
        """Scenario: Whisper API returns an error status; transcriber handles it gracefully by returning empty string."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        dummy_audio = np.zeros(16000, dtype=np.float32)
        result = transcribe_audio(dummy_audio)
        
        assert isinstance(result, str)
        assert result == ""