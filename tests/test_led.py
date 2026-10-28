import sys
import time
import unittest
from enum import Enum
from unittest.mock import MagicMock

# --- 1. MOCK USB & BLINKSTICK ---
mock_usb = MagicMock()
sys.modules['usb'] = mock_usb
sys.modules['usb.core'] = mock_usb.core
sys.modules['usb.util'] = mock_usb.util

mock_bs_module = MagicMock()
mock_device = MagicMock()
mock_bs_module.find_first.return_value = mock_device
mock_bs_module.blinkstick = mock_bs_module

sys.modules['blinkstick'] = mock_bs_module
sys.modules['blinkstick.blinkstick'] = mock_bs_module

# --- 2. IMPORT LED ---
from led import LED, AssistantState


class TestLEDManager(unittest.TestCase):

  def setUp(self):
    mock_device.reset_mock()
    mock_bs_module.find_first.return_value = mock_device
    # Initialize singleton in instant test mode
    LED(test_mode=True)

  def tearDown(self):
    if LED._instance:
      LED._instance.close()

  def test_singleton_pattern(self):
    led1 = LED(test_mode=True)
    led2 = LED(test_mode=True)
    self.assertIs(led1, led2)

  def test_initial_state(self):
    led = LED()
    self.assertEqual(led.get_state(), AssistantState.PASSIVE)

  def test_state_transitions(self):
    led = LED()
    led.set_state(AssistantState.LISTENING)
    self.assertEqual(led.get_state(), AssistantState.LISTENING)

    led.set_state(AssistantState.PROCESSING)
    self.assertEqual(led.get_state(), AssistantState.PROCESSING)

  def test_hardware_called_on_state_change(self):
    led = LED()
    led.set_state(AssistantState.LISTENING)

    # Near-instantaneous check thanks to test_mode=True
    time.sleep(0.02)

    self.assertTrue(mock_device.set_color.called)
    mock_device.set_color.assert_any_call(
        channel=0, index=0, name="deepskyblue"
    )

  def test_error_state_resets_to_passive(self):
    led = LED()
    led.set_state(AssistantState.ERROR)
    self.assertEqual(led.get_state(), AssistantState.ERROR)

    # Error loop completes instantly in test mode
    time.sleep(0.05)

    self.assertEqual(led.get_state(), AssistantState.PASSIVE)


if __name__ == '__main__':
  unittest.main()