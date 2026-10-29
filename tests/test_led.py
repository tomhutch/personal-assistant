import sys
import time
import unittest
from unittest.mock import MagicMock, patch

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
    self.find_first_patcher = patch("led.blinkstick.find_first", return_value=mock_device)
    self.find_first_patcher.start()
    self.addCleanup(self.find_first_patcher.stop)
    # Initialize singleton in instant test mode
    LED(test_mode=True)

  def tearDown(self):
    mock_device.set_color.side_effect = None
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

  def test_listening_brightness_follows_cosine_cycle(self):
    led = LED(brightness=0.1)

    self.assertAlmostEqual(led._listening_brightness(0.0), 0.1)
    self.assertAlmostEqual(led._listening_brightness(0.5), 0.05)
    self.assertAlmostEqual(led._listening_brightness(1.0), 0.0)
    self.assertAlmostEqual(led._listening_brightness(1.5), 0.05)
    self.assertAlmostEqual(led._listening_brightness(2.0), 0.1)

  def test_hardware_called_on_state_change(self):
    led = LED()
    led.set_state(AssistantState.LISTENING)

    # Near-instantaneous check thanks to test_mode=True
    time.sleep(0.02)

    self.assertTrue(mock_device.set_color.called)
    red, green, blue = (int(value * led.brightness) for value in (0, 191, 255))
    mock_device.set_color.assert_any_call(
      channel=0, index=0, red=red, green=green, blue=blue
    )

  def test_color_write_failure_is_logged(self):
    led = LED()
    led._running = False
    led._thread.join()
    mock_device.set_color.reset_mock()
    mock_device.set_color.side_effect = PermissionError("USB access denied")

    with self.assertLogs("led", level="ERROR") as captured:
      led._set_both("red")

    self.assertIn("Failed to set BlinkStick LED 0: USB access denied", captured.output[0])
    self.assertIn("Failed to set BlinkStick LED 1: USB access denied", captured.output[1])
    self.assertEqual(mock_device.set_color.call_count, 2)
    mock_device.set_color.side_effect = None

  def test_usb_eio_is_debug_logged_and_other_led_is_still_written(self):
    led = LED()
    led._running = False
    led._thread.join()
    mock_device.set_color.reset_mock()

    class USBError(OSError):
      pass

    mock_device.set_color.side_effect = [USBError(5, "Input/output error"), None]

    with self.assertLogs("led", level="DEBUG") as captured:
      led._set_both("red")

    self.assertIn("BlinkStick USB EIO while setting LED 0", captured.output[0])
    self.assertEqual(mock_device.set_color.call_count, 2)
    self.assertEqual(mock_device.set_color.call_args_list[1].kwargs["index"], 1)
    mock_device.set_color.side_effect = None

  def test_failure_state_resets_to_passive(self):
    led = LED()
    led.set_state(AssistantState.FAILURE)
    self.assertEqual(led.get_state(), AssistantState.FAILURE)

    # Error loop completes instantly in test mode
    time.sleep(0.05)

    self.assertEqual(led.get_state(), AssistantState.PASSIVE)


if __name__ == '__main__':
  unittest.main()