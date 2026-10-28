import logging
import sys
import threading
import time
from enum import Enum
from blinkstick import blinkstick

logger = logging.getLogger(__name__)

# Try loading settings from config, with safe defaults
try:
  from config import LED_BRIGHTNESS, LED_ENABLED
except ImportError:
  LED_ENABLED = True
  LED_BRIGHTNESS = 0.1


class AssistantState(Enum):
  PASSIVE = "passive"
  LISTENING = "listening"
  PROCESSING = "processing"
  SUCCESS = "success"
  FAILURE = "failure"
  ERROR = "failure"


COLOR_MAP = {
    "black": (0, 0, 0),
    "deepskyblue": (0, 191, 255),
    "purple": (160, 32, 240),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
}


class LED:
  _instance = None
  _lock = threading.Lock()

  def __new__(cls, test_mode=None, brightness=None, enabled=None):
    with cls._lock:
      if cls._instance is None:
        cls._instance = super(LED, cls).__new__(cls)
        cls._instance._initialize(
            test_mode=test_mode if test_mode is not None else False,
            brightness=brightness if brightness is not None else LED_BRIGHTNESS,
            enabled=enabled if enabled is not None else LED_ENABLED,
        )
      else:
        if test_mode is not None:
          cls._instance.test_mode = test_mode
        if brightness is not None:
          cls._instance.brightness = brightness
        if enabled is not None:
          cls._instance.enabled = enabled
    return cls._instance

  def _initialize(
      self, test_mode=False, brightness=LED_BRIGHTNESS, enabled=LED_ENABLED
  ):
    self.test_mode = test_mode
    self.brightness = max(0.0, min(1.0, brightness))
    self.enabled = enabled
    self.bstick = None

    if self.enabled:
      try:
        self.bstick = blinkstick.find_first()
        if not self.bstick:
          logger.warning(
              "No BlinkStick Nano visible to %s. Check the USB connection, "
              "libusb installation, and this process's USB permissions; "
              "continuing without LED hardware.",
              sys.executable,
          )
      except Exception as e:
        logger.warning("Could not initialize BlinkStick USB: %s", e, exc_info=True)
    else:
      logger.info("LEDs are disabled via config.py.")

    self._current_state = AssistantState.PASSIVE
    self._state_lock = threading.Lock()
    self._running = True

    # Start background animation thread
    self._thread = threading.Thread(target=self._animation_loop, daemon=True)
    self._thread.start()

  def _sleep(self, seconds: float):
    if self.test_mode:
      time.sleep(0.001)
    else:
      time.sleep(seconds)

  def set_state(self, state: AssistantState):
    with self._state_lock:
      if self._current_state != state:
        self._current_state = state

  def get_state(self) -> AssistantState:
    with self._state_lock:
      return self._current_state

  def _set_both(self, name: str):
    # Skip hardware calls entirely if LEDs are disabled or no device found
    if not self.enabled or not self.bstick:
      return

    try:
      base_rgb = COLOR_MAP.get(name, (255, 255, 255))

      if name == "black":
        r, g, b = 0, 0, 0
      else:
        r = int(base_rgb[0] * self.brightness)
        g = int(base_rgb[1] * self.brightness)
        b = int(base_rgb[2] * self.brightness)

      self.bstick.set_color(channel=0, index=0, red=r, green=g, blue=b)
      self.bstick.set_color(channel=0, index=1, red=r, green=g, blue=b)
    except Exception as e:
      logger.critical("CRITICAL LED ERROR: Failed to set color: %s", e, exc_info=True)

  def _animation_loop(self):
    while self._running:
      state = self.get_state()

      if state == AssistantState.PASSIVE:
        self._set_both("black")
        self._sleep(0.1)

      elif state == AssistantState.LISTENING:
        self._set_both("deepskyblue")
        self._sleep(0.2)

      elif state == AssistantState.PROCESSING:
        self._set_both("purple")
        self._sleep(1.0)
        self._set_both("black")
        self._sleep(1.0)

      elif state == AssistantState.SUCCESS:
        for _ in range(2):
          if self.get_state() != AssistantState.SUCCESS:
            break
          self._set_both("green")
          self._sleep(0.15)
          self._set_both("black")
          self._sleep(0.15)
        self.set_state(AssistantState.PASSIVE)

      elif state == AssistantState.FAILURE:
        for _ in range(2):
          if self.get_state() != AssistantState.FAILURE:
            break
          self._set_both("red")
          self._sleep(0.15)
          self._set_both("black")
          self._sleep(0.15)
        self.set_state(AssistantState.PASSIVE)

  def close(self):
    self._running = False
    self._set_both("black")
    if self._thread.is_alive():
      self._thread.join()
    LED._instance = None