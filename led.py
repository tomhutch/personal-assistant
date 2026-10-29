import math
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

LISTENING_CYCLE_SECONDS = 2.0
LISTENING_FRAME_SECONDS = 0.02


def _configure_libusb_backend():
  try:
    import libusb_package
    from usb.backend import libusb1
  except ImportError:
    logger.debug("libusb-package unavailable; using system libusb discovery.")
    return

  if libusb1.get_backend(find_library=libusb_package.find_library) is None:
    logger.warning("Could not load the libusb-package backend; using system discovery.")


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
        _configure_libusb_backend()
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

  def _listening_brightness(self, elapsed: float) -> float:
    cycle_position = (elapsed % LISTENING_CYCLE_SECONDS) / LISTENING_CYCLE_SECONDS
    curve = (1 + math.cos(2 * math.pi * cycle_position)) / 2
    return self.brightness * curve

  def _set_both(self, name: str, brightness=None):
    # Skip hardware calls entirely if LEDs are disabled or no device found
    if not self.enabled or not self.bstick:
      return

    try:
      base_rgb = COLOR_MAP.get(name, (255, 255, 255))

      if name == "black":
        r, g, b = 0, 0, 0
      else:
        brightness = self.brightness if brightness is None else brightness
        r = int(base_rgb[0] * brightness)
        g = int(base_rgb[1] * brightness)
        b = int(base_rgb[2] * brightness)

      for index in (0, 1):
        try:
          self.bstick.set_color(channel=0, index=index, red=r, green=g, blue=b)
        except Exception as e:
          if type(e).__name__ == "USBError" and getattr(e, "errno", None) == 5:
            logger.debug("BlinkStick USB EIO while setting LED %d: %s", index, e)
          else:
            logger.error(
                "Failed to set BlinkStick LED %d: %s", index, e, exc_info=True
            )
    except Exception:
      logger.exception("Unexpected error preparing BlinkStick color %r.", name)

  def _animation_loop(self):
    while self._running:
      state = self.get_state()

      if state == AssistantState.PASSIVE:
        self._set_both("black")
        self._sleep(0.1)

      elif state == AssistantState.LISTENING:
        cycle_started = time.monotonic()
        while self._running and self.get_state() == AssistantState.LISTENING:
          elapsed = time.monotonic() - cycle_started
          brightness = self._listening_brightness(elapsed)
          self._set_both("deepskyblue", brightness=brightness)
          self._sleep(LISTENING_FRAME_SECONDS)

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