import argparse
import time
from led import LED, AssistantState


def main():
  parser = argparse.ArgumentParser(
      description="Manually control the BlinkStick Nano voice assistant state."
  )
  parser.add_argument(
      "state",
      type=str.lower,
      choices=["passive", "listening", "processing", "success", "error"],
      help="The target state for the LED indicator.",
  )
  parser.add_argument(
      "-b",
      "--brightness",
      type=float,
      default=None,
      help=(
          "Override global brightness from config.py (value between 0.0 and"
          " 1.0, e.g., 0.05 for 5%)"
      ),
  )
  args = parser.parse_args()

  # Map string inputs to your AssistantState enum
  state_mapping = {
      "passive": AssistantState.PASSIVE,
      "listening": AssistantState.LISTENING,
      "processing": AssistantState.PROCESSING,
      "success": AssistantState.SUCCESS,
      "error": AssistantState.ERROR,
  }

  target_state = state_mapping[args.state]

  print(f"🔌 Connecting to BlinkStick Nano...")
  
  # Pass the optional brightness override into the LED constructor
  leds = LED(test_mode=False, brightness=args.brightness)

  if not leds.bstick:
    print(
        "[Warning] No physical BlinkStick device detected via USB! Check connection."
    )
    return

  brightness_display = (
      f"{args.brightness * 100}%"
      if args.brightness is not None
      else "Config Default"
  )
  print(
      f"✨ Setting state to: [{args.state.upper()}] | Brightness:"
      f" {brightness_display}"
  )

  try:
    leds.set_state(target_state)
    print("💡 LED controller active. Press Ctrl+C to stop and clear LEDs.")
    while True:
      time.sleep(1)

  except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

  finally:
    leds.close()
    print("👋 BlinkStick closed successfully.")


if __name__ == "__main__":
  main()