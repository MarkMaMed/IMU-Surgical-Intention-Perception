from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PedalEvent:
    confirm: bool = False
    reset: bool = False
    quit: bool = False
    raw_key: int = -1
    key_name: str = ""


class KeyboardPedal:
    """Keyboard-based fallback for pedal confirmation."""

    def poll(self, key_code: int) -> PedalEvent:
        if key_code == -1:
            return PedalEvent()

        key = key_code & 0xFF
        event = PedalEvent(raw_key=key)

        if key == ord(" "):
            event.confirm = True
            event.key_name = "SPACE"
        elif key in (ord("r"), ord("R")):
            event.reset = True
            event.key_name = "R"
        elif key in (ord("q"), ord("Q"), 27):
            event.quit = True
            event.key_name = "Q/ESC"
        else:
            event.key_name = chr(key) if 32 <= key <= 126 else str(key)

        return event

