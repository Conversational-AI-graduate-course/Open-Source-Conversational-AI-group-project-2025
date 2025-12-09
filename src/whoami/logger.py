import time
import os
import json

class InteractionLogger:
    """File logger for prompts, outputs, and user replies."""

    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(logs_dir, f"log-{ts}.txt")
        self.file = open(self.path, "a", encoding="utf-8", buffering=4096)
        self.t0 = time.monotonic()
        self._line_count = 0

    def log_line(self, text: str):
        elapsed = int(time.monotonic() - self.t0)
        stamp = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        try:
            self.file.write(f"{stamp} {text.rstrip()}\n")
            self._line_count += 1
            if self._line_count % 10 == 0:
                self.file.flush()
        except Exception:
            pass

    def log_turn(self, turn_key: str, payload: dict):
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.log_line(f"{turn_key}:{compact}")

    def close(self):
        try:
            self.log_line("[logger] closing")
            self.file.close()
        except Exception:
            pass