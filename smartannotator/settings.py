"""Per-user preferences (last folder, model choice, ...), stored as JSON."""
import json
import os

DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".smartannotator", "settings.json")


class Settings:
    """A dict-like store that writes through to disk. Failing to save is
    never fatal: preferences are a convenience."""

    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        try:
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (OSError, ValueError):
            self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, **values):
        self._data.update(values)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass
