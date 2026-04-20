import pathlib
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dreamerv3-main'))

import elements.checkpoint as checkpoint  # noqa: E402
from dreamerv3 import main as dreamer_main  # noqa: E402


_orig_cleanup = checkpoint.Checkpoint._cleanup


def _safe_cleanup(self):
  try:
    _orig_cleanup(self)
  except PermissionError:
    time.sleep(0.5)
    try:
      _orig_cleanup(self)
    except PermissionError:
      print('Skipping checkpoint cleanup on Windows due to file lock.')


checkpoint.Checkpoint._cleanup = _safe_cleanup


if __name__ == '__main__':
  dreamer_main.main(sys.argv[1:])
