"""CLI entry: 4-panel figure (original, bilinear, bicubic, SEN2SR). See ``opensr_fourway.py``."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opensr_fourway import main  # noqa: E402

if __name__ == "__main__":
    main()
