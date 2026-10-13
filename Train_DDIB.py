"""Dataset-agnostic entry point for the unified all-condition trainer."""

import runpy
from pathlib import Path


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('Train_DDIB_IEMOCAP.py')), run_name='__main__')
