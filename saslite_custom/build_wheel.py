"""Build portable wheel and source archive only after the required help review."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / 'saslite_custom/check_help.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-m', 'build', str(ROOT)], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
