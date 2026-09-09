"""Require a reviewed help reference before packaging changed SASLite source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / 'help_review.json'


def source_snapshot(package):
    result = {}
    for source in sorted(package.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
            continue
        data = source.read_bytes()
        if source == package / '__init__.py':
            # Wheel installation changes only this release label.
            data = re.sub(rb'__version__ = "[^"]+"', b'__version__ = "RELEASE"', data)
        result[source.relative_to(package).as_posix()] = hashlib.sha256(data).hexdigest()
    return result


def changed_files(review, current):
    return sorted(name for name in review.keys() | current.keys()
                  if review.get(name) != current.get(name))


def check_review(current, review_path=REVIEW):
    if not review_path.exists():
        raise ValueError('No help review recorded. Review help, then run check_help.py --record.')
    reviewed = json.loads(review_path.read_text())['files']
    changed = changed_files(reviewed, current)
    if changed:
        raise ValueError('Help review is stale for: ' + ', '.join(changed) +
                         '. Update and inspect help, run tests, then use check_help.py --record.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='Record a completed help review after updating docs and testing')
    args = parser.parse_args(argv)
    from saslite.cli.help import PACKAGE, validate_reference
    try:
        validate_reference()
        current = source_snapshot(PACKAGE)
        if args.record:
            REVIEW.write_text(json.dumps({'schema': 1, 'files': current}, indent=2) + '\n')
            print('Help review recorded for current package source.')
        else:
            check_review(current)
            print('Help procedure coverage and source review are current.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
