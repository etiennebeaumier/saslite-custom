"""CLI main entry point for SASLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saslite.api.facade import SasInterpreter


def build_parser() -> argparse.ArgumentParser:
    """Single source for executable CLI options and their help reference."""
    parser = argparse.ArgumentParser(
        prog="saslite-custom",
        description="SASLite — lightweight local SAS language interpreter",
        epilog="Full guide: saslite-custom help | Topics: saslite-custom help topics",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="SAS script file to execute",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive REPL mode",
    )
    parser.add_argument(
        "-e", "--execute",
        type=str,
        help="Execute a SAS statement directly",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Assign a persistent DISK library; WORK remains in memory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sas7bdat", "xpt"],
        default="sas7bdat",
        help="DISK format: sas7bdat is read-only; choose xpt for persistent writes",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used to read SAS script files",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    parser.add_argument('--no-plots', action='store_true', help='Show statistical tables without terminal plots')
    parser.add_argument('--plot-width', type=int, help='Terminal plot width in columns (40 to 160)')
    parser.add_argument('--lib', action='append', default=[], metavar='NAME=PATH',
                        help='Assign/override a SAS library without changing source; repeat for multiple libraries')
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point, including help without loading a project or dataset."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0].lower() == 'help':
        from saslite.cli.help import show_help
        return show_help(' '.join(argv[1:]) or 'all', parser)
    args = parser.parse_args(argv)
    if args.plot_width is not None and not 40 <= args.plot_width <= 160:
        parser.error('--plot-width must be between 40 and 160')

    if args.version:
        from saslite import __version__
        print(f"SASLite {__version__}")
        return 0

    sas = SasInterpreter(work_dir=args.workdir, sas_format=args.format)
    from saslite.session.project import library_mappings, configure_project
    try:
        mappings = {}
        for assignment in args.lib:
            name, separator, path = assignment.partition('=')
            if not separator:
                raise ValueError('--lib expects NAME=PATH')
            mappings[name] = path
        sas.session.set_option('CLI_LIBRARIES', library_mappings(mappings, Path.cwd()))
        if not args.file:
            configure_project(sas.session, sas.reporter, Path.cwd())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    sas.session.set_option('TERMINAL_PLOTS', not args.no_plots)
    if args.plot_width is not None:
        sas.session.set_option('PLOTWIDTH', args.plot_width)

    if args.execute:
        return _run_text(sas, args.execute)

    if args.file:
        return _run_file(sas, args.file, encoding=args.encoding)

    # Default: REPL mode
    return _run_repl(sas)


def _run_file(sas: SasInterpreter, filepath: str, encoding: str = "utf-8") -> int:
    """Run a SAS script file."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return 1

    try:
        summary = sas.execute_file(path, encoding=encoding)
    except UnicodeDecodeError as exc:
        print(
            f"ERROR: Failed to read {filepath!r} with encoding {encoding!r}: {exc}. "
            "Try --encoding latin1 or --encoding cp1252.",
            file=sys.stderr,
        )
        return 1

    if not summary.success:
        if summary.error:
            print(f"ERROR: {summary.error}", file=sys.stderr)
        for step in summary.steps:
            if step.error:
                print(f"ERROR: {step.error}", file=sys.stderr)
        return 1

    return 0


def _run_text(sas: SasInterpreter, text: str) -> int:
    """Run a SAS statement."""
    summary = sas.execute(text)
    if not summary.success:
        for step in summary.steps:
            if step.error:
                print(f"ERROR: {step.error}", file=sys.stderr)
        return 1
    return 0


def _run_repl(sas: SasInterpreter) -> int:
    """Interactive REPL mode."""
    from saslite import __version__
    print(f"SASLite {__version__} — Interactive Mode")
    print("Type SAS statements. End each with a semicolon (;)")
    print("Type 'help;' for the guide, or 'help fp;' for a topic.")
    print("Type 'quit;' or 'exit;' to leave.\n")

    buffer = ""

    while True:
        try:
            if not buffer:
                prompt = "sas> "
            else:
                prompt = "...  "

            line = input(prompt)

            help_command = line.strip().rstrip(';').strip()
            if help_command.lower() == 'help' or help_command.lower().startswith('help '):
                from saslite.cli.help import show_help
                topic = help_command[4:].strip() or 'all'
                show_help(topic, build_parser())
                continue

            if line.strip().lower() in ("quit;", "exit;", "quit", "exit"):
                break

            buffer += line + "\n"

            # Check if we have a complete statement (ends with ;)
            if ";" in buffer:
                try:
                    summary = sas.execute(buffer)
                    if not summary.success:
                        for step in summary.steps:
                            if step.error:
                                print(f"ERROR: {step.error}")
                except Exception as e:
                    print(f"ERROR: {e}")
                buffer = ""

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            buffer = ""

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
