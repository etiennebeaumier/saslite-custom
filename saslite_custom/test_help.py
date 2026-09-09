"""Public help navigation, examples, and release-review guard regression tests."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from saslite.cli.help import help_data, render_help, supported_procedures, validate_reference
from saslite.cli.main import build_parser, main, _run_repl
from saslite.functions import build_default_registry
from saslite.parser.program_parser import ProgramParser
from check_help import source_snapshot, check_review


class HelpTests(unittest.TestCase):
    def test_full_guide_includes_live_options_procedures_functions(self):
        validate_reference()
        parser = build_parser()
        guide = render_help('all', parser)
        for action in parser._actions:
            for option in action.option_strings:
                self.assertIn(option, guide)
        for name in supported_procedures():
            self.assertIn('PROC ' + name, guide)
        for name in build_default_registry().names:
            self.assertIn(name, guide)
        for text in ('Std Dev', 'sas7bdat', '--lib', 'Fligner-Policello', 'WORK'):
            self.assertIn(text, guide)

    def test_help_navigation_never_creates_a_session(self):
        for topic in ('', 'topics', 'fp', 'mww', 'libraries', 'proc print', 'function sqrt'):
            with self.subTest(topic=topic), patch('saslite.cli.main.SasInterpreter') as session, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['help', *topic.split()]), 0)
                self.assertTrue(output.getvalue())
                session.assert_not_called()

    def test_aliases_and_unknown_topics(self):
        parser = build_parser()
        self.assertEqual(render_help('fp', parser), render_help('npar1way', parser))
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(['help', 'nonexistent']), 2)
            self.assertIn('help topics', errors.getvalue())
            self.assertEqual(main(['help', 'function', 'nonexistent']), 2)

    def test_brief_help_points_to_guide(self):
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as exc:
            main(['--help'])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn('saslite-custom help', output.getvalue())

    def test_repl_help_preserves_pending_input(self):
        sas = Mock()
        sas.execute.return_value.success = True
        with patch('builtins.input', side_effect=['data demo', 'help fp;', '; x=1; run;', 'exit;']), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(_run_repl(sas), 0)
        sas.execute.assert_called_once_with('data demo\n; x=1; run;\n')
        self.assertIn('Fligner-Policello', output.getvalue())

    def test_procedure_examples_parse(self):
        parser = ProgramParser()
        for name, entry in help_data()['procedures'].items():
            # Each paragraph of indented lines is a standalone SAS example.
            for paragraph in entry['body'].split('\n\n'):
                if paragraph.startswith('    proc '):
                    with self.subTest(procedure=name, example=paragraph):
                        parser.parse(paragraph)

    def test_quickstart_executes(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['-e', 'data demo; x=42; run; proc print data=demo; run;']), 0)

    def test_new_procedure_requires_documentation(self):
        with patch('saslite.cli.help.supported_procedures', return_value=[*supported_procedures(), 'NEWPROC']):
            with self.assertRaisesRegex(ValueError, 'NEWPROC'):
                validate_reference()

    def test_review_guard_detects_edits_additions_and_deletions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'package'
            package.mkdir()
            (package / '__init__.py').write_text('__version__ = "1"\n')
            source = package / 'behavior.py'
            source.write_text('value = 1\n')
            initial = source_snapshot(package)
            review = root / 'review.json'
            with self.assertRaisesRegex(ValueError, 'No help review'):
                check_review(initial, review)
            review.write_text(json.dumps({'files': initial}))
            (package / '__init__.py').write_text('__version__ = "2"\n')
            check_review(source_snapshot(package), review)
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'behavior.py'):
                check_review(source_snapshot(package), review)
            source.unlink()
            with self.assertRaisesRegex(ValueError, 'behavior.py'):
                check_review(source_snapshot(package), review)
            source.write_text('value = 1\n')
            (package / 'new.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'new.py'):
                check_review(source_snapshot(package), review)


if __name__ == '__main__':
    unittest.main()
