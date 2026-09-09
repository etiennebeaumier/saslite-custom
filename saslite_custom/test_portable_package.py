"""Portable resources and native reader dispatch without private datasets."""
import json
from importlib.resources import files
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from saslite.storage.sas_backend import SasBackend, _read_sas7bdat


class PortablePackageTests(unittest.TestCase):
    def test_installed_resources_are_available(self):
        package = files('saslite')
        guide = json.loads(package.joinpath('cli/help_data.json').read_text(encoding='utf-8'))
        self.assertIn('installation', guide['topics'])
        self.assertTrue(package.joinpath('parser/grammar/saslite.lark').read_text(encoding='utf-8'))
        self.assertTrue(package.joinpath('gui/static/index.html').is_file())

    def test_readstat_preserves_dates_and_metadata(self):
        frame = pd.DataFrame({'day': [0., 365.]})
        meta = SimpleNamespace(column_names_to_labels={'day': 'SAS day'}, original_variable_types={'day': 'DATE9'})
        with patch('saslite.storage.sas_backend.pyreadstat.read_sas7bdat', return_value=(frame, meta)) as reader:
            result = _read_sas7bdat(Path('synthetic.sas7bdat'))
        reader.assert_called_once_with('synthetic.sas7bdat', disable_datetime_conversion=True)
        self.assertEqual(result.day.tolist(), [0., 365.])
        self.assertEqual(result.attrs['sas_labels'], {'day': 'SAS day'})
        self.assertEqual(result.attrs['sas_formats'], {'day': 'DATE9'})

    def test_native_member_priority_and_case_without_course_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / 'SaMpLe.SAS7BDAT'
            native.write_bytes(b'synthetic reader stub')
            (root / 'sample.csv').write_text('x\n999\n')
            with patch('saslite.storage.sas_backend._read_sas7bdat', return_value=pd.DataFrame({'x': [42.]})) as reader:
                result = SasBackend(root, format='xpt').read('SAMPLE')
            reader.assert_called_once()
            self.assertTrue(reader.call_args.args[0].samefile(native))
            self.assertEqual(result.data.iloc[0, 0], 42.)
