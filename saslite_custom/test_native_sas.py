import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
from saslite import SasInterpreter
from saslite.storage.sas_backend import SasBackend, _read_sas7bdat
from saslite.runtime.dataset import Dataset

ROOT = Path(os.environ.get('SASLITE_COURSE_FIXTURES', Path(__file__).resolve().parent / 'fixtures/course'))
HAS_COURSE = all((ROOT / name).is_file() for name in ['comptant.sas7bdat', 'food.sas7bdat', 'reponse exercice 2 chapitre 3(1).sas', 'reponse exercice 2 chapitre 3(2).sas', '.saslite.json'])


class NativeSasTests(unittest.TestCase):
    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_both_native_files(self):
        backend = SasBackend(ROOT, libref='TMP1', format='xpt')
        for name, shape in [('comptant', (64,2)), ('food', (68,5))]:
            with self.subTest(name=name):
                native = backend.read(name)
                expected, _ = pyreadstat.read_sas7bdat(str(ROOT / f'{name}.sas7bdat'), disable_datetime_conversion=True)
                pd.testing.assert_frame_equal(native.data, expected)
                self.assertEqual(native.data.shape, shape)
        self.assertEqual(backend.read('COMPTANT').data.iloc[0]['offre'], 62)

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_original_scripts_unchanged_and_files_untouched(self):
        files = list(ROOT.glob('*.sas')) + list(ROOT.glob('*.sas7bdat'))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for name in ['reponse exercice 2 chapitre 3(1).sas', 'reponse exercice 2 chapitre 3(2).sas']:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                sas = SasInterpreter()
                result = sas.execute_file(ROOT / name)
                self.assertTrue(result.success, result.error)
                out = '\n'.join(m for s in result.steps for m in s.output_messages)
                self.assertEqual(out.count('The TTEST Procedure'), 2)
                self.assertEqual(out.count('The NPAR1WAY Procedure'), 2)
                self.assertIn('7.5247', out)
                self.assertIn('22.7671', out)
                self.assertEqual(sas.get_dataset('TMP1', 'COMPTANT').iloc[0]['offre'], 62)
                self.assertEqual(sas.get_dataset('WORK', 'MODIF').iloc[0]['offre'], 180)
                self.assertIn('NPAR1WAY Graphs', out)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_config_relative_to_script_from_other_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with contextlib.redirect_stderr(io.StringIO()):
                    result = SasInterpreter().execute_file(ROOT / 'reponse exercice 2 chapitre 3(1).sas')
                self.assertTrue(result.success, result.error)
                self.assertEqual(list(Path(folder).iterdir()), [])
            finally:
                os.chdir(previous)

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_original_windows_libname_is_overridden(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / '.saslite.json').write_text(json.dumps({'libraries': {'TMP1': str(ROOT)}}))
            script = base / 'unchanged.sas'
            script.write_text('libname tmp1 "C:\\course\\data"; proc means data=tmp1.food; var attraction; run;')
            with contextlib.redirect_stderr(io.StringIO()):
                result = SasInterpreter().execute_file(script)
            self.assertTrue(result.success, result.error)

    def test_invalid_config_is_clear_error_without_new_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / '.saslite.json').write_text(json.dumps({'libraries': {'TMP1': './missing'}}))
            (base / 'test.sas').write_text('data x; a=1; run;')
            with contextlib.redirect_stderr(io.StringIO()):
                result = SasInterpreter().execute_file(base / 'test.sas')
            self.assertFalse(result.success)
            self.assertIn('directory does not exist', result.error)
            self.assertFalse((base / 'missing').exists())

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_native_precedes_xpt_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            shutil.copy2(ROOT / 'comptant.sas7bdat', base / 'CoMpTaNt.SAS7BDAT')
            wrong = pd.DataFrame({'offre': [999.], 'groupe': [0.]})
            pyreadstat.write_xport(wrong, str(base / 'comptant.xpt'))
            wrong.to_csv(base / 'comptant.csv', index=False)
            native = SasBackend(base, format='xpt').read('COMPTANT')
            self.assertEqual(native.nrow, 64)
            self.assertEqual(native.data.iloc[0]['offre'], 62)

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_pandas_fallback(self):
        with patch('saslite.storage.sas_backend.pyreadstat.read_sas7bdat', side_effect=pyreadstat.PyreadstatError('test fallback')):
            native = _read_sas7bdat(ROOT / 'food.sas7bdat')
        self.assertEqual(native.shape, (68,5))
        self.assertEqual(native.iloc[0]['Attraction'], -15.06)

    def test_native_write_is_not_disguised_as_transport(self):
        dataset = Dataset.from_dataframe(pd.DataFrame({'x': [1.,2.]}), name='test')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            with self.assertRaises(NotImplementedError):
                SasBackend(base).write('test', dataset)
            self.assertFalse((base / 'TEST.sas7bdat').exists())
            backend = SasBackend(base, format='xpt')
            backend.write('test', dataset)
            self.assertEqual(backend.read('test').nrow, 2)
            (base / 'comptant.sas7bdat').write_bytes(b'protected native member')
            before = (base / 'comptant.sas7bdat').read_bytes()
            with self.assertRaises(FileExistsError):
                backend.write('comptant', dataset)
            self.assertEqual(before, (base / 'comptant.sas7bdat').read_bytes())

    def test_corrupt_native_is_not_silently_replaced_by_other_format(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'bad.sas7bdat').write_bytes(b'not a SAS dataset')
            pyreadstat.write_xport(pd.DataFrame({'x': [999.]}), str(base / 'bad.xpt'))
            with self.assertRaises(Exception):
                SasBackend(base, format='xpt').read('bad')


class WilcoxonTests(unittest.TestCase):
    def run_proc(self, groups, values, options=''):
        with contextlib.redirect_stderr(io.StringIO()):
            sas = SasInterpreter()
            sas.create_dataset('test', pd.DataFrame({'g': groups, 'x': values}))
            result = sas.execute(f'proc npar1way data=test wilcoxon {options}; class g; var x; run;')
        return result, '\n'.join(m for s in result.steps for m in s.output_messages)

    def test_published_sas_reaction_times(self):
        # SAS NPAR1WAY Example 71.3, approximate output (EXACT not requested).
        first = [1.94]*2 + [2.92]*4 + [3.27]*4 + [3.70]*2 + [3.74]
        second = [3.27]*3 + [3.70]*2 + [3.74]
        result, out = self.run_proc([1]*13 + [2]*6, first + second, 'correct=no')
        self.assertTrue(result.success)
        for value in ['79.5000', '11.004784', '1.7720', '0.0382', '0.0764', '0.0467', '0.0933', '3.1398']:
            self.assertIn(value, out)

    @unittest.skipUnless(HAS_COURSE, "Optional local course fixtures not supplied")
    def test_course_normal_approximation_matches_scipy(self):
        frame, _ = pyreadstat.read_sas7bdat(str(ROOT / 'comptant.sas7bdat'))
        a = frame.loc[frame.groupe == 0, 'offre']
        b = frame.loc[frame.groupe == 1, 'offre']
        for correction in [True, False]:
            result, out = self.run_proc(frame.groupe, frame.offre, 'correct=yes' if correction else 'correct=no')
            self.assertTrue(result.success)
            expected = stats.mannwhitneyu(a, b, method='asymptotic', use_continuity=correction).pvalue
            self.assertIn(f'{expected:.4f}', out)

    def test_multigroup_kruskal_and_missing_observations(self):
        result, out = self.run_proc([1,1,2,2,3,3,None,1], [1,2,4,8,5,9,999,np.nan])
        self.assertTrue(result.success)
        expected = stats.kruskal([1,2], [4,8], [5,9])
        self.assertIn(f'{expected.statistic:.4f}', out)
        self.assertNotIn('Wilcoxon Two-Sample Test', out)

    def test_all_tied_data_and_noprint(self):
        result, out = self.run_proc([1,1,2,2], [4,4,4,4])
        self.assertTrue(result.success)
        self.assertTrue(result.steps[-1].warnings)
        self.assertNotIn('nan', out)
        result, out = self.run_proc([1,1,2,2], [1,2,3,4], 'noprint')
        self.assertTrue(result.success)
        self.assertEqual(out, '')


if __name__ == '__main__':
    unittest.main()
