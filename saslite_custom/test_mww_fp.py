import contextlib
import io
import math
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from saslite import SasInterpreter
from saslite.executor.proc.npar1way import fligner_policello


class MwwFpTests(unittest.TestCase):
    def run_proc(self, options, frame=None, variables='x'):
        if frame is None:
            frame = pd.DataFrame({'g': ['A']*4 + ['B']*5, 'x': [1,2,2,7,2,3,4,6,8]})
        with contextlib.redirect_stderr(io.StringIO()):
            sas = SasInterpreter()
            sas.create_dataset('test', frame)
            result = sas.execute(f'proc npar1way data=test {options}; class g; var {variables}; run;')
        return result, '\n'.join(m for step in result.steps for m in step.output_messages)

    def test_nist_published_example(self):
        # https://www.itl.nist.gov/div898/software/dataplot/data/SHOEMAKE.DAT
        x = [4.81,5.71,4.90,5.35,5.26,6.26,3.76,8.07,8.79,7.30]
        y = [5.17,6.17,6.26,4.26,3.17,3.76,4.76,4.90,6.57,5.17]
        result = fligner_policello([x,y])
        self.assertAlmostEqual(result['z'], 1.55802, places=5)
        self.assertAlmostEqual(result['two_sided'], .11923, places=5)
        self.assertAlmostEqual(result['one_sided'], .05961, places=5)

    def test_placements_match_direct_pairwise_definition(self):
        x, y = np.array([1,2,2,7]), np.array([2,3,4,6,8])
        px = np.array([sum(float(b < a) + .5 * float(b == a) for b in y) for a in x])
        py = np.array([sum(float(a < b) + .5 * float(a == b) for a in x) for b in y])
        result = fligner_policello([x,y])
        np.testing.assert_array_equal(result['placements'][0], px)
        np.testing.assert_array_equal(result['placements'][1], py)
        denominator = 2 * math.sqrt(sum((px-px.mean())**2) + sum((py-py.mean())**2) + px.mean()*py.mean())
        self.assertAlmostEqual(result['z'], (px.sum()-py.sum())/denominator)
        self.assertEqual(px.sum()+py.sum(), len(x)*len(y))

    def test_mww_alias_and_u_match_wilcoxon_and_scipy(self):
        r1, canonical = self.run_proc('wilcoxon plots=none')
        r2, alias = self.run_proc('mww plots=none')
        self.assertTrue(r1.success and r2.success)
        self.assertEqual(canonical, alias)
        scipy = stats.mannwhitneyu([1,2,2,7], [2,3,4,6,8], method='asymptotic')
        self.assertIn(f'{scipy.statistic:.4f}', alias)
        self.assertIn(f'{scipy.pvalue:.4f}', alias)
        self.assertIn('Mann-Whitney U', alias)

    def test_fp_only_and_combined_test_selection(self):
        result, out = self.run_proc('fp')
        self.assertTrue(result.success)
        self.assertIn('Fligner-Policello Test', out)
        self.assertNotIn('Wilcoxon Scores', out)
        self.assertNotIn('Kruskal-Wallis Test', out)
        result, out = self.run_proc('mww fp')
        self.assertTrue(result.success)
        self.assertIn('Mann-Whitney U', out)
        self.assertIn('Fligner-Policello Test', out)

    def test_reference_position_and_quoted_class(self):
        _, default = self.run_proc('fp plots=none')
        result, first = self.run_proc('fp(refclass=1) plots=none')
        result2, named = self.run_proc("fp(refclass='A') plots=none")
        self.assertTrue(result.success and result2.success)
        self.assertEqual(first, named)
        self.assertIn('Reference class X: A', first)
        self.assertIn('Reference class X: B', default)
        a = fligner_policello([[1,2,2,7], [2,3,4,6,8]], reference=0)
        b = fligner_policello([[1,2,2,7], [2,3,4,6,8]], reference=1)
        self.assertEqual(a['z'], -b['z'])
        self.assertEqual(a['two_sided'], b['two_sided'])
        self.assertIn(f"{a['z']:.4f}", first)

    def test_numeric_class_value_is_distinct_from_position(self):
        frame = pd.DataFrame({'g': [2,2,2,1,1,1], 'x': [1,3,6,2,4,5]})
        r1, first = self.run_proc('fp(refclass=1)', frame)
        r2, by_label = self.run_proc("fp(refclass='2')", frame)
        self.assertTrue(r1.success and r2.success)
        self.assertEqual(first, by_label)
        self.assertIn('Reference class X: 2', first)

    def test_missing_rows_and_multiple_variables(self):
        frame = pd.DataFrame({'g': ['A','A','A','B','B','B',None],
                              'x': [1,2,np.nan,2,4,5,999], 'y': [4,5,7,3,np.nan,8,999]})
        result, out = self.run_proc('fp plots=none', frame, 'x y')
        self.assertTrue(result.success)
        self.assertEqual(out.count('Fligner-Policello Test'), 2)
        for samples in [[[1,2],[2,4,5]], [[4,5,7],[3,8]]]:
            expected = fligner_policello(samples)
            self.assertIn(f"{expected['z']:.4f}", out)

    def test_invalid_reference_or_group_sizes(self):
        for options in ['fp(refclass=3)', "fp(refclass='missing')"]:
            result, _ = self.run_proc(options)
            self.assertFalse(result.success)
        for frame in [pd.DataFrame({'g': [1,2,2], 'x': [1,2,3]}),
                      pd.DataFrame({'g': [1,1,2,2,3,3], 'x': [1,2,3,4,5,6]})]:
            result, _ = self.run_proc('fp', frame)
            self.assertFalse(result.success)

    def test_degenerate_samples_report_undefined(self):
        result, out = self.run_proc('fp', pd.DataFrame({'g': [1,1,2,2], 'x': [1,1,9,9]}))
        self.assertTrue(result.success)
        self.assertTrue(result.steps[-1].warnings)
        self.assertNotIn('nan', out)
        self.assertNotIn('inf', out)
        all_ties = fligner_policello([[4,4],[4,4,4]])
        self.assertEqual(all_ties['z'], 0)
        self.assertEqual(all_ties['two_sided'], 1)

    def test_plot_selection_and_output_suppression(self):
        result, out = self.run_proc('wilcoxon fp plots=(fpboxplot)')
        self.assertTrue(result.success)
        graphs = out.split('NPAR1WAY Graphs:')[1]
        self.assertIn('Fligner-Policello Placements', graphs)
        self.assertNotIn('Wilcoxon Scores (MWW)', graphs)
        result, out = self.run_proc('fp plots=none')
        self.assertTrue(result.success)
        self.assertNotIn('NPAR1WAY Graphs:', out)
        result, out = self.run_proc('wilcoxon fp noprint')
        self.assertTrue(result.success)
        self.assertEqual(out, '')


if __name__ == '__main__':
    unittest.main()
