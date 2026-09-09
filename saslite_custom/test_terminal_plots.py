import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from saslite import SasInterpreter
from saslite.runtime.terminal_plots import box_summary, histogram_data, render_ttest_plots


class TerminalPlotTests(unittest.TestCase):
    def execute(self, code, frame=None):
        with contextlib.redirect_stderr(io.StringIO()):
            sas = SasInterpreter()
            sas.create_dataset('test', frame if frame is not None else pd.DataFrame({'g': [1,1,1,2,2,2], 'x': [1,2,4,3,5,9]}))
            result = sas.execute(code)
        return result, '\n'.join(m for s in result.steps for m in s.output_messages)

    def test_default_panels_and_selection(self):
        result, out = self.execute('proc ttest data=test; class g; var x; run;')
        self.assertTrue(result.success)
        for title in ['Histograms and Density Curves', 'Box Plots', 'Normal Q-Q Plots', 'Confidence Intervals for the Mean']:
            self.assertIn(title, out)
        result, out = self.execute('proc ttest data=test plots=(box qqplot); class g; var x; run;')
        self.assertTrue(result.success)
        self.assertIn('Box Plots', out)
        self.assertIn('Normal Q-Q Plots', out)
        self.assertNotIn('Histograms and Density Curves', out)

    def test_sas_controls_and_state(self):
        for options in ['plots=none', 'noprint']:
            result, out = self.execute(f'proc ttest data=test {options}; var x; run;')
            self.assertTrue(result.success)
            self.assertNotIn('TTEST Graphs:', out)
        result, out = self.execute('ods graphics off; proc ttest data=test; var x; run; ods graphics on; proc ttest data=test; var x; run;')
        self.assertTrue(result.success)
        self.assertEqual(out.count('TTEST Graphs:'), 1)

    def test_bad_plot_option_is_rejected(self):
        for option in ['banana', '(none box)']:
            result, _ = self.execute(f'proc ttest data=test plots={option}; var x; run;')
            self.assertFalse(result.success)

    def test_paired_graphs_use_complete_pairs(self):
        frame = pd.DataFrame({'a': [2,4,7,11,500], 'b': [1,2,3,4,None]})
        result, out = self.execute('proc ttest data=test; paired a*b; run;', frame)
        self.assertTrue(result.success)
        self.assertIn('Profiles Plot (N=4 complete pairs)', out)
        self.assertIn('Agreement Plot (N=4 complete pairs)', out)
        self.assertIn('TTEST Graphs: A - B', out)
        self.assertNotIn('N=5', out)

    def test_histogram_denominators_and_common_edges(self):
        samples = [np.array([1,2,3,100]), np.array([2,2,3,4,5,6])]
        edges, percentages = histogram_data(samples, 80)
        self.assertLessEqual(edges[0], 1)
        self.assertGreaterEqual(edges[-1], 100)
        for sample, percent in zip(samples, percentages):
            self.assertAlmostEqual(sum(percent), 100)
            np.testing.assert_allclose(percent * len(sample) / 100, np.histogram(sample, edges)[0])

    def test_sas_quartiles_and_outliers(self):
        q1, med, q3, low, high, outliers = box_summary([1,2,3,4,5,6,7,100])
        self.assertEqual((q1,med,q3,low,high), (2.5,4.5,6.5,1,7))
        np.testing.assert_array_equal(outliers, [100])

    def test_width_ascii_and_reference_visibility(self):
        for width in [40,80,120,160]:
            with self.subTest(width=width):
                out = render_ttest_plots([np.array([-10,-5,0,3,4,100])], ['A'*90], 'A long variable label',
                      [('Mean', -2., -5., .001)], 0, .05, width=width)
                self.assertTrue(out.isascii())
                self.assertLessEqual(max(map(len, out.splitlines())), width)
                self.assertIn(':\n', out)
                self.assertIn('Normal Quantiles', out)

    def test_constant_sample_has_box_and_qq_without_density_failure(self):
        out = render_ttest_plots([np.array([5,5,5])], ['X'], 'X', [('X',5,5,5)], 0, .05)
        self.assertIn('Constant sample: density fits unavailable.', out)
        self.assertIn('Box Plots', out)
        self.assertIn('Normal Q-Q Plots', out)
        self.assertNotIn('nan', out)

    def test_no_plots_skips_calculation(self):
        with patch('saslite.runtime.terminal_plots.render_ttest_plots', side_effect=AssertionError('must not render')):
            result, _ = self.execute('proc ttest data=test plots=none; var x; run;')
            self.assertTrue(result.success)

    def test_nonfinite_plot_input_is_reported(self):
        out = render_ttest_plots([np.array([1,2,np.inf])], ['X'], 'X', [], 0, .05)
        self.assertIn('Graphs unavailable', out)


if __name__ == '__main__':
    unittest.main()
