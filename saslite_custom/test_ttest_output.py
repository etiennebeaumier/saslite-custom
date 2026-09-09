"""Regression checks for the local SASLite TTEST patch. Run with unittest."""
import contextlib
import io
from pathlib import Path
import unittest

import pandas as pd
from scipy import stats
from saslite import SasInterpreter


class TTestOutputTests(unittest.TestCase):
    def run_proc(self, frame, statements, options=""):
        with contextlib.redirect_stderr(io.StringIO()):
            sas = SasInterpreter()
            sas.create_dataset("test", frame)
            result = sas.execute(f"proc ttest data=test {options}; {statements} run;")
        output = "\n".join(m for step in result.steps for m in step.output_messages)
        return result, output

    def test_official_sas_golf_example(self):
        # SAS/STAT 9.3 TTEST, Comparing Group Means, Figures 95.4–95.7.
        frame = pd.DataFrame({"gender": ["f"] * 7 + ["m"] * 7,
                              "score": [75, 76, 80, 77, 80, 77, 73, 82, 80, 85, 85, 78, 87, 82]})
        result, out = self.run_proc(frame, "class gender; var score;")
        self.assertTrue(result.success)
        for expected in ["Std Dev", "Std Err", "Minimum", "Maximum", "76.8571", "2.5448", "0.9619",
                         "82.7143", "3.1472", "1.1895", "-5.8571", "2.8619", "1.5298",
                         "74.5036", "79.2107", "1.6399", "5.6039", "-9.1902", "-2.5241",
                         "-9.2064", "-2.5078", "11.496", "0.0024", "0.0026", "Folded F", "0.6189"]:
            self.assertIn(expected, out)

    def test_one_sample_uses_student_t_and_alpha(self):
        values = [1, 2, 4, 8]
        result, out = self.run_proc(pd.DataFrame({"x": values}), "var x;", "alpha=.10 h0=2")
        self.assertTrue(result.success)
        self.assertIn("90% CL Mean", out)
        sample = pd.Series(values)
        lower = sample.mean() - stats.t.ppf(.95, 3) * sample.sem()
        self.assertIn(f"{lower:.4f}", out)
        self.assertIn(f"{stats.ttest_1samp(values, 2).pvalue:.4f}", out)

    def test_paired_complete_cases_and_nonzero_null(self):
        frame = pd.DataFrame({"a": [4, 7, 9, 11, 50], "b": [1, 2, 3, 4, None]})
        result, out = self.run_proc(frame, "paired a*b;", "h0=2")
        self.assertTrue(result.success)
        differences = pd.Series([3, 5, 6, 7])
        self.assertIn(f"{differences.std():.4f}", out)
        self.assertIn(f"{stats.ttest_1samp(differences, 2).pvalue:.4f}", out)
        self.assertIn(f"{differences.mean() - stats.t.ppf(.975, 3) * differences.sem():.4f}", out)

    def test_unequal_sizes_and_nonzero_null(self):
        a, b = [1, 3, 4], [2, 6, 12, 20, 25]
        result, out = self.run_proc(pd.DataFrame({"g": [1]*3 + [2]*5, "x": a+b}),
                                    "class g; var x;", "h0=3")
        self.assertTrue(result.success)
        for equal in [True, False]:
            expected = stats.ttest_ind([x-3 for x in a], b, equal_var=equal)
            self.assertIn(f"{expected.statistic:.2f}", out)
            self.assertIn(f"{expected.pvalue:.4f}", out)

    def test_missing_class_sorted_levels_and_default_variables(self):
        frame = pd.DataFrame({"g": [2, 1, 2, 1, None], "x": [10, 1, 14, 3, 100]})
        result, out = self.run_proc(frame, "class g;")
        self.assertTrue(result.success)
        self.assertIn("Difference: 1 - 2", out)
        self.assertIn("-10.0000", out)
        self.assertNotIn("Variable: G", out)

    def test_small_probabilities_not_rounded_to_zero(self):
        result, out = self.run_proc(pd.DataFrame({"x": [100, 101, 102, 103, 104]}), "var x;")
        self.assertTrue(result.success)
        self.assertIn("<.0001", out)

    def test_zero_variance_does_not_crash_or_invent_pvalue(self):
        result, out = self.run_proc(pd.DataFrame({"g": [1,1,2,2], "x": [3,3,4,4]}), "class g; var x;")
        self.assertTrue(result.success)
        self.assertTrue(result.steps[-1].warnings)
        self.assertNotIn("nan", out)
        self.assertNotIn("inf", out)

    def test_invalid_inputs_are_reported(self):
        for statements, options in [("var absent;", ""), ("var x;", "alpha=1"),
                                    ("var x;", "alpha=0"), ("class x; var x;", "")]:
            with self.subTest(statements=statements, options=options):
                result, _ = self.run_proc(pd.DataFrame({"x": [1,2,3]}), statements, options)
                self.assertFalse(result.success)

    def test_insufficient_data_is_not_silent(self):
        result, _ = self.run_proc(pd.DataFrame({"x": [1, None]}), "var x;")
        self.assertFalse(result.success)
        self.assertTrue(result.steps[-1].warnings)

    def test_noprint(self):
        result, out = self.run_proc(pd.DataFrame({"x": [1,2,3]}), "var x;", "noprint")
        self.assertTrue(result.success)
        self.assertEqual(out, "")

    def test_standalone_sas_fixture(self):
        root = Path(__file__).resolve().parent.parent
        for name in ["saslite_custom/fixtures/test_ttest.sas"]:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                result = SasInterpreter().execute_file(root / name)
                self.assertTrue(result.success)
                output = "\n".join(m for step in result.steps for m in step.output_messages)
                self.assertIn("Std Dev", output)
                self.assertIn("Satterthwaite", output)


if __name__ == "__main__":
    unittest.main()
