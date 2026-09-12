"""Synthetic acceptance cases for summaries, grouped inference, and SIDES=."""
import io
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from saslite import SasInterpreter
from saslite.executor.proc.stats import _ttest_limits, _ttest_test, _ttest_pvalue


class CourseworkTests(unittest.TestCase):
    def run_code(self, frame, code):
        sas = SasInterpreter()
        sas.reporter._stream = io.StringIO()
        sas.create_dataset('D', frame)
        result = sas.execute(code)
        return sas, result, sas.reporter._stream.getvalue()

    def test_output_names_partial_lists_and_subsets(self):
        frame = pd.DataFrame({'X':[1.,3.,5.], 'Y':[10.,30.,None]})
        sas,r,_ = self.run_code(frame, 'proc means data=d noprint; var x y; output out=a mean=mx my n=nx ny nmiss=nmx nmy; output out=b mean=m; output out=c mean(y)=ym; run;')
        self.assertTrue(r.success, r.error)
        a=sas.get_dataset('WORK','A').iloc[0]
        self.assertEqual((a.MX,a.MY,a.NX,a.NY,a.NMX,a.NMY), (3,20,3,2,0,1))
        b=sas.get_dataset('WORK','B')
        self.assertEqual(list(b.columns), ['_TYPE_','_FREQ_','M'])
        self.assertEqual(b.M.iloc[0],3)
        self.assertEqual(sas.get_dataset('WORK','C').YM.iloc[0],20)
        pd.testing.assert_frame_equal(frame,sas.get_dataset('WORK','D'))

    def test_autoname_and_unnamed_output(self):
        sas,r,_=self.run_code(pd.DataFrame({'x':[1,3], 'y':[10,30]}),
            'proc means data=d; var y x; output mean= n= / autoname; run;')
        self.assertTrue(r.success,r.error)
        out=sas.get_dataset('WORK','DATA1')
        self.assertEqual(list(out.columns),['_TYPE_','_FREQ_','Y_MEAN','X_MEAN','Y_N','X_N'])
        self.assertEqual(out.Y_MEAN.iloc[0],20)

    def test_default_output_and_display(self):
        sas,r,out=self.run_code(pd.DataFrame({'X':[1.,3.]}),'proc means data=d; var x; output out=z; run;')
        self.assertTrue(r.success,r.error)
        z=sas.get_dataset('WORK','Z')
        self.assertEqual(list(z._STAT_),['N','MIN','MAX','MEAN','STD'])
        np.testing.assert_allclose(z.X,[2,1,3,2,np.sqrt(2)])
        self.assertNotIn('25%',out)
        for text in ['N','MEAN','STD','MIN','MAX']: self.assertIn(text,out)

    def test_by_class_rollups_and_nway(self):
        frame=pd.DataFrame({'SITE':['A','A','B','B'], 'G':['X','Y','X','Y'], 'X':[1,3,5,7]})
        sas,r,out=self.run_code(frame,'proc means data=d mean; by site; class g; var x; output out=z mean=mx; run; proc summary data=d nway; by site; class g; var x; output out=leaf mean=mx; run;')
        self.assertTrue(r.success,r.error)
        z=sas.get_dataset('WORK','Z')
        self.assertEqual(list(z.MX),[2,1,3,6,5,7])
        self.assertEqual(list(z._TYPE_),[0,1,1,0,1,1])
        self.assertEqual(list(z._FREQ_),[2,1,1,2,1,1])
        self.assertEqual(list(sas.get_dataset('WORK','LEAF').MX),[1,3,5,7])
        self.assertIn('BY SITE=A',out)
        self.assertNotIn('The SUMMARY Procedure',out)

    def test_two_class_types(self):
        frame=pd.DataFrame({'A':['a','a','b','b'],'B':['x','y','x','y'],'X':[1,2,3,4]})
        sas,r,_=self.run_code(frame,'proc means data=d noprint; class a b; var x; output out=z sum=s; run;')
        self.assertTrue(r.success,r.error)
        z=sas.get_dataset('WORK','Z')
        self.assertEqual(z.groupby('_TYPE_').size().to_dict(),{0:1,1:2,2:2,3:4})
        self.assertEqual(z.groupby('_TYPE_').S.sum().tolist(),[10,10,10,10])

    def test_missing_class_and_by_levels(self):
        frame=pd.DataFrame({'SITE':[None,None,'B','B'], 'G':['',None,'x','x'],'X':[1.,3.,5.,None]})
        sas,r,_=self.run_code(frame,'proc means data=d nway missing; by site; class g; var x; output out=z mean=m n=n nmiss=nm; run;')
        self.assertTrue(r.success,r.error)
        z=sas.get_dataset('WORK','Z')
        self.assertEqual(list(z.M),[2,5])
        self.assertEqual(list(z.N),[2,1])
        self.assertEqual(list(z.NM),[0,1])
        self.assertTrue(pd.isna(z.SITE.iloc[0]))
        sas,r,_=self.run_code(frame,'proc means data=d nway; class g; var x; output out=z mean=m; run;')
        self.assertEqual(list(sas.get_dataset('WORK','Z').M),[5])

    def test_summary_count_only_and_print(self):
        frame=pd.DataFrame({'G':['a','a','b'],'X':[1,2,3]})
        sas,r,out=self.run_code(frame,'proc summary data=d nway; class g; output out=z; run;')
        self.assertTrue(r.success,r.error)
        self.assertEqual(list(sas.get_dataset('WORK','Z').columns),['G','_TYPE_','_FREQ_'])
        self.assertEqual(out,'')
        _,r,out=self.run_code(frame,'proc summary data=d print; var x; run;')
        self.assertTrue(r.success,r.error)
        self.assertIn('MEAN',out)
        _,r,_=self.run_code(frame,'proc summary data=d mean; run;')
        self.assertFalse(r.success)

    def test_invalid_means_does_not_write(self):
        frame=pd.DataFrame({'X':[1,2], 'C':['a','b']})
        for body in ['var x absent; output out=z mean=m;', 'var c; output out=z mean=m;',
                     'var x; output out=z typo=m;', 'var x; output out=z mean=m n=m;',
                     'var x; output out=z mean=a b;', 'by absent; var x; output out=z mean=m;']:
            with self.subTest(body=body):
                sas,r,_=self.run_code(frame,f'proc means data=d; {body} run;')
                self.assertFalse(r.success)
                self.assertTrue(r.error)
                with self.assertRaises(KeyError):sas.get_dataset('WORK','Z')

    def test_by_sorting_descending_and_notsorted(self):
        frame=pd.DataFrame({'G':['B','B','A','A','B','B'],'X':[1.,2.,4.,5.,8.,9.]})
        for proc,body in [('means','var x;'),('ttest','var x;'),('npar1way','class x; var x;')]:
            _,r,_=self.run_code(frame,f'proc {proc} data=d {"wilcoxon" if proc=="npar1way" else ""}; by g; {body} run;')
            self.assertFalse(r.success)
            self.assertIn('PROC SORT',r.error)
        _,r,out=self.run_code(frame,'proc ttest data=d plots=none; by g notsorted; var x; run;')
        self.assertTrue(r.success,r.error)
        self.assertEqual(out.count('The TTEST Procedure'),3)
        _,r,out=self.run_code(frame.iloc[:4],'proc ttest data=d plots=none; by descending g; var x; run;')
        self.assertTrue(r.success,r.error)

    def test_default_output_cannot_overwrite_summary_metadata(self):
        for variable in ['_TYPE_', '_FREQ_', '_STAT_']:
            with self.subTest(variable=variable):
                frame = pd.DataFrame({variable: [1., 3.]})
                sas, result, _ = self.run_code(frame,
                    f'proc means data=d; var {variable}; output out=z; run;')
                self.assertFalse(result.success)
                self.assertIn('conflicting output name', result.error)
                with self.assertRaises(KeyError):
                    sas.get_dataset('WORK', 'Z')
                sas, result, _ = self.run_code(frame,
                    f'proc means data=d; var {variable}; output out=z mean=value; run;')
                self.assertTrue(result.success, result.error)
                self.assertEqual(sas.get_dataset('WORK', 'Z').VALUE.iloc[0], 2)

    def test_grouped_tests_equal_independent_subsets(self):
        frame=pd.DataFrame({'SITE':['A']*6+['B']*6,'G':([1]*3+[2]*3)*2,'X':[1,2,4,3,5,9,2,3,7,4,8,11]})
        for proc in ['ttest','npar1way']:
            options='wilcoxon fp' if proc=='npar1way' else 'sides=l h0=-1'
            statement=f'proc {proc} data=d {options} plots=none; class g; var x;'
            _,r,out=self.run_code(frame,statement+' by site; run;')
            self.assertTrue(r.success,r.error)
            for site,sub in frame.groupby('SITE'):
                _,expected,independent=self.run_code(sub,statement+' run;')
                self.assertTrue(expected.success,expected.error)
                self.assertIn(independent.strip(),out)

    def test_failed_group_does_not_hide_valid_group(self):
        frame=pd.DataFrame({'SITE':['A','B','B','B'],'X':[1.,2.,4.,6.]})
        _,r,out=self.run_code(frame,'proc ttest data=d plots=none; by site; var x; run;')
        self.assertFalse(r.success)
        self.assertIn('SITE=A',r.error)
        self.assertIn('BY SITE=B',out)
        self.assertIn('4.0000',out)
        self.assertEqual(out.count('WARNING:'),1)

    def test_one_sided_all_designs(self):
        a=np.array([1.,3.,5.,8.]); b=np.array([0.,1.,2.,4.]); h0=-.5
        designs=[(pd.DataFrame({'X':a}),'var x;',lambda side:stats.ttest_1samp(a,h0,alternative=side).pvalue),
                 (pd.DataFrame({'A':a,'B':b}),'paired a*b;',lambda side:stats.ttest_1samp(a-b,h0,alternative=side).pvalue)]
        for frame,body,expected in designs:
            for sides,alternative in [('L','less'),('U','greater'),('2','two-sided')]:
                _,r,out=self.run_code(frame,f'proc ttest data=d sides={sides} h0={h0} plots=none; {body} run;')
                self.assertTrue(r.success,r.error)
                self.assertIn(_ttest_pvalue(expected(alternative)),out)
                if sides!='2':self.assertIn('Infinity',out)
        frame=pd.DataFrame({'G':[1]*4+[2]*4,'X':np.r_[a,b]})
        for sides,alternative in [('L','less'),('U','greater'),('2','two-sided')]:
            _,r,out=self.run_code(frame,f'proc ttest data=d sides={sides} h0={h0} plots=none; class g; var x; run;')
            self.assertTrue(r.success,r.error)
            for equal in [True,False]:
                self.assertIn(_ttest_pvalue(stats.ttest_ind(a-h0,b,equal_var=equal,alternative=alternative).pvalue),out)

    def test_one_sided_limits_and_unchanged_sd_limits(self):
        for side in ['L','U']:
            limits=_ttest_limits(4,2,8,3,.1,side)
            two=_ttest_limits(4,2,8,3,.1)
            self.assertEqual(limits[3:],two[3:])
            if side=='L':
                self.assertEqual(limits[1],-np.inf)
                self.assertAlmostEqual(limits[2],4+stats.t.ppf(.9,8)*2)
            else:
                self.assertAlmostEqual(limits[1],4-stats.t.ppf(.9,8)*2)
                self.assertEqual(limits[2],np.inf)
        self.assertTrue(np.isnan(_ttest_test(4,0,8,0,'U')[1]))

    def test_invalid_sides_and_warning_once(self):
        _,r,_=self.run_code(pd.DataFrame({'X':[1,2,3]}),'proc ttest data=d sides=bad; var x; run;')
        self.assertFalse(r.success)
        _,r,out=self.run_code(pd.DataFrame({'X':[3,3,3]}),'proc ttest data=d plots=none; var x; run;')
        self.assertTrue(r.success,r.error)
        self.assertEqual(out.count('WARNING:'),1)


if __name__=='__main__':
    unittest.main()
