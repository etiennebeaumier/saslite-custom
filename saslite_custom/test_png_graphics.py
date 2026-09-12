"""PNG files, ODS state, data fidelity, and headless rendering contracts."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.figure import Figure

from saslite import SasInterpreter
from saslite.cli.main import main
from saslite.runtime.png_graphics import draw_boxes, draw_histograms, draw_intervals


class PngGraphicsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.source=self.root/'course.sas'
        self.sas=SasInterpreter()
        self.log=io.StringIO()
        self.sas.reporter._stream=self.log
        self.frame=pd.DataFrame({'AGE':[18.,21.,25.,31.,35.,40.,52.,90.],
                                 'VISITS':[1.,2.,2.,3.,4.,5.,8.,10.],
                                 'G':[1,1,1,1,2,2,2,2]})
        self.sas.create_dataset('D',self.frame)

    def execute(self,code):
        return self.sas.execute(code,source_name=str(self.source))

    def check_images(self,result,count):
        self.assertTrue(result.success,result.error)
        self.assertEqual(len(result.image_paths),count)
        for path in result.image_paths:
            with Image.open(path) as image:
                self.assertEqual(image.format,'PNG')
                self.assertEqual(image.size,(1200,800))
                image.load()
                self.assertGreater(np.asarray(image).std(),5)
            self.assertIn(path,self.log.getvalue())

    def test_three_coursework_graphs(self):
        fixture=Path(__file__).resolve().parents[1]/'examples/graphics.sas'
        self.source.write_text(fixture.read_text())
        result=self.sas.execute_file(self.source)
        self.check_images(result,3)
        dataset=self.sas.session.get_dataset('WORK','INTRO_EX1')
        self.assertEqual(dataset.metadata.variables['AGE'].label,'Âge')
        self.assertEqual(dataset.metadata.variables['N_VISITE'].label,'Nombre de visites')
        self.assertEqual({Path(p).parent for p in result.image_paths},{self.root/'graphs'})
        self.assertEqual({Path(p).name for p in result.image_paths},
                         {'intro_ex1_histogram_AGE.png','intro_ex1_hbox_AGE.png','intro_ex1_scatter_AGE_N_VISITE.png'})

    def test_explicit_ods_required_and_options_do_not_enable(self):
        code='proc sgplot data=d; histogram age; run;'
        result=self.execute('ods graphics / imagename="custom" imagefmt=png; '+code)
        self.assertTrue(result.success,result.error)
        self.assertEqual(result.image_paths,[])
        self.assertFalse((self.root/'graphs').exists())
        result=self.execute('ods graphics on; '+code+' ods graphics off; '+code)
        self.check_images(result,1)

    def test_default_stats_stay_text_only(self):
        result=self.execute('proc ttest data=d; var age; run;')
        self.assertTrue(result.success,result.error)
        self.assertEqual(result.image_paths,[])
        self.assertFalse((self.root/'graphs').exists())
        self.assertIn('TTEST Graphs:',self.log.getvalue())

    def test_custom_folder_names_and_existing_files(self):
        code='ods listing gpath="figures with spaces"; ods graphics on / imagename="intro" imagefmt=png; proc sgplot data=d; hbox age; run;'
        first=self.execute(code)
        self.check_images(first,1)
        old=Path(first.image_paths[0]).read_bytes()
        second=self.execute(code)
        self.check_images(second,1)
        self.assertEqual(Path(first.image_paths[0]).name,'intro.png')
        self.assertEqual(Path(second.image_paths[0]).name,'intro1.png')
        self.assertEqual(Path(first.image_paths[0]).read_bytes(),old)
        another=SasInterpreter();another.reporter._stream=io.StringIO();another.create_dataset('D',self.frame)
        third=another.execute(code,source_name=str(self.source))
        self.assertEqual(Path(third.image_paths[0]).name,'intro2.png')

    def test_no_plots_and_procedure_suppression(self):
        self.sas.session.set_option('TERMINAL_PLOTS',False)
        r=self.execute('ods graphics on; proc sgplot data=d; histogram age; run; proc ttest data=d; var age; run;')
        self.assertEqual(r.image_paths,[])
        self.assertFalse((self.root/'graphs').exists())
        self.sas.session.set_option('TERMINAL_PLOTS',True)
        for option in ['noprint','plots=none']:
            r=self.execute(f'ods graphics on; proc ttest data=d {option}; var age; run; proc npar1way data=d wilcoxon fp {option}; class g; var age; run;')
            self.assertTrue(r.success,r.error)
            self.assertEqual(r.image_paths,[])

    def test_statistical_panels_and_selection(self):
        result=self.execute('ods graphics on; proc ttest data=d; paired age*visits; run;')
        self.check_images(result,6)
        for panel in ['histogram','box','interval','qqplot','profiles','agreement']:
            self.assertTrue(any('_'+panel+'_' in p for p in result.image_paths))
        result=self.execute('proc npar1way data=d wilcoxon fp plots=(boxplot wilcoxonboxplot fpboxplot); class g; var age; run;')
        self.check_images(result,3)
        result=self.execute('proc ttest data=d sides=u plots=(interval qqplot); class g; var age; run;')
        self.check_images(result,2)
        self.assertIn('Infinity',self.log.getvalue())

    def test_by_groups_have_distinct_files_and_labels(self):
        result=self.execute('ods graphics on; proc ttest data=d plots=box; by g; var age; run;')
        self.check_images(result,2)
        self.assertNotEqual(*result.image_paths)
        self.assertTrue(any('BY_G_1' in p for p in result.image_paths))
        self.assertTrue(any('BY_G_2' in p for p in result.image_paths))

    def test_missing_values_and_dataset_preservation(self):
        frame=pd.DataFrame({'AGE':[18.,None,30.,np.inf],'VISITS':[1.,2.,None,5.]})
        self.sas.create_dataset('D',frame)
        result=self.execute('ods graphics on; proc sgplot data=d; scatter x=age y=visits; run;')
        self.check_images(result,1)
        self.assertIn('omitted 3 observations',self.log.getvalue())
        pd.testing.assert_frame_equal(frame,self.sas.get_dataset('WORK','D'))

    def test_scatter_same_variable(self):
        r=self.execute('ods graphics on; proc sgplot data=d; scatter x=age y=age; run;')
        self.check_images(r,1)

    def test_include_uses_top_level_script_folder(self):
        included=self.root/'nested'
        included.mkdir()
        (included/'plot.sas').write_text('ods graphics on; proc sgplot data=d; hbox age; run;')
        self.source.write_text('%include "nested/plot.sas";')
        r=self.sas.execute_file(self.source)
        self.check_images(r,1)
        self.assertEqual(Path(r.image_paths[0]).parent,self.root/'graphs')
        self.assertFalse((included/'graphs').exists())

    @unittest.skipUnless(importlib.util.find_spec('flask'), 'Flask optional API test dependency not installed')
    def test_gui_execution_json_exposes_paths(self):
        from saslite.gui.app import app
        with patch.dict(app.config, SAS_SESSION=self.sas, TESTING=True):
            with patch('pathlib.Path.cwd',return_value=self.root):
                response=app.test_client().post('/api/execute',
                    json={'code':'ods graphics on; proc sgplot data=d; histogram age; run;'},
                    headers={'X-SASLite-Token':app.config['SAS_API_TOKEN']})
            self.assertEqual(response.status_code,200)
            body=response.get_json()
            self.assertTrue(body['success'],body.get('error'))
            self.assertEqual(len(body['image_paths']),1)
            self.assertEqual(body['image_paths'],body['steps'][-1]['image_paths'])
            self.assertTrue(Path(body['image_paths'][0]).is_file())

    def test_constant_samples_and_french_labels(self):
        self.sas.create_dataset('D',pd.DataFrame({'AGE':[30.,30.,30.]}))
        ds=self.sas.session.get_dataset('WORK','D')
        ds.metadata.variables['AGE'].label='Âge des élèves'
        result=self.execute('ods graphics on; proc sgplot data=d; hbox age; run; proc ttest data=d; var age; run;')
        self.check_images(result,5)

    def test_label_statement_reaches_plot_axes(self):
        captured=[]
        def inspect_draw(writer, kind, variable, title, draw):
            figure=Figure();axis=figure.subplots();draw(axis)
            captured.append((title,axis.get_xlabel(),axis.get_ylabel()))
        with patch('saslite.runtime.png_graphics.PngWriter.draw',new=inspect_draw):
            r=self.execute('data labelled; age=23; n_visite=2; label age="Âge" n_visite="Nombre de visites"; run; ods graphics on; proc sgplot data=labelled; scatter x=age y=n_visite; run;')
        self.assertTrue(r.success,r.error)
        self.assertEqual(captured[0][1:],('Âge','Nombre de visites'))

    def test_invalid_requests_do_not_create_images(self):
        for code in ['ods graphics on / imagefmt=jpeg;', 'ods graphics on / imagename="../escape";',
                     'ods graphics on / width=500;', 'ods listing gpath="";',
                     'ods graphics on; proc sgplot data=d; histogram absent; run;',
                     'ods graphics on; proc sgplot data=d; histogram age; hbox age; run;',
                     'ods graphics on; proc sgplot data=d; histogram age / unsupported; run;']:
            with self.subTest(code=code):
                r=self.execute(code)
                self.assertFalse(r.success)
                self.assertTrue(r.error)
                self.assertEqual(r.image_paths,[])
                self.assertFalse((self.root/'graphs').exists())

    def test_unusable_data(self):
        for values in [[None,None],['a','b']]:
            self.sas.create_dataset('D',pd.DataFrame({'AGE':values}))
            r=self.execute('ods graphics on; proc sgplot data=d; histogram age; run;')
            self.assertFalse(r.success)
            self.assertFalse((self.root/'graphs').exists())

    def test_write_failure_and_no_partial_file(self):
        (self.root/'blocked').write_text('existing file')
        result=self.execute('ods listing gpath="blocked"; ods graphics on; proc sgplot data=d; histogram age; run;')
        self.assertFalse(result.success)
        self.assertIn('PNG',result.error)
        self.assertEqual((self.root/'blocked').read_text(),'existing file')
        self.assertEqual(list(self.root.glob('*.tmp')),[])
        with patch('matplotlib.figure.Figure.savefig',side_effect=OSError('render failure')):
            r=self.execute('ods listing gpath="new"; proc sgplot data=d; histogram age; run;')
        self.assertFalse(r.success)
        self.assertFalse((self.root/'new').exists())

    def test_plot_data_against_known_values(self):
        fig=Figure();ax=fig.subplots()
        values=np.array([1,2,3,4,5,6,7,100])
        draw_boxes(ax,[values],['sample'],'X')
        rectangle=ax.patches[0]
        self.assertEqual((rectangle.get_x(),rectangle.get_width()),(2.5,4.0))
        ax.clear();draw_histograms(ax,[values],['sample'],'X')
        self.assertAlmostEqual(sum(p.get_height() for p in ax.patches),100)
        ax.clear();draw_intervals(ax,[('one-sided',4.,1.,np.inf)],0)
        self.assertTrue(any(getattr(t,'arrow_patch',None) is not None for t in ax.texts))

    def test_cli_file_base_from_another_directory_and_no_plots(self):
        fixture=Path(__file__).resolve().parents[1]/'examples/graphics.sas'
        self.source.write_text(fixture.read_text())
        with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.source)]),0)
            self.assertEqual(main(['--no-plots',str(self.source)]),0)
        self.assertEqual(len(list((self.root/'graphs').glob('*.png'))),3)


if __name__=='__main__':
    unittest.main()
