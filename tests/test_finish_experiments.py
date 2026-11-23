"""Verification for final reporting; run with Python containing matplotlib/sklearn."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import finish_experiments as finish

CLASSES=['Benign','Malignant','Normal','PCOS']

class FinalReportsTests(unittest.TestCase):
    def test_paired_groups(self):
        rows=[{'image_group':str(i//2),'class_name':CLASSES[i%4]} for i in range(80)]
        y=np.array([i%4 for i in range(80)])
        r=finish.paired_group_bootstrap(rows,{'a':y,'b':y,'c':(y+1)%4},CLASSES,[('a','b'),('a','c')],100)
        self.assertEqual(r[0]['balanced_accuracy_difference_95_ci'],[0.,0.])
        self.assertEqual(r[0]['macro_f1_difference'],0.)
        self.assertEqual(r[1]['balanced_accuracy_difference'],1.)
        self.assertEqual(r[1]['macro_f1_difference'],1.)
        self.assertEqual(r[1]['image_groups'],40)

    def test_complete_report_and_alignment_guard(self):
        with tempfile.TemporaryDirectory(prefix='continuation_report_check_') as tmp:
            root=Path(tmp)/'Classification';results=Path(tmp)/'results'
            (root/'metadata').mkdir(parents=True)
            rows=[{'record_id':str(i),'relative_path':f'test/{i}.png','class_name':CLASSES[i%4],
                   'image_group':str(i//2),'training_role':'test'} for i in range(80)]
            def csv_write(path,rows):
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            csv_write(root/'metadata/dataset_manifest.csv',rows)
            for name,_,_ in finish.EXPERIMENTS:
                run=results/name;(run/'evaluation').mkdir(parents=True);(run/'calibration').mkdir()
                metrics={'accuracy':.75,'balanced_accuracy':.75,'macro_f1':.75,'macro_roc_auc_ovr':.9,
                         'nll':.5,'brier':.4,'ece_10_bins':.1,'retained_coverage':.5,'retained_accuracy':.9,
                         'confusion_matrix':[[15,5,0,0],[0,15,5,0],[0,0,15,5],[5,0,0,15]],
                         'group_bootstrap_95_ci':{'balanced_accuracy':[.6,.9],'macro_f1':[.6,.9]}}
                finish.save(run/'evaluation/test_metrics.json',metrics)
                finish.save(run/'selection.json',{'best_selection_balanced_accuracy':.8})
                finish.save(run/'calibration/calibration.json',{'temperature':1.,'referral_confidence_threshold':.8})
                predictions=[]
                for r in rows:
                    i=int(r['record_id']);y=i%4;pr=y if i//4%4 else (y+1)%4
                    predictions.append({'record_id':r['record_id'],'true_class':r['class_name'],
                        'predicted_class':CLASSES[pr],**{c:.85 if j==pr else .05 for j,c in enumerate(CLASSES)}})
                csv_write(run/'evaluation/test_predictions.csv',predictions)
                (run/'evaluation/risk_coverage.csv').write_text('threshold,coverage,error_rate\n0,1,0.25\n0.8,0.5,0.1\n1,0,\n')
            report=finish.make_reports(results,root,CLASSES)
            self.assertEqual(len(list(report.glob('*.png'))),5)
            comparisons=json.loads((report/'paired_comparisons.json').read_text())
            self.assertTrue(all(r['balanced_accuracy_difference']==0 and r['macro_f1_difference']==0 for r in comparisons))
            self.assertEqual(len(finish.read_csv(report/'model_comparison.csv')),6)
            self.assertIn('Instance F1',(report/'RESULTS_AND_LIMITATIONS.txt').read_text())
            bad=results/'fusion/evaluation/test_predictions.csv'
            records=finish.read_csv(bad);records[0]['record_id']='unknown'
            csv_write(bad,records)
            with self.assertRaisesRegex(ValueError,'record IDs'):
                finish.make_reports(results,root,CLASSES)

if __name__=='__main__':unittest.main()
