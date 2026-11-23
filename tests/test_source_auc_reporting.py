"""Source subsets must remain reportable without changing full-test statistics."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import roc_auc_score
from continuation.common import config,dump,file_hash,write_csv,CLASSES
from continuation.artifacts import inference_signature
from continuation.evaluation import metrics,evaluate

class SourceAUCReportingTests(unittest.TestCase):
    def test_missing_classes_have_no_four_class_auc_and_valid_json(self):
        for present in [[0],[0,1],[0,1,2]]:
            with self.subTest(classes=present):
                y=np.array(present*2)
                p=np.full((len(y),4),.05);p[np.arange(len(y)),y]=.85
                result=metrics(y,p)
                self.assertIsNone(result['macro_roc_auc_ovr'])
                self.assertEqual(result['accuracy'],1.)
                self.assertEqual(result['balanced_accuracy'],1.)
                json.dumps(result,allow_nan=False)

    def test_all_four_classes_keep_the_same_auc_and_other_metrics(self):
        y=np.array([0,1,2,3,0,1,2,3])
        p=np.array([[.6,.2,.1,.1],[.1,.7,.1,.1],[.1,.1,.6,.2],[.1,.1,.2,.6],
                    [.3,.4,.2,.1],[.2,.5,.2,.1],[.1,.3,.4,.2],[.2,.1,.2,.5]])
        result=metrics(y,p)
        self.assertEqual(result['macro_roc_auc_ovr'],roc_auc_score(y,p,labels=np.arange(4),multi_class='ovr',average='macro'))
        self.assertEqual(result['accuracy'],7/8)
        self.assertEqual(result['balanced_accuracy'],7/8)
        json.dumps(result,allow_nan=False)

    def test_complete_evaluation_writes_partial_source_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            dump(run/'config.json',config(overrides={'mode':'image'}))
            dump(run/'data.json',{'domains':{}})
            (run/'best.weights.h5').write_bytes(b'fixed fixture checkpoint')
            y=np.array([0,1,2,3,0,1,2,3])
            rows=[{'record_id':str(i),'relative_path':str(i)+'.png','class_name':CLASSES[label],
                   'image_group':str(i),'source':'source_a' if label<2 else 'source_b','training_role':'test'} for i,label in enumerate(y)]
            write_csv(root/'metadata/dataset_manifest.csv',rows)
            checkpoint=file_hash(run/'best.weights.h5')
            calibration={'temperature':1.,'referral_confidence_threshold':.8,'tta':False,
                         'checkpoints':[checkpoint],'inference_signatures':[inference_signature(run)],
                         'manifest_sha256':file_hash(root/'metadata/dataset_manifest.csv')}
            dump(root/'calibration.json',calibration)
            p=np.full((len(y),4),.05);p[np.arange(len(y)),y]=.85
            with patch('continuation.evaluation.predict_runs',return_value=(rows,p,np.zeros(len(y)),[checkpoint])):
                result=evaluate(root,[run],root/'evaluation',root/'calibration.json',bootstrap=16)
            written=json.loads((root/'evaluation/test_metrics.json').read_text())
            self.assertEqual(written['macro_roc_auc_ovr'],1.)
            self.assertEqual(written['accuracy'],1.)
            for value in written['source_metrics'].values():
                self.assertIsNone(value['macro_roc_auc_ovr'])
            self.assertTrue((root/'evaluation/test_predictions.csv').is_file())
            self.assertTrue((root/'evaluation/risk_coverage.csv').is_file())

if __name__=='__main__':unittest.main()
