"""Regressions for prediction parity, saved-artifact binding and recovery."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from continuation.common import config,dump,file_hash,write_csv,runtime,CLASSES
from continuation.artifacts import inference_signature,validate_calibration

def run_fixture(path,mode='image'):
    path.mkdir();dump(path/'config.json',config(overrides={'mode':mode,'weights':None,'image_size':32}))
    dump(path/'data.json',{'domains':{}});dump(path/'normalizer.json',{'median':[0],'scale':[1]})
    (path/'best.weights.h5').write_bytes(b'fixture weights')
    return {'checkpoints':[file_hash(path/'best.weights.h5')],
        'inference_signatures':[inference_signature(path)],'tta':False,'temperature':1.,
        'referral_confidence_threshold':.8,'manifest_sha256':'fixture'}

class ArtifactTests(unittest.TestCase):
    def test_same_pixels_cannot_span_selection_and_calibration(self):
        from continuation.common import validate_manifest,evaluation_role
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'image.png').write_bytes(b'fixture');groups={}
            for i in range(100):
                row={'image_group':str(i),'training_role':'val'};groups[evaluation_role(row)]=str(i)
            rows=[{'relative_path':'image.png','class_name':'Normal','pixel_sha256':'same',
                'image_group':group,'training_role':'val'} for group in groups.values()]
            self.assertEqual(len(rows),2);write_csv(root/'metadata/dataset_manifest.csv',rows)
            with self.assertRaisesRegex(ValueError,'overlaps'):validate_manifest(root)

    def test_recipe_rejects_modified_config_metadata_normalizer_and_tta(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';calibration=run_fixture(run,'fusion')
            validate_calibration(calibration,[run],False,'fixture')
            for name in ['config.json','data.json','normalizer.json']:
                with self.subTest(name=name):
                    original=(run/name).read_text();(run/name).write_text(original+'\n')
                    with self.assertRaisesRegex(ValueError,'recipe'):validate_calibration(calibration,[run],False)
                    (run/name).write_text(original)
            with self.assertRaisesRegex(ValueError,'recipe'):validate_calibration(calibration,[run],True)
            with self.assertRaisesRegex(ValueError,'manifest'):validate_calibration(calibration,[run],False,'other')

    def test_old_calibration_requires_recalibration(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';calibration=run_fixture(run);del calibration['inference_signatures']
            with self.assertRaisesRegex(ValueError,'rerun calibration'):validate_calibration(calibration,[run],False)

    def test_modified_roi_is_rejected(self):
        from continuation.features import FeatureTable
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'roi.png').write_bytes(b'original')
            write_csv(root/'features.csv',[{'pixel_sha256':'p','roi_path':'roi.png','roi_sha256':file_hash(root/'roi.png')}])
            dump(root/'provenance.json',{'features_csv_sha256':file_hash(root/'features.csv')})
            self.assertEqual(FeatureTable(root).roi({'pixel_sha256':'p'}),str(root/'roi.png'))
            (root/'roi.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'ROI'):FeatureTable(root).roi({'pixel_sha256':'p'})

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tf=runtime(config())
    def tearDown(self):self.tf.keras.backend.clear_session()

    def test_png_and_jpeg_prediction_matches_evaluation_input(self):
        from continuation.data import make_data
        from continuation.inference import predict
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';calibration=run_fixture(run);dump(root/'calibration.json',calibration)
            rng=np.random.default_rng(5);image=rng.integers(0,256,(41,73,3),dtype='uint8')
            captured=[];tf=self.tf
            class Predictor:
                def load_weights(self,*args):pass
                def __call__(self,x,training=False):
                    captured.append(x['image'].numpy());return tf.constant([[.1,.2,.3,.4]])
            for suffix in ['png','jpg']:
                with self.subTest(suffix=suffix):
                    path=root/('image.'+suffix);Image.fromarray(image).save(path)
                    write_csv(root/'metadata/dataset_manifest.csv',[{'relative_path':path.name,'class_name':'Normal',
                        'pixel_sha256':'p','training_role':'test','image_group':'g','source':'fixture'}])
                    ds,_=make_data(root,'test',json.loads((run/'config.json').read_text()))
                    expected=next(iter(ds))[0]['image'].numpy()
                    with patch('continuation.inference.build_classifier',return_value=(Predictor(),None)):
                        predict(path,[run],root/'calibration.json',root/('out_'+suffix))
                    np.testing.assert_array_equal(captured[-1],expected)

    def test_classifier_resume_preserves_better_interrupted_stage(self):
        from continuation.training import train
        tf=self.tf
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'run';out.mkdir();rows=[]
            Image.fromarray(np.full((32,32,3),100,'uint8')).save(root/'image.png')
            for role in ['fit','select','calibrate']:
                for i,name in enumerate(CLASSES):
                    group=role+str(i);n=0
                    if role!='fit':
                        while (int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3==0)!=(role=='calibrate'):
                            n+=1;group=role+str(i)+'_'+str(n)
                    rows.append({'relative_path':'image.png','class_name':name,'training_role':'fit' if role=='fit' else 'val',
                        'pixel_sha256':role+str(i),'image_group':group,'source':'fixture'})
            write_csv(root/'metadata/dataset_manifest.csv',rows)
            c=config(overrides={'mode':'image','weights':None,'image_size':32,'head_epochs':1,'fine_epochs':0,'full_epochs':0})
            inp=tf.keras.Input((32,32,3),name='image');output=tf.keras.layers.Dense(4,activation='softmax')(tf.keras.layers.GlobalAveragePooling2D()(inp))
            model=tf.keras.Model({'image':inp},output);model.save_weights(out/'head.weights.h5')
            original=file_hash(out/'head.weights.h5')
            def fit(*args,**kwargs):
                model.weights[0].assign(tf.ones_like(model.weights[0])*9)
                for callback in kwargs['callbacks']:
                    if isinstance(callback,tf.keras.callbacks.ModelCheckpoint):
                        callback.set_model(model);callback.on_epoch_end(0,{'val_balanced_accuracy':.25})
                return SimpleNamespace(history={'val_balanced_accuracy':[.25]})
            with patch('continuation.training.build_classifier',return_value=(model,None)),patch('continuation.training.unfreeze'),\
                 patch.object(model,'evaluate',return_value={'balanced_accuracy':.75}),patch.object(model,'fit',side_effect=fit):
                train(root,out,c,resume=True)
            self.assertEqual(file_hash(out/'head.weights.h5'),original)
            self.assertEqual(json.loads((out/'selection.json').read_text())['best_selection_balanced_accuracy'],.75)

    def test_anatomy_resume_preserves_better_checkpoint(self):
        from continuation.anatomy import train
        tf=self.tf
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'run';out.mkdir();rows=[]
            for i,role in enumerate(['train','val']):
                a=np.zeros((16,16),'uint8');np.savez(root/f'{i}.npz',image=a,ovary=a,follicle=a,distance=a)
                rows.append({'file':f'{i}.npz','split':role,'instance_valid':0})
            write_csv(root/'manifest.csv',rows)
            c=config(overrides={'seg_size':16,'seg_filters':1,'seg_epochs':1})
            inp=tf.keras.Input((16,16,1));model=tf.keras.Model(inp,{n:tf.keras.layers.Conv2D(1,1,name=n)(inp) for n in ['ovary','follicle','distance']})
            model.save_weights(out/'best.weights.h5');original=file_hash(out/'best.weights.h5')
            dump(out/'config.json',c);dump(out/'training_data.json',{'manifest_sha256':file_hash(root/'manifest.csv')})
            def fit(*args,**kwargs):
                model.weights[0].assign(tf.ones_like(model.weights[0])*9)
                for callback in kwargs['callbacks']:
                    if isinstance(callback,tf.keras.callbacks.ModelCheckpoint):
                        callback.set_model(model);callback.on_epoch_end(0,{'val_loss':.9})
                return SimpleNamespace(history={'val_loss':[.9]})
            with patch('continuation.anatomy.build_segmenter',return_value=model),patch.object(model,'evaluate',return_value={'loss':.1}),\
                 patch.object(model,'fit',side_effect=fit):train(root,out,c,resume=True)
            self.assertEqual(file_hash(out/'best.weights.h5'),original)

    def test_anatomy_evaluation_uses_saved_thresholds(self):
        from continuation.anatomy import evaluate
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();a=np.zeros((16,16),'uint8')
            np.savez(root/'slice.npz',image=a,ovary=a,follicle=a,instances=a)
            write_csv(root/'manifest.csv',[{'file':'slice.npz','group':'g','split':'test','instance_valid':0}])
            dump(run/'training_data.json',{'manifest_sha256':file_hash(root/'manifest.csv')})
            dump(run/'config.json',config(overrides={'seg_size':16,'ovary_threshold':.9,'follicle_threshold':.8,'min_follicle_area':11,'min_peak_distance':7}))
            tf=self.tf
            class Predictor:
                def load_weights(self,*args):pass
                def __call__(self,*args,**kwargs):return {n:tf.ones((1,16,16,1))*.6 for n in ['ovary','follicle','distance']}
            with patch('continuation.anatomy.build_segmenter',return_value=Predictor()):evaluate(root,run,root/'out')
            metrics=json.loads((root/'out/anatomy_metrics.json').read_text())
            self.assertEqual(metrics['ovary_dice'],1.)
            self.assertIsNone(metrics['instance_f1_iou50'])
            (root/'manifest.csv').write_text((root/'manifest.csv').read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'differs'):evaluate(root,run,root/'other')

if __name__=='__main__':unittest.main()
