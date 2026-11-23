import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from continuation.common import config,role_rows,write_csv,dump,runtime
from continuation.morphology import measurements,instances_from_predictions,FEATURES
from continuation.features import normalized
from continuation.data import training_weights

class MorphologyTests(unittest.TestCase):
    def test_empty_ovary_means_missing(self):
        self.assertTrue(np.isnan(measurements(np.zeros((32,32)),np.zeros((32,32)))).all())
    def test_distinct_touching_follicles(self):
        y,x=np.ogrid[:64,:64];mask=((x-24)**2+(y-32)**2<100)|((x-40)**2+(y-32)**2<100)
        ovary=np.ones((64,64));ov,instances=instances_from_predictions(ovary,mask.astype(float),min_peak_distance=5)
        self.assertEqual(len(np.unique(instances[instances>0])),2)
        vector=measurements(ov,instances)
        self.assertEqual(vector[FEATURES.index('visible_count')],2)
    def test_missingness_is_explicit(self):
        stats={'median':[2]*len(FEATURES),'scale':[1]*len(FEATURES)}
        a=normalized(np.full(len(FEATURES),np.nan),stats)
        np.testing.assert_array_equal(a[:len(FEATURES)],0)
        np.testing.assert_array_equal(a[len(FEATURES):],1)
    def test_instance_matching_counts_misses_and_merges(self):
        from continuation.morphology import instance_matches
        gt=np.zeros((8,8),int);gt[:2,:2]=1;gt[4:6,4:6]=2
        pred=gt.copy();pred[pred==2]=0
        self.assertEqual(instance_matches(gt,pred),{'instance_tp':1,'instance_fp':0,'instance_fn':1})
    def test_repeated_files_do_not_dominate_as_much(self):
        rows=[{'pixel_sha256':'same','class_name':'Normal'}]*9+[{'pixel_sha256':'other','class_name':'Normal'}]
        w=training_weights(rows,config(overrides={'class_balance':False}))
        self.assertLess(w[0],w[-1]);self.assertAlmostEqual(float(w.mean()),1.,places=6)
    def test_modified_feature_cache_is_rejected(self):
        from continuation.features import FeatureTable
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'features.csv').write_text('pixel_sha256,roi_path\n')
            dump(p/'provenance.json',{'features_csv_sha256':'not-the-current-table'})
            with self.assertRaises(ValueError):FeatureTable(p)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tf=runtime(config())
    def tearDown(self):self.tf.keras.backend.clear_session()
    def test_domain_gradient_is_reversed(self):
        from continuation.models import GradientReverse
        tf=self.tf;x=tf.Variable([1.,2.])
        with tf.GradientTape() as tape:y=tf.reduce_sum(GradientReverse()(x))
        np.testing.assert_array_equal(tape.gradient(y,x).numpy(),[-1,-1])
    def test_finetuning_and_serialization(self):
        from continuation.models import build_classifier,unfreeze,compile_classifier
        tf=self.tf;c=config(overrides={'weights':None,'mode':'fusion','image_size':64})
        model,encoder=build_classifier(c)
        inputs={'image':np.ones((2,64,64,3),'float32')*100,'roi':np.ones((2,64,64,3),'float32')*90,
                'morphology':np.zeros((2,len(FEATURES)*2),'float32')}
        target=np.eye(4,dtype='float32')[[0,1]]
        unfreeze(encoder,.3);compile_classifier(model,c,.001,10)
        bn=[l for l in encoder.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
        means=[l.moving_mean.numpy().copy() for l in bn]
        before=encoder.trainable_weights[-1].numpy().copy()
        model.train_on_batch(inputs,target)
        self.assertFalse(np.array_equal(before,encoder.trainable_weights[-1].numpy()))
        for layer,mean in zip(bn,means):np.testing.assert_array_equal(layer.moving_mean.numpy(),mean)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.weights.h5';model.save_weights(path)
            new,_=build_classifier(c);new.load_weights(path)
            np.testing.assert_allclose(model(inputs,training=False).numpy(),new(inputs,training=False).numpy(),atol=1e-6)
    def test_legacy_builds_without_downloading_weights(self):
        from continuation.models import build_classifier
        model,_=build_classifier(config(overrides={'mode':'legacy','weights':None,'image_size':64}))
        self.assertEqual(model({'image':np.zeros((1,64,64,3),'float32')}).shape,(1,4))
    def test_confusion_metric_and_temperature(self):
        from continuation.models import ConfusionMetric
        from continuation.evaluation import apply_temperature
        metric=ConfusionMetric();metric.update_state(np.eye(4),np.eye(4))
        self.assertAlmostEqual(float(metric.result()),1.)
        p=np.array([[.7,.1,.1,.1]])
        self.assertEqual(int(apply_temperature(p,2).argmax()),0)
        self.assertLess(apply_temperature(p,2).max(),p.max())
    def test_domain_head_can_train(self):
        from continuation.models import build_classifier,compile_classifier
        c=config(overrides={'mode':'image','weights':None,'image_size':64,'domain_weight':.05})
        model,_=build_classifier(c,2);compile_classifier(model,c,.001,10)
        result=model.train_on_batch({'image':np.ones((2,64,64,3),'float32')*100},
            {'class':np.eye(4,dtype='float32')[[0,1]],'domain':np.eye(2,dtype='float32')},return_dict=True)
        self.assertIn('class_balanced_accuracy',result)
    def test_matched_capacity_control_removes_morphology_information(self):
        from continuation.data import make_data
        from continuation.morphology import FEATURES
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.fromarray(np.ones((32,32,3),'uint8')*100).save(root/'image.png')
            write_csv(root/'metadata/dataset_manifest.csv',[{'relative_path':'image.png','class_name':'Normal',
                'pixel_sha256':'p','training_role':'fit','image_group':'g','source':'fixture'}])
            class Table:
                def vector(self,row):return np.ones(len(FEATURES),'float32')*3
                def roi(self,row):return str(root/'image.png')
            c=config(overrides={'mode':'fusion','image_size':32,'batch_size':1,'data_threads':1,'morphology_ablation':'zeros'})
            stats={'median':[0]*len(FEATURES),'scale':[1]*len(FEATURES)}
            ds,_=make_data(root,'fit',c,Table(),stats)
            x,_,_=next(iter(ds));np.testing.assert_array_equal(x['morphology'].numpy(),0)

if __name__=='__main__':unittest.main()
