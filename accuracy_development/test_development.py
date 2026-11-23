import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from .models import AppearanceAugment,build,residual_anatomy,warm_start
from .data import dataset,mix_batch,fitting_weights
from .runner import fit_stage
from continuation.common import config,runtime
from continuation.models import build_classifier

class DevelopmentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):runtime({'seed':2627,'mixed_precision':False});tf.config.experimental.enable_op_determinism()
 def test_test_and_calibration_roles_are_rejected_before_reading_files(self):
  for role in ['test','calibrate','val','anything']:
   with self.assertRaises(ValueError):dataset('/not/a/dataset',role,{},32)
 def test_augmentation_is_identity_and_does_not_advance_rng_at_inference(self):
  layer=AppearanceAugment();x=tf.reshape(tf.range(2*16*16*3,dtype=tf.float32),[2,16,16,3]);state=layer.generator.state.numpy().copy()
  np.testing.assert_array_equal(layer(x,training=False).numpy(),x.numpy());np.testing.assert_array_equal(state,layer.generator.state.numpy())
 def test_weighted_mixup_preserves_the_weighted_cross_entropy_target(self):
  n=8;x=tf.reshape(tf.eye(n),[n,1,1,n]);y=tf.one_hot(tf.range(n)%4,4);w=tf.cast(tf.range(1,n+1),tf.float32)
  mixed,target,weight=mix_batch({'image':x},y,w,tf.constant(121),.3)
  coefficients=mixed['image'].numpy().reshape(n,n);expected_weight=coefficients@w.numpy()
  expected_target=(coefficients@(w.numpy()[:,None]*y.numpy()))/expected_weight[:,None]
  np.testing.assert_allclose(weight.numpy(),expected_weight,atol=2e-6);np.testing.assert_allclose(target.numpy(),expected_target,atol=2e-6)
  self.assertTrue(np.any(np.count_nonzero(coefficients,axis=1)>1))
 def test_class_weights_preserve_all_records_and_have_unit_mean(self):
  rows=[{'class_name':n,'pixel_sha256':n} for n in ['Normal']*8+['PCOS']*4+['Benign']*2+['Malignant']]
  for power in [0,.5,1]:
   w=fitting_weights(rows,power);self.assertEqual(len(w),len(rows));self.assertTrue(np.all(np.isfinite(w)&(w>0)));self.assertAlmostEqual(float(w.mean()),1.,places=6)
 def test_residual_is_zero_initially_and_missing_ovary_disables_it(self):
  c={'seed':2627,'backbone':'efficientnetv2b0'};base,_=build(c,weights=None);m=residual_anatomy(base);x=tf.ones([2,32,32,3])*100;features=np.zeros([2,26],np.float32)
  p=base({'image':x},training=False).numpy();np.testing.assert_allclose(p,m({'image':x,'morphology':tf.convert_to_tensor(features)},training=False).numpy(),atol=2e-5)
  layer=m.get_layer('anatomy_correction');weights=layer.get_weights();weights[1]=np.array([.5,-.5,.2,-.1],np.float32);layer.set_weights(weights)
  valid=m({'image':x,'morphology':tf.convert_to_tensor(features)},training=False).numpy();self.assertGreater(np.max(np.abs(valid-p)),1e-4)
  features[:,13]=1;np.testing.assert_allclose(p,m({'image':x,'morphology':tf.convert_to_tensor(features)},training=False).numpy(),atol=2e-5)
  tf.keras.backend.clear_session()
 def test_warm_transfer_preserves_original_classifier_before_training(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);c=config(overrides={'mode':'image','backbone':'efficientnetv2b0','weights':None,'image_size':32,'mixed_precision':False})
   old,_=build_classifier(c);old.save_weights(root/'best.weights.h5');(root/'config.json').write_text(json.dumps(c))
   new,encoder=build(c,weights=None);receipt=warm_start(new,encoder,root);self.assertLess(receipt['synthetic_transfer_max_abs'],2e-5)
  tf.keras.backend.clear_session()
 def test_ema_checkpoint_matches_the_validation_score_logged_for_selection(self):
  with tempfile.TemporaryDirectory() as temp:
   inp=tf.keras.layers.Input((3,),name='image');model=tf.keras.Model({'image':inp},tf.keras.layers.Dense(4,activation='softmax')(inp))
   x=tf.reshape(tf.linspace(-1.,1.,24),[8,3]);y=tf.one_hot(tf.range(8)%4,4);w=tf.ones(8)
   options=tf.data.Options();options.threading.private_threadpool_size=1
   ds=tf.data.Dataset.from_tensor_slices(({'image':x},y,w)).batch(4).with_options(options)
   stage={'name':'ema_check','size':32,'fraction':0,'epochs':2,'lr':.01,'power':1.,'mixup':0.}
   model._initial_epoch=999  # stale epoch from a different recovered stage
   with patch('accuracy_development.runner.dataset',return_value=(ds,[{}]*8)):
    result=fit_stage(model,None,None,Path(temp),{'batch_size':4,'image_size':32},stage,time.monotonic()+120)
   actual=model.evaluate(ds,verbose=0,return_dict=True)['balanced_accuracy'];self.assertAlmostEqual(actual,result['selection_balanced_accuracy'],places=6)
  tf.keras.backend.clear_session()
 def test_ema_and_anatomy_branch_support_mixed_precision(self):
  tf.keras.mixed_precision.set_global_policy('mixed_float16')
  try:
   with tempfile.TemporaryDirectory() as temp:
    image=tf.keras.layers.Input((8,8,3),name='image');h=tf.keras.layers.GlobalAveragePooling2D()(image)
    h=tf.keras.layers.Dense(8,activation='gelu')(h);logits=tf.keras.layers.Dense(4,dtype='float32',name='class_logits')(h)
    base=tf.keras.Model({'image':image},tf.keras.layers.Activation('softmax',dtype='float32')(logits));model=residual_anatomy(base)
    x={'image':tf.ones([8,8,8,3])*100,'morphology':tf.zeros([8,26])};y=tf.one_hot(tf.range(8)%4,4);w=tf.ones(8)
    options=tf.data.Options();options.threading.private_threadpool_size=1
    ds=tf.data.Dataset.from_tensor_slices((x,y,w)).batch(4).with_options(options)
    stage={'name':'mixed_ema_check','size':8,'fraction':0,'epochs':2,'lr':.001,'power':1.,'mixup':0.}
    with patch('accuracy_development.runner.dataset',return_value=(ds,[{}]*8)):
     result=fit_stage(model,None,None,Path(temp),{'batch_size':4,'image_size':8},stage,time.monotonic()+120)
    self.assertTrue(np.isfinite(result['selection_balanced_accuracy']));self.assertEqual(model(x,training=False).dtype,tf.float32)
  finally:tf.keras.mixed_precision.set_global_policy('float32');tf.keras.backend.clear_session()

if __name__=='__main__':unittest.main()
