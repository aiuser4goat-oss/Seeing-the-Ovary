import unittest
import numpy as np
from PIL import Image
from accuracy_development.anatomy_safety import crop_decision,guarded_postprocess
class AnatomySafetyTests(unittest.TestCase):
 def test_margin_cannot_hide_small_region(self):
  mask=np.zeros((256,256),bool);mask[50:85,50:85]=True
  self.assertTrue(crop_decision(mask,margin=.1)['fallback'])
  self.assertTrue(crop_decision(mask,margin=2.)['fallback'])
 def test_fallback_keeps_image_and_marks_measurements_missing(self):
  image=Image.fromarray(np.arange(64*32*3,dtype=np.uint8).reshape(32,64,3));ov=np.zeros((256,256),'float32');ov[110:120,110:120]=1
  pred={'ovary':ov,'follicle':ov.copy(),'distance':ov.copy()};config={'seg_size':256,'ovary_threshold':.5,'follicle_threshold':.8,'min_follicle_area':30,'min_peak_distance':3,'follicle_instance_method':'connected_components'}
  crop,vector,_,_,decision=guarded_postprocess(image,pred,config)
  self.assertTrue(decision['fallback']);np.testing.assert_array_equal(np.asarray(crop),np.asarray(image));self.assertTrue(np.isnan(vector).all())
 def test_zero_follicles_is_distinct_from_missing_anatomy(self):
  image=Image.new('RGB',(256,256));ov=np.zeros((256,256),'float32');ov[40:210,40:210]=1;zero=np.zeros_like(ov)
  config={'seg_size':256,'ovary_threshold':.5,'follicle_threshold':.8,'min_follicle_area':30,'min_peak_distance':3,'follicle_instance_method':'connected_components'}
  _,vector,_,_,decision=guarded_postprocess(image,{'ovary':ov,'follicle':zero,'distance':zero},config)
  self.assertFalse(decision['fallback']);self.assertTrue(np.isfinite(vector[0]));self.assertEqual(vector[3],0)
 def test_empty_mask_uses_full_bounds(self):
  result=crop_decision(np.zeros((70,80),bool));self.assertEqual(result['bounds'],(0,70,0,80));self.assertTrue(result['fallback'])
if __name__=='__main__':unittest.main()
