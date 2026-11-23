import unittest
import numpy as np
from continuation.morphology import instances_from_predictions,measurements,FEATURES

class RefinedPostprocessingTests(unittest.TestCase):
    def test_connected_component_preserves_one_contiguous_region(self):
        y,x=np.ogrid[:64,:64]
        p=(((x-24)**2+(y-32)**2<100)|((x-40)**2+(y-32)**2<100)).astype(float)
        ov,cc=instances_from_predictions(np.ones_like(p),p,method='connected_components')
        _,ws=instances_from_predictions(np.ones_like(p),p,min_peak_distance=5)
        self.assertEqual(np.count_nonzero(np.unique(cc)),1)
        self.assertEqual(np.count_nonzero(np.unique(ws)),2)
        np.testing.assert_array_equal(cc>0,p>0)
        self.assertEqual(measurements(ov,cc)[FEATURES.index('visible_count')],1)
    def test_small_regions_removed_and_separate_regions_retained(self):
        p=np.zeros((32,32));p[2:6,2:6]=1;p[20:24,20:24]=1;p[10,10]=1
        _,labels=instances_from_predictions(np.ones_like(p),p,min_area=5,method='connected_components')
        self.assertEqual(np.count_nonzero(np.unique(labels)),2)
        self.assertEqual(labels[10,10],0)
    def test_invalid_method_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown follicle'):
            instances_from_predictions(np.ones((8,8)),np.zeros((8,8)),method='unknown')

if __name__=='__main__':unittest.main()
