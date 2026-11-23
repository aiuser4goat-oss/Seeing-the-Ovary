import unittest
import numpy as np
from continuation.common import config,runtime

class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tf=runtime(config())
    def test_validation_is_identity_and_does_not_consume_rng(self):
        from continuation.augmentation import UltrasoundAugment
        layer=UltrasoundAugment();x=np.linspace(0,255,64*64*3,dtype='float32').reshape(1,64,64,3)
        before=layer.generator.state.numpy().copy()
        np.testing.assert_array_equal(layer(x,training=False).numpy(),x)
        np.testing.assert_array_equal(layer.generator.state.numpy(),before)
    def test_training_changes_intensity_and_stays_bounded(self):
        from continuation.augmentation import UltrasoundAugment
        layer=UltrasoundAugment(flips=False);x=np.ones((4,64,64,3),'float32')*120
        first=layer(x,training=True).numpy();second=layer(x,training=True).numpy()
        self.assertFalse(np.array_equal(first,x));self.assertFalse(np.array_equal(first,second))
        self.assertTrue(np.isfinite(first).all());self.assertGreaterEqual(first.min(),0);self.assertLessEqual(first.max(),255)
    def test_shared_global_roi_transforms_and_black_padding(self):
        from continuation.augmentation import UltrasoundAugment
        rng=np.random.default_rng(7);x=rng.uniform(20,180,(2,32,32,3)).astype('float32');x[:,:3]=0
        a,b=UltrasoundAugment(speckle=0)([x,x],training=True)
        np.testing.assert_array_equal(a.numpy(),b.numpy())
        self.assertEqual(int(np.count_nonzero(np.all(a.numpy()==0,axis=-1))),int(np.count_nonzero(np.all(x==0,axis=-1))))
    def test_grayscale_noise_does_not_create_color(self):
        from continuation.augmentation import UltrasoundAugment
        x=np.ones((1,16,16,3),'float32')*100
        result=UltrasoundAugment()(x,training=True).numpy()
        np.testing.assert_array_equal(result[...,0],result[...,1]);np.testing.assert_array_equal(result[...,1],result[...,2])
    def test_layer_serialization(self):
        from continuation.augmentation import UltrasoundAugment
        original=UltrasoundAugment(seed=19,gain=.1)
        restored=self.tf.keras.layers.deserialize(self.tf.keras.layers.serialize(original))
        x=np.ones((2,8,8,1),'float32')*100
        np.testing.assert_array_equal(original(x,training=True).numpy(),restored(x,training=True).numpy())
    def test_segmentation_masks_are_not_photometrically_modified(self):
        from continuation.anatomy import segmentation_data
        from pathlib import Path
        import tempfile
        from continuation.common import write_csv
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mask=np.zeros((32,32),'uint8');mask[10:20,9:23]=1
            np.savez(root/'a.npz',image=np.ones((32,32),'uint8')*120,ovary=mask,follicle=mask,instances=mask,distance=mask.astype('float32'))
            write_csv(root/'manifest.csv',[{'file':'a.npz','split':'train','instance_valid':1}])
            c=config(overrides={'seg_size':32,'batch_size':1,'data_threads':1})
            x,y,w=next(iter(segmentation_data(root,'train',c)))
            for name in ['ovary','follicle','distance']:
                self.assertEqual(float(y[name].numpy().sum()),float(mask.sum()))
                self.assertTrue(set(np.unique(y[name].numpy()))<=set([0,1]))
            self.assertFalse(np.allclose(x.numpy(),120/255))
    def test_bulk_run_covers_every_fit_file_and_never_reads_heldout(self):
        from continuation.augmentation_preview import bulk_check
        from continuation.common import write_csv,file_hash
        from pathlib import Path
        from PIL import Image
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.fromarray(np.ones((16,16,3),'uint8')*120).save(root/'a.png')
            records=[{'record_id':str(i),'relative_path':'a.png','class_name':'Normal',
                'training_role':'fit','sha256':file_hash(root/'a.png')} for i in range(2)]
            # A deliberately nonexistent held-out file must not be opened.
            records.append({'record_id':'protected','relative_path':'heldout-do-not-read.png',
                'class_name':'Normal','training_role':'test','sha256':'unused'})
            write_csv(root/'metadata/dataset_manifest.csv',records)
            report=bulk_check(root,root/'audit',config(overrides={'image_size':16}),4,2)
            self.assertEqual(report['fitting_image_files_processed'],2)
            self.assertEqual(report['augmented_views_generated'],8)
            self.assertEqual(report['stored_augmented_training_copies'],0)

if __name__=='__main__':unittest.main()
