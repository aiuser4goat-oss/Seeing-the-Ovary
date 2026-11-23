from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy')
from continuation.common import role_rows,runtime,CLASSES
from continuation.anatomy import build_segmenter
from accuracy_development.anatomy_safety import guarded_postprocess
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');R=B/'Classification';S=B/'Colab_Transfer/recovered/ovarian_continuation/segmentation';O=B/'Accuracy_Development_2026_11_22/anatomy_audit/classification_sample';O.mkdir(exist_ok=True)
c=json.loads((S/'config.json').read_text());c.update(data_threads=1);runtime(c)
model=build_segmenter(c);model.load_weights(S/'best.weights.h5');c.update(follicle_instance_method='connected_components',follicle_threshold=.8,min_follicle_area=30)
rows=[]
for role in ['fit','select']:
 candidates=role_rows(R,role)
 for name in CLASSES:
  unique={r['pixel_sha256']:r for r in candidates if r['class_name']==name}
  rows.extend(sorted(unique.values(),key=lambda r:hashlib.sha256((r['pixel_sha256']+'crop-audit-v1').encode()).hexdigest())[:2])
report=[]
for i,r in enumerate(rows):
 with Image.open(R/r['relative_path']) as f:im=f.convert('RGB')
 x=np.asarray(ImageOps.pad(im.convert('L'),(256,256),method=Image.Resampling.BILINEAR),dtype='float32')/255
 pred={k:v[0,...,0].numpy() for k,v in model(x[None,...,None],training=False).items()}
 crop,vector,ov,ins,decision=guarded_postprocess(im,pred,c)
 im.save(O/(str(i)+'_original.png'));crop.save(O/(str(i)+'_guarded_crop.png'))
 np.savez_compressed(O/(str(i)+'_predictions.npz'),**pred,ovary_mask=ov,follicle_instances=ins)
 report.append({'index':i,'record_id':r['record_id'],'pixel_sha256':r['pixel_sha256'],'role':r['training_role'],'provided_class':r['class_name'],'source':r['source'],'guard':decision,'predicted_visible_count':None if not np.isfinite(vector[3]) else float(vector[3]),'anatomical_truth_available':False});print('CLASSIFICATION_CROP_AUDIT',i+1,'/',len(rows),flush=True)
(O/'audit.json').write_text(json.dumps({'scope':'Deterministic stratified FIT/SELECT sample. Predicted masks only; no independent anatomical truth. Provided classification targets do not validate the anatomy. No test/calibration inputs.','checkpoint_sha256':hashlib.sha256((S/'best.weights.h5').read_bytes()).hexdigest(),'records':report},indent=2))
