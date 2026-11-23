"""Locked identity versus four flips; SELECT only, no fitting or calibration/test inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'selection_tta_cpu_v2';O.mkdir(exist_ok=False)
CODE=B/'Kaggle_Launch/github-mirror-checkout/frozen_v2_3_1';sys.path.insert(0,str(CODE))
import tensorflow as tf
from continuation.common import role_rows,CLASSES,runtime
from continuation.models import build_classifier
from continuation.data import load_classifier_image
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=B/'Classification';run=B/'Preserved_Best_Completed_Version_2026_11_22/results/image_only'
assert sha(root/'metadata/dataset_manifest.csv')=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
assert sha(run/'best.weights.h5')=='aa809ac1c0eadca3e7b7e0591607df9c0ed5630a355c9650915b69ec811a3364'
config=json.loads((run/'config.json').read_text());config.update(weights=None,mixed_precision=False)
runtime(config);tf.config.experimental.enable_op_determinism();tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(4)
rows=role_rows(root,'select');assert len(rows)==2112
for r in rows:assert sha(root/r['relative_path'])==r['sha256']
keys=list(dict.fromkeys(r['sha256'] for r in rows));representatives={r['sha256']:r for r in rows};mapping={k:i for i,k in enumerate(keys)}
declaration={'before_inference':True,'worker_sha256':sha(__file__),'model_source_sha256':sha(CODE/'continuation/models.py'),'checkpoint_sha256':sha(run/'best.weights.h5'),'records':2112,'unique_encoded_files':len(keys),'tensorflow_version':tf.__version__,'numpy_version':np.__version__,'keras_version':tf.keras.__version__,'saved_config_sha256':sha(run/'config.json'),'roles':['select'],'test_calibration_access':False,'fitting':False,'class_order':CLASSES,'views':['identity','horizontal','vertical','both'],'fixed_comparison':'Identity versus equal average of all four views. No alternative subsets/weights selected.','runtime':'Local CPU float32 deterministic; separate matched runtime variant, not a replacement for historical GPU identity scores.','decision':'Descriptive matched SELECT balanced accuracy, macro F1 secondary. No classification TEST inference or automatic promotion.'}
(O/'declaration.json').write_text(json.dumps(declaration,indent=2));model,_=build_classifier(config);model.load_weights(run/'best.weights.h5');pred=[];start=time.monotonic()
for start_index in range(0,len(keys),8):
 x=tf.stack([load_classifier_image(root/representatives[k]['relative_path'],224) for k in keys[start_index:start_index+8]])
 batch=[]
 for axis in [None,[2],[1],[1,2]]:
  v=x if axis is None else tf.reverse(x,axis);p=model({'image':v},training=False).numpy().astype('float32');assert np.isfinite(p).all();batch.append(p)
 pred.append(np.stack(batch,axis=1))
 if start_index%80==0:print('TTA_SELECTION',start_index,'/',len(keys),flush=True)
probs=np.concatenate(pred);indices=np.array([mapping[r['sha256']] for r in rows]);full=probs[indices];truth=np.array([CLASSES.index(r['class_name']) for r in rows]);identity=full[:,0];tta=full.mean(1)
np.savez_compressed(O/'selection_views.npz',record_ids=np.array([r['record_id'] for r in rows]),true_class_indices=truth,classes=np.array(CLASSES),view_probabilities=full)
result={'status':'complete','elapsed_seconds':time.monotonic()-start,'test_evaluated':False,'results':{}}
for name,p in [('identity',identity),('four_flip_average',tta)]:
 y=p.argmax(1);result['results'][name]={'balanced_accuracy':float(balanced_accuracy_score(truth,y)),'macro_f1':float(f1_score(truth,y,average='macro')),'confusion_matrix':confusion_matrix(truth,y,labels=range(4)).tolist()}
result['balanced_accuracy_difference']=result['results']['four_flip_average']['balanced_accuracy']-result['results']['identity']['balanced_accuracy'];(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
