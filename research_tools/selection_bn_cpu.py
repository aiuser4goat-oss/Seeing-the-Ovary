"""Isolated FIT-only BN running-statistics adaptation; SELECT-only scoring, no gradients or protected inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'selection_bn_cpu_v1';O.mkdir(exist_ok=False)
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

control=D/'selection_tta_cpu_v2';assert (control/'result.json').is_file()
fit=role_rows(root,'fit');assert len(fit)==12763
for r in fit:assert sha(root/r['relative_path'])==r['sha256']
assert not {r['image_group'] for r in fit}&{r['image_group'] for r in rows}
assert not {r['sha256'] for r in fit}&{r['sha256'] for r in rows}
assert not {r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in rows}
control_decl=json.loads((control/'declaration.json').read_text());assert control_decl['checkpoint_sha256']==sha(run/'best.weights.h5') and control_decl['model_source_sha256']==sha(CODE/'continuation/models.py') and control_decl['worker_sha256']==sha(Path(__file__).with_name('selection_tta_cpu.py'))
cache=np.load(control/'selection_views.npz',allow_pickle=False);np.testing.assert_array_equal(cache['classes'],CLASSES);np.testing.assert_array_equal(cache['true_class_indices'],[CLASSES.index(r['class_name']) for r in rows]);np.testing.assert_array_equal(cache['record_ids'],[r['record_id'] for r in rows]);identity=cache['view_probabilities'][:,0];truth=cache['true_class_indices']
declaration={'before_statistics_update':True,'control_cache_sha256':sha(control/'selection_views.npz'),'control_declaration_sha256':sha(control/'declaration.json'),'worker_sha256':sha(__file__),'checkpoint_sha256':sha(run/'best.weights.h5'),'config_sha256':sha(run/'config.json'),'fit_records':len(fit),'selection_records':len(rows),'batch_size':32,'momentum':.9,'seed':2627,'passes':1,'statistics':'EMA batch estimates from saved initial statistics, not exact population moments; all FIT records once in fixed seeded shuffle. No sample weights or class labels in update.','source':'https://keras.io/api/layers/normalization_layers/batch_normalization/','test_calibration_access':False,'gradient_updates':False,'augmentation':False,'dropout':False,'precision':'CPU float32, matched TTA identity control, no TTA in treatment','tensorflow_version':tf.__version__,'numpy_version':np.__version__,'decision':'Predeclared single BN treatment versus identity SELECT BA; no momentum/batch search or automatic promotion.'}
(O/'declaration.json').write_text(json.dumps(declaration,indent=2))
model,encoder=build_classifier(config);model.load_weights(run/'best.weights.h5')
x=tf.stack([load_classifier_image(root/r['relative_path'],224) for r in rows[:8]])
p=model({'image':x},training=False).numpy();difference=float(np.max(abs(p-identity[:8])));assert difference<1e-6;np.testing.assert_array_equal(p.argmax(1),identity[:8].argmax(1))
encoder.trainable=True;bn=[l for l in encoder.layers if isinstance(l,tf.keras.layers.BatchNormalization)];assert bn
moving={id(v) for l in bn for v in [l.moving_mean,l.moving_variance]}
def learned_hashes():return {v.path:hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in model.weights if id(v) not in moving}
before=learned_hashes();means_before=[l.moving_mean.numpy().copy() for l in bn]
momenta=[l.momentum for l in bn]
for l in bn:l.trainable=True;l.momentum=.9
dropouts=[l for l in encoder.layers if isinstance(l,tf.keras.layers.Dropout)];rates=[l.rate for l in dropouts]
for l in dropouts:l.rate=0.
order=np.random.default_rng(2627).permutation(len(fit));started=time.monotonic()
for start in range(0,len(order),32):
 batch=order[start:start+32];x=tf.stack([load_classifier_image(root/fit[i]['relative_path'],224) for i in batch]);encoder(x,training=True)
 if start%1024==0:print('BN_FIT_PASS',start,'/',len(fit),flush=True)
for l,rate in zip(dropouts,rates):l.rate=rate
for l,momentum in zip(bn,momenta):l.momentum=momentum
encoder.trainable=False;assert learned_hashes()==before,'Learned parameters changed unexpectedly'
changed=sum(not np.array_equal(old,l.moving_mean.numpy()) for old,l in zip(means_before,bn));assert changed>0
pred=[]
for start in range(0,len(keys),8):
 x=tf.stack([load_classifier_image(root/representatives[k]['relative_path'],224) for k in keys[start:start+8]]);pred.append(model({'image':x},training=False).numpy().astype('float32'))
lookup=np.concatenate(pred);prob=lookup[[mapping[r['sha256']] for r in rows]];assert np.isfinite(prob).all()
np.savez_compressed(O/'selection_predictions.npz',record_ids=cache['record_ids'],classes=np.array(CLASSES),true_class_indices=truth,probabilities=prob)
result={'status':'complete','initial_max_difference':difference,'bn_layers':len(bn),'bn_means_changed':changed,'learned_parameters_unchanged':True,'fit_records_seen':len(fit),'elapsed_seconds':time.monotonic()-started,'test_evaluated':False,'results':{}}
for name,p in [('identity',identity),('bn_reestimated',prob)]:
 y=p.argmax(1);result['results'][name]={'balanced_accuracy':float(balanced_accuracy_score(truth,y)),'macro_f1':float(f1_score(truth,y,average='macro')),'confusion_matrix':confusion_matrix(truth,y,labels=range(4)).tolist()}
model.save_weights(O/'bn.weights.h5');result['checkpoint_sha256']=sha(O/'bn.weights.h5');(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
