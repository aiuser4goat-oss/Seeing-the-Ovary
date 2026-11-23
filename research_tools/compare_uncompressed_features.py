"""Check whether prior PCA compression removed useful classification information."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'uncompressed_feature_study';O.mkdir(exist_ok=True)
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy');sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy/accuracy_development')
from ultrasound_manifest import rows_for,fitting_weights,CLASSES
root=B/'Classification';run=B/'Preserved_Best_Completed_Version_2026_11_22/results/image_only';p=D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz';z=np.load(p,allow_pickle=False);fit=rows_for(root,'fit');val=rows_for(root,'select');assert len(fit)==12763 and len(val)==2112
np.testing.assert_array_equal(z['fit_record_ids'],[r['record_id'] for r in fit]);np.testing.assert_array_equal(z['selection_record_ids'],[r['record_id'] for r in val]);assert not set(z['pixel_sha256'][z['fit_indices']])&set(z['pixel_sha256'][z['selection_indices']])
assert hashlib.sha256((run/'best.weights.h5').read_bytes()).hexdigest()=='aa809ac1c0eadca3e7b7e0591607df9c0ed5630a355c9650915b69ec811a3364'
declaration={'declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_fitting':True,'fit_records':12763,'select_records':2112,'test_and_calibration_used':False,'feature_sets':['full_encoder_pool1280','actual_original_normalized_head256'],'preprocessing':'Unique-FIT StandardScaler then per-image L2. No PCA.','fixed_C':[.1,1.,10.],'classifier':'lbfgs LogisticRegression max_iter2000','weights':'Unchanged original inverse-sqrt duplicate and capped class weights, applied once','scope':'Exploratory selection comparison after earlier compressed-feature study. No test inference. Backbone unchanged.','features_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};(O/'declaration.json').write_text(json.dumps(declaration,indent=2))
import tensorflow as tf
from continuation.models import build_classifier
tf.keras.mixed_precision.set_global_policy('float32');config=json.loads((run/'config.json').read_text());meta=json.loads((run/'data.json').read_text());model,_=build_classifier({**config,'weights':None},len(meta['domains']));model.load_weights(run/'best.weights.h5');norms=[l for l in model.layers if isinstance(l,tf.keras.layers.LayerNormalization)];assert len(norms)==1
head=norms[0](z['head_features'],training=False).numpy();original=model.get_layer('class')(head,training=False).numpy();fi=z['fit_indices'];vi=z['selection_indices'];y=z['fit_y'];yv=z['selection_y'];weights=fitting_weights(fit)
reference=np.load(D/'results_v1/warm_b0/Ovarian Continuation Results/accuracy_development/references/image_only/selection_probabilities.npz',allow_pickle=False);np.testing.assert_array_equal(reference['record_ids'],z['selection_record_ids']);receipt={'scope':'CPU float32 head reconstruction from frozen CPU embeddings, not a new full-image inference or new model','selection_balanced_accuracy':float(balanced_accuracy_score(yv,original[vi].argmax(1))),'max_probability_difference_from_prior_GPU_reference':float(np.max(np.abs(original[vi]-reference['raw_probabilities']))),'changed_top_classes':int(np.count_nonzero(original[vi].argmax(1)!=reference['raw_probabilities'].argmax(1)))};(O/'head_reconstruction.json').write_text(json.dumps(receipt,indent=2));del model;tf.keras.backend.clear_session();results={}
with threadpool_limits(limits=4):
 for name,features in [('full_encoder_pool1280',z['encoder_pool']),('actual_original_normalized_head256',head)]:
  scaler=StandardScaler().fit(features[np.unique(fi)]);normalizer=Normalizer();x=normalizer.transform(scaler.transform(features))
  for C in [.1,1.,10.]:
   clf=LogisticRegression(C=C,max_iter=2000,solver='lbfgs');clf.fit(x[fi],y,sample_weight=weights);pred=clf.predict(x[vi]);key=name+'_C'+str(C)
   results[key]={'selection_balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'selection_macro_f1':float(f1_score(yv,pred,average='macro')),'confusion_matrix':confusion_matrix(yv,pred,labels=range(4)).tolist(),'fit_balanced_accuracy':float(balanced_accuracy_score(y,clf.predict(x[fi])))}
   joblib.dump({'scaler':scaler,'normalizer':normalizer,'classifier':clf,'classes':CLASSES,'feature_set':name},O/(key+'.joblib'));np.savez_compressed(O/(key+'_validation.npz'),record_ids=z['selection_record_ids'],true_class_indices=yv,predictions=pred,decision_scores=clf.decision_function(x[vi]));print(key,json.dumps(results[key]),flush=True)
(O/'results.json').write_text(json.dumps(results,indent=2));print('HEAD_RECONSTRUCTION',json.dumps(receipt),flush=True)
