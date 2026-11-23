"""Fit new readouts to frozen learned features; fit/select only, no test access."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib

def score(y,p):return {'balanced_accuracy':float(balanced_accuracy_score(y,p)),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'confusion_matrix':confusion_matrix(y,p,labels=range(4)).tolist()}
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def extractor_for(model):
 import tensorflow as tf
 layers=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense) and l.units==256]
 if len(layers)!=1:raise ValueError('Unexpected image-only head topology')
 return tf.keras.Model(model.input,[layers[0].input,layers[0].output])
def main():
 p=argparse.ArgumentParser()
 for n in ['code','classification','run','freeze','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
 import sys;sys.path.insert(0,str(a.code))
 import tensorflow as tf
 from continuation.common import role_rows,runtime,file_hash,CLASSES
 from continuation.models import build_classifier
 from continuation.data import load_classifier_image,training_weights
 from continuation.artifacts import inference_signature
 config=json.loads((a.run/'config.json').read_text());meta=json.loads((a.run/'data.json').read_text());freeze=json.loads(a.freeze.read_text())['freeze']
 assert config['mode']=='image' and config['domain_weight']==0
 assert inference_signature(a.run)==freeze['models']['image_only']['signature']
 assert file_hash(a.classification/'metadata/dataset_manifest.csv')==freeze['manifest_sha256']=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
 runtime(config);tf.config.experimental.enable_op_determinism()
 fit=role_rows(a.classification,'fit');val=role_rows(a.classification,'select')
 assert len(fit)==12763 and len(val)==2112
 assert not {r['image_group'] for r in fit}&{r['image_group'] for r in val}
 assert not {r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in val}
 declaration={'created_before_feature_extraction':True,'frozen_signature':inference_signature(a.run),'manifest_sha256':freeze['manifest_sha256'],'roles':['fit','select'],'fit_records':len(fit),'selection_records':len(val),'feature_sets':['encoder_pool','encoder_pool_plus_head'],'preprocessing':'StandardScaler and PCA128 fit on unique FIT images only, then per-image L2 normalization. All FIT records are used for supervised fitting.','classifiers':{'logistic_regression_C':[.1,1.,10.],'rbf_svm_C':[1.,10.,100.]},'sample_weights':'Existing duplicate and class weights, applied once.','selection_metric':'balanced_accuracy','max_seconds_after_worker_start':3600,'calibration_and_test_used':False,'scope':'Exploratory follow-up. Backbones already selected on this validation collection. No independent-confirmation claim.'}
 save(a.out/'declaration.json',declaration)
 model,_=build_classifier({**config,'weights':None},len(meta['domains']));model.load_weights(a.run/'best.weights.h5');extractor=extractor_for(model)
 representatives={r['pixel_sha256']:r for r in [*fit,*val]};keys=list(representatives);paths=np.array([str(a.classification/representatives[k]['relative_path']) for k in keys])
 ds=tf.data.Dataset.from_tensor_slices(paths).map(lambda x:load_classifier_image(x,config['image_size']),num_parallel_calls=4,deterministic=True).batch(16).prefetch(1)
 pools=[];heads=[]
 for i,x in enumerate(ds):
  pool,head=extractor({'image':x},training=False);pools.append(pool.numpy().astype('float32'));heads.append(head.numpy().astype('float32'))
  if i%50==0:print('FROZEN_FEATURE_BATCH',i,'/',(len(keys)+15)//16,flush=True)
 pool=np.concatenate(pools);head=np.concatenate(heads);index={k:i for i,k in enumerate(keys)}
 fi=np.array([index[r['pixel_sha256']] for r in fit]);vi=np.array([index[r['pixel_sha256']] for r in val]);unique_fit=np.unique(fi)
 y=np.array([CLASSES.index(r['class_name']) for r in fit]);yv=np.array([CLASSES.index(r['class_name']) for r in val]);weights=training_weights(fit,config)
 np.savez_compressed(a.out/'frozen_features.npz',pixel_sha256=np.array(keys),encoder_pool=pool,head_features=head,fit_indices=fi,selection_indices=vi,fit_record_ids=np.array([r['record_id'] for r in fit]),selection_record_ids=np.array([r['record_id'] for r in val]),fit_y=y,selection_y=yv)
 del model,extractor;tf.keras.backend.clear_session()
 results={};best=-1.
 with threadpool_limits(limits=4):
  for name,features in [('encoder_pool',pool),('encoder_pool_plus_head',np.concatenate([pool,head],axis=1))]:
   scaler=StandardScaler().fit(features[unique_fit]);scaled=scaler.transform(features)
   pca=PCA(n_components=min(128,len(unique_fit)-1,features.shape[1]),svd_solver='randomized',random_state=2627).fit(scaled[unique_fit]);normalizer=Normalizer();x=normalizer.transform(pca.transform(scaled));xf=x[fi];xv=x[vi]
   for kind,grid in [('logistic',[.1,1.,10.]),('rbf_svm',[1.,10.,100.])]:
    for C in grid:
     if time.monotonic()-started>3600:break
     label=name+'_'+kind+'_C'+str(C);print('READOUT_FITTING',label,flush=True)
     clf=LogisticRegression(C=C,max_iter=2000,solver='lbfgs') if kind=='logistic' else SVC(C=C,gamma='scale',kernel='rbf',break_ties=True,cache_size=512)
     begin=time.monotonic();clf.fit(xf,y,sample_weight=weights);pred=clf.predict(xv)
     assert list(clf.classes_)==[0,1,2,3]
     result={'fit':score(y,clf.predict(xf)),'selection':score(yv,pred),'elapsed_fit_seconds':time.monotonic()-begin,'feature_set':name,'classifier':kind,'C':C};results[label]=result
     save(a.out/'readout_results.json',{'candidates':results,'test_evaluated':False})
     value=result['selection']['balanced_accuracy'];print('READOUT_RESULT',label,json.dumps(result),flush=True)
     if value>best:
      best=value;joblib.dump({'scaler':scaler,'pca':pca,'normalizer':normalizer,'classifier':clf,'feature_set':name,'frozen_signature':declaration['frozen_signature'],'classes':CLASSES},a.out/'selected_readout.joblib')
      np.savez_compressed(a.out/'selected_validation_predictions.npz',record_ids=np.array([r['record_id'] for r in val]),true_class_indices=yv,predictions=pred,decision_scores=clf.decision_function(xv));save(a.out/'selected_readout.json',{'candidate':label,**result,'scope':'Validation only; decision scores are not calibrated confidence.'})
 save(a.out/'status.json',{'stage':'complete','selected_validation_balanced_accuracy':best,'elapsed_seconds':time.monotonic()-started,'test_evaluated':False,'source_sha256':file_hash(Path(__file__))})
 print('FROZEN_READOUT_COMPLETE',best,flush=True)
if __name__=='__main__':main()
