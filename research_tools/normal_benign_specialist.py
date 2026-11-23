"""Predeclared routed binary specialist with FIT-only grouped head-error mining."""
from pathlib import Path
import csv,datetime,hashlib,json,sys,time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib,sklearn
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'normal_benign_specialist_v1';O.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy/accuracy_development')
from ultrasound_manifest import rows_for,fitting_weights,CLASSES
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=B/'Classification';fit=rows_for(root,'fit');val=rows_for(root,'select');assert len(fit)==12763 and len(val)==2112
assert sha(root/'metadata/dataset_manifest.csv')=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
features_path=D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz';assert sha(features_path)=='4c516c2831268648104d4f065ad0f6374b6dd8b7b537e34182136c8123825e9a';z=np.load(features_path,allow_pickle=False)
base_path=D/'selection_tta_cpu_v2/selection_views.npz';base=np.load(base_path,allow_pickle=False)
for names,records in [('fit',fit),('selection',val)]:
 np.testing.assert_array_equal(z[names+'_record_ids'],[r['record_id'] for r in records]);np.testing.assert_array_equal(z['fit_y' if names=='fit' else 'selection_y'],[CLASSES.index(r['class_name']) for r in records])
np.testing.assert_array_equal(base['record_ids'],z['selection_record_ids']);np.testing.assert_array_equal(base['true_class_indices'],z['selection_y']);np.testing.assert_array_equal(base['classes'],CLASSES)
assert not {r['image_group'] for r in fit}&{r['image_group'] for r in val}
assert not {r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in val}
features=np.concatenate([z['encoder_pool'],z['head_features']],axis=1);fi=z['fit_indices'];vi=z['selection_indices'];all_y=z['fit_y'];indices=np.flatnonzero(np.isin(all_y,[0,2]));binary_rows=[fit[i] for i in indices];target=(all_y[indices]==2).astype(int);weights=fitting_weights(fit)[indices];weights=weights/weights.mean();feature_indices=fi[indices]
# Connected grouping prevents declared similarity groups AND byte/pixel copies crossing folds.
parent=list(range(len(indices)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
seen={}
for i,r in enumerate(binary_rows):
 for kind in ['image_group','pixel_sha256','sha256']:
  key=(kind,r[kind])
  if key in seen:parent[find(i)]=find(seen[key])
  else:seen[key]=i
groups=np.array([find(i) for i in range(len(indices))])
declaration={'declared_before_fitting_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_sha256':sha(__file__),'features_sha256':sha(features_path),'main_selection_cache_sha256':sha(base_path),'global_checkpoint_sha256':'aa809ac1c0eadca3e7b7e0591607df9c0ed5630a355c9650915b69ec811a3364','class_order':CLASSES,'main_runtime':'Deterministic CPU float32 image-only identity; historical GPU scores unchanged','fit_records_full_main':len(fit),'specialist_fit_records':len(indices),'selection_records':len(val),'specialist_classes':['Benign','Normal'],'threshold':.5,'routing':'Only main-predicted Benign may switch to Normal. All other predicted classes unchanged. No calibrated four-class probabilities claimed.','controls':['unchanged_main','plain_specialist','mined_specialist'],'specialist':'Pool1280+pre-LNhead256; StandardScaler/PCA128 fit on unique training-fold features, then L2. LogisticRegressionC.1 lbfgs maxiter2000. Final preprocessing identical for plain/mined.','mining':'5-fold StratifiedGroupKFold seed2627 on connected similarity/byte/pixel groups, only binaryFIT rows. Both Benign hard negatives and Normal hard positives misclassified out-of-fold receive2x original positive base weight once; normalize mean1. No rows removed/relabelled. Base weights inherited from fullFIT original weighting protocol.','cv_limit':'Conditional frozen-feature head mining only; backbone learned from allFIT. Not independent whole-neural-networkCV or an unbiased generalization estimate.','test_calibration_access':False,'statistics_plan':'2500 paired group bootstrap draws;5000 paired group label-swap permutations; Holm adjustment over main-vs-plain,main-vs-mined,plain-vs-mined. Exploratory after earlier SELECT review; no correction for all earlier research attempts.','selection_decision':'Primary balanced accuracy on all2112SELECTrows, macroF1 secondary; no threshold orCsearch.'};(O/'declaration.json').write_text(json.dumps(declaration,indent=2))

def preprocessing(train_indices):
 unique=np.unique(train_indices);assert len(unique)>128
 scaler=StandardScaler().fit(features[unique]);scaled=scaler.transform(features);pca=PCA(n_components=128,svd_solver='randomized',random_state=2627).fit(scaled[unique]);norm=Normalizer();x=norm.transform(pca.transform(scaled));return x,scaler,pca,norm

def score(pred):
 truth=z['selection_y'];m=confusion_matrix(truth,pred,labels=range(4));return {'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro')),'accuracy':float((truth==pred).mean()),'confusion_matrix':m.tolist(),'recall':dict(zip(CLASSES,(np.diag(m)/m.sum(1)).tolist()))}

oof=np.full(len(indices),np.nan);fold_id=np.full(len(indices),-1);folds=[];started=time.monotonic()
with threadpool_limits(limits=2):
 for fold,(train,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=2627).split(indices,target,groups)):
  assert set(np.unique(target[train]))==set(np.unique(target[held]))=={0,1}
  for kind in ['image_group','pixel_sha256','sha256']:assert not {binary_rows[i][kind] for i in train}&{binary_rows[i][kind] for i in held}
  x,_,_,_=preprocessing(feature_indices[train]);clf=LogisticRegression(C=.1,solver='lbfgs',max_iter=2000);w=weights[train];clf.fit(x[feature_indices[train]],target[train],sample_weight=w/w.mean());assert list(clf.classes_)==[0,1]
  oof[held]=clf.predict_proba(x[feature_indices[held]])[:,1];fold_id[held]=fold;folds.append({'fold':fold,'train':len(train),'held':len(held),'converged':bool(clf.n_iter_.max()<2000),'head_only_held_balanced_accuracy':float(balanced_accuracy_score(target[held],oof[held]>=.5))});print('SPECIALIST_FOLD',fold,json.dumps(folds[-1]),flush=True)
 assert np.isfinite(oof).all() and (fold_id>=0).all();hard=(oof>=.5)!=target;mined=weights*np.where(hard,2.,1.);mined/=mined.mean();assert (mined>0).all()
 x,scaler,pca,norm=preprocessing(feature_indices);main=base['view_probabilities'][:,0].argmax(1);predictions={'unchanged_main':main};probabilities={};results={'unchanged_main':score(main)}
 for name,w in [('plain_specialist',weights),('mined_specialist',mined)]:
  clf=LogisticRegression(C=.1,solver='lbfgs',max_iter=2000);clf.fit(x[feature_indices],target,sample_weight=w);assert list(clf.classes_)==[0,1];prob=clf.predict_proba(x[vi])[:,1];pred=main.copy();pred[(main==0)&(prob>=.5)]=2
  assert np.array_equal(pred[main!=0],main[main!=0]);truth=z['selection_y']
  for c in [1,3]:assert ((pred==truth)&(truth==c)).sum()==((main==truth)&(truth==c)).sum()
  assert ((pred==truth)&(truth==0)).sum()<=((main==truth)&(truth==0)).sum();assert ((pred==truth)&(truth==2)).sum()>=((main==truth)&(truth==2)).sum()
  predictions[name]=pred;probabilities[name]=prob;results[name]={**score(pred),'routed_Benign_to_Normal':int((pred!=main).sum()),'converged':bool(clf.n_iter_.max()<2000),'changes_by_true_class':{c:int(((pred!=main)&(truth==i)).sum()) for i,c in enumerate(CLASSES)}}
  joblib.dump({'scaler':scaler,'pca':pca,'normalizer':norm,'classifier':clf,'declaration':declaration,'feature_order':['encoder_pool','pre_layernorm_head'],'routing_threshold':.5},O/(name+'.joblib'));print(name,json.dumps(results[name]),flush=True)
np.savez_compressed(O/'selection_predictions.npz',record_ids=z['selection_record_ids'],true_class_indices=z['selection_y'],classes=np.array(CLASSES),main_probabilities=base['view_probabilities'][:,0],**predictions,**{name+'_normal_probability':p for name,p in probabilities.items()})
np.savez_compressed(O/'fit_mining.npz',record_ids=np.array([r['record_id'] for r in binary_rows]),binary_targets=target,oof_normal_probability=oof,fold=fold_id,base_weight=weights,mined_weight=mined,groups=groups)
(O/'results.json').write_text(json.dumps({'results':results,'folds':folds,'mining_records':int(hard.sum()),'mining_by_class':{'Benign':int((hard&(target==0)).sum()),'Normal':int((hard&(target==1)).sum())},'elapsed_seconds':time.monotonic()-started,'test_evaluated':False},indent=2));print('SPECIALIST_COMPLETE',flush=True)
