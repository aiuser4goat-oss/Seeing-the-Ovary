"""Cheap matched source-balancing screen on frozen features, not end-to-end retraining."""
from pathlib import Path
import sys,json,hashlib,datetime
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'source_balance_readout_v1';O.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy/accuracy_development')
from ultrasound_manifest import rows_for,fitting_weights,CLASSES
root=B/'Classification';fit=rows_for(root,'fit');val=rows_for(root,'select');path=D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz';z=np.load(path,allow_pickle=False)
assert len(fit)==12763 and len(val)==2112
np.testing.assert_array_equal(z['fit_record_ids'],[r['record_id'] for r in fit]);np.testing.assert_array_equal(z['selection_record_ids'],[r['record_id'] for r in val]);np.testing.assert_array_equal(z['fit_y'],[CLASSES.index(r['class_name']) for r in fit]);np.testing.assert_array_equal(z['selection_y'],[CLASSES.index(r['class_name']) for r in val])
assert hashlib.sha256((root/'metadata/dataset_manifest.csv').read_bytes()).hexdigest()=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
assert not set(z['pixel_sha256'][z['fit_indices']])&set(z['pixel_sha256'][z['selection_indices']])
y=z['fit_y'];yv=z['selection_y'];w=fitting_weights(fit);source=np.array([r['source'] for r in fit]);treatment=w.copy().astype('float64');mass={}
for c in range(4):
 mask=y==c;cells={s:float(w[mask&(source==s)].sum()) for s in sorted(set(source[mask]))};target=sum(cells.values())/len(cells);factor=np.array([min(target/cells[s],4.) for s in source[mask]]);treatment[mask]*=factor;treatment[mask]*=w[mask].sum()/treatment[mask].sum();mass[CLASSES[c]]={'original_cells':cells,'balanced_cells':{s:float(treatment[mask&(source==s)].sum()) for s in cells}}
assert np.isfinite(treatment).all() and (treatment>0).all()
for c in range(4):np.testing.assert_allclose(treatment[y==c].sum(),w[y==c].sum(),rtol=1e-6)
decl={'before_fitting':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'features_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fit_records':len(fit),'selection_records':len(val),'test_calibration_used':False,'recipe':'Fixed pool+head StandardScaler/PCA128 on uniqueFIT/L2; logisticC.1,lbfgs,maxiter2000. Original weighting control versus within-class source balancing. Raw source cell factor capped4, then class mass restored exactly. All FIT records retain positive influence. No source input at inference.','scope':'Exploratory frozen-representation screen, not end-to-end source-balanced sampling. Cannot establish source-invariant generalization. Source IDs are collection folders. No hyperparameter search.','mass':mass};(O/'declaration.json').write_text(json.dumps(decl,indent=2))
features=np.concatenate([z['encoder_pool'],z['head_features']],axis=1);fi=z['fit_indices'];vi=z['selection_indices'];uf=np.unique(fi);results={}
with threadpool_limits(limits=2):
 scaler=StandardScaler().fit(features[uf]);scaled=scaler.transform(features);pca=PCA(n_components=128,svd_solver='randomized',random_state=2627).fit(scaled[uf]);norm=Normalizer();x=norm.transform(pca.transform(scaled))
 for name,weights in [('control',w),('source_balanced',treatment.astype('float32'))]:
  clf=LogisticRegression(C=.1,max_iter=2000,solver='lbfgs');clf.fit(x[fi],y,sample_weight=weights);pred=clf.predict(x[vi]);result={'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,average='macro')),'confusion_matrix':confusion_matrix(yv,pred,labels=range(4)).tolist(),'by_source':{}}
  for s in sorted({r['source'] for r in val}):
   idx=np.array([r['source']==s for r in val]);result['by_source'][s]={'records':int(idx.sum()),'correct':int((pred[idx]==yv[idx]).sum()),'class_recall':{c:(float((pred[idx&(yv==i)]==i).mean()) if (idx&(yv==i)).any() else None) for i,c in enumerate(CLASSES)}}
  results[name]=result;np.savez_compressed(O/(name+'.npz'),record_ids=z['selection_record_ids'],true_class_indices=yv,probabilities=clf.predict_proba(x[vi]));joblib.dump({'scaler':scaler,'pca':pca,'normalizer':norm,'classifier':clf,'classes':CLASSES},O/(name+'.joblib'));print(name,json.dumps(result),flush=True)
(O/'results.json').write_text(json.dumps(results,indent=2))
