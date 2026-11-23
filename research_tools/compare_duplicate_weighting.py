"""One fixed readout; compare duplicate-weight exponents with every FIT record retained."""
from pathlib import Path
import sys,json,hashlib,datetime
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import sklearn,joblib
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'duplicate_weighting_study';O.mkdir(exist_ok=True)
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy/accuracy_development')
from ultrasound_manifest import rows_for,fitting_weights,CLASSES
root=B/'Classification';fit=rows_for(root,'fit');val=rows_for(root,'select');path=D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz';z=np.load(path,allow_pickle=False)
assert len(fit)==12763 and len(val)==2112
np.testing.assert_array_equal(z['fit_record_ids'],[r['record_id'] for r in fit]);np.testing.assert_array_equal(z['selection_record_ids'],[r['record_id'] for r in val])
assert hashlib.sha256((root/'metadata/dataset_manifest.csv').read_bytes()).hexdigest()=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
assert not set(z['pixel_sha256'][z['fit_indices']])&set(z['pixel_sha256'][z['selection_indices']])
declaration={'created_before_fitting':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fit_records':len(fit),'selection_records':len(val),'features_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'features':'Fixed previously selected encoder_pool_plus_head readout; unique FIT StandardScaler/PCA128, then L2 normalization','classifier':'LogisticRegression C=.1 max_iter=2000 lbfgs; no hyperparameter search','duplicate_exponents':[.5,1.],'class_balance':'Recalculate existing capped inverse-effective-class weights on FIT only; apply once','labels_or_records_removed_or_changed':False,'test_and_calibration_used':False,'sklearn_version':sklearn.__version__,'scope':'Exploratory selection study. A new class-weight distribution changes with duplicate exponent; any effect belongs to the combined weighting rule, not exclusively to duplicate scaling.'}
(O/'declaration.json').write_text(json.dumps(declaration,indent=2));y=z['fit_y'];yv=z['selection_y'];features=np.concatenate([z['encoder_pool'],z['head_features']],axis=1);fi=z['fit_indices'];vi=z['selection_indices'];uf=np.unique(fi)
with threadpool_limits(limits=4):
 scaler=StandardScaler().fit(features[uf]);scaled=scaler.transform(features);pca=PCA(n_components=128,svd_solver='randomized',random_state=2627).fit(scaled[uf]);normalizer=Normalizer();x=normalizer.transform(pca.transform(scaled));results={};counts=Counter(r['pixel_sha256'] for r in fit)
 for exponent in [.5,1.]:
  base=np.array([counts[r['pixel_sha256']]**(-exponent) for r in fit]);effective=np.bincount(y,weights=base,minlength=4);cw=np.minimum(base.sum()/np.maximum(4*effective,1),4.);w=base*cw[y];w=(w/w.mean()).astype('float32')
  if exponent==.5:np.testing.assert_allclose(w,fitting_weights(fit),rtol=1e-7,atol=1e-7)
  assert np.isfinite(w).all() and (w>0).all() and len(w)==12763
  clf=LogisticRegression(C=.1,max_iter=2000,solver='lbfgs');clf.fit(x[fi],y,sample_weight=w);pred=clf.predict(x[vi]);key='exponent_'+str(exponent)
  results[key]={'selection_balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'selection_macro_f1':float(f1_score(yv,pred,average='macro')),'selection_confusion_matrix':confusion_matrix(yv,pred,labels=range(4)).tolist(),'fit_balanced_accuracy':float(balanced_accuracy_score(y,clf.predict(x[fi]))),'fit_records_with_positive_weight':len(w)}
  joblib.dump({'scaler':scaler,'pca':pca,'normalizer':normalizer,'classifier':clf,'classes':CLASSES,'declaration':declaration},O/(key+'.joblib'))
  np.savez_compressed(O/(key+'_validation.npz'),record_ids=z['selection_record_ids'],true_class_indices=yv,predictions=pred,decision_scores=clf.decision_function(x[vi]));print(key,json.dumps(results[key]),flush=True)
(O/'results.json').write_text(json.dumps(results,indent=2))
