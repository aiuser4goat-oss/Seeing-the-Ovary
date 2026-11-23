"""Predeclared paired, frozen-feature linear-head experiment; no test access."""
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib

def main():
 p=argparse.ArgumentParser()
 for n in ['features','code','classification','run','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.code))
 from accuracy_development.group_targets import aggregate_targets,objective_gradient
 from continuation.common import role_rows,CLASSES,file_hash
 from continuation.data import training_weights
 assert file_hash(a.classification/'metadata/dataset_manifest.csv')=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
 fit=role_rows(a.classification,'fit');val=role_rows(a.classification,'select')
 z=np.load(a.features,allow_pickle=False);fi=z['fit_indices'];vi=z['selection_indices']
 assert len(fit)==12763 and len(val)==2112
 assert not {r['image_group'] for r in fit}&{r['image_group'] for r in val}
 assert np.all(np.isfinite(z['encoder_pool']))
 assert list(z['fit_record_ids'])==[r['record_id'] for r in fit]
 assert list(z['selection_record_ids'])==[r['record_id'] for r in val]
 assert not set(z['pixel_sha256'][fi])&set(z['pixel_sha256'][vi])
 y=np.array([CLASSES.index(r['class_name']) for r in fit]);yv=np.array([CLASSES.index(r['class_name']) for r in val])
 np.testing.assert_array_equal(y,z['fit_y']);np.testing.assert_array_equal(yv,z['selection_y'])
 config=json.loads((a.run/'config.json').read_text());weights=training_weights(fit,config)
 grouped=aggregate_targets(z['pixel_sha256'][fi],y,weights)
 declaration={'source_sha256':file_hash(Path(__file__)),'helper_sha256':file_hash(a.code/'accuracy_development/group_targets.py'),'features_sha256':file_hash(a.features),'roles':['fit','select'],'input':'Frozen image_only encoder pool; unique FIT StandardScaler/PCA128 and L2 normalization','objectives':['soft_ce','candidate_marginal'],'l2_penalty':.001,'optimizer':'L-BFGS-B; maxiter=300; same zero initialization; candidate marginal objective is nonconvex, so this is one controlled start, not a global-optimum claim','all_fit_records_represented':len(fit),'selection_only_for_comparison':True,'test_and_calibration_access':False,'assumption':'Candidate marginal loss treats supplied FIT classes for the identical input as alternatives. This is an experimental modeling assumption, not verified clinical truth or a modification of original record labels. Marginal loss discards within-set target proportions while retaining group mass and candidate membership. Groups with every class have no marginal-loss supervision.'}
 (a.out/'declaration.json').write_text(json.dumps(declaration,indent=2))
 with threadpool_limits(limits=4):
  features=z['encoder_pool'];uf=np.unique(fi);scaler=StandardScaler().fit(features[uf]);scaled=scaler.transform(features)
  pca=PCA(n_components=min(128,len(uf)-1,features.shape[1]),svd_solver='randomized',random_state=2627).fit(scaled[uf]);normalizer=Normalizer();x=normalizer.transform(pca.transform(scaled)).astype('float64')
  index={k:i for i,k in enumerate(z['pixel_sha256'])};gi=[index[k] for k in grouped['keys']]
  xg=np.c_[x[gi],np.ones(len(gi))];xv=np.c_[x[vi],np.ones(len(vi))];xf=np.c_[x[fi],np.ones(len(fi))]
  shape=(xg.shape[1],4);reports={}
  for kind in declaration['objectives']:
   targets=grouped['soft_targets'] if kind=='soft_ce' else grouped['candidate_mask']
   def objective(flat):
    w=flat.reshape(shape);loss,grad=objective_gradient(xg@w,targets,grouped['group_weights'],kind)
    penalty=.001*np.sum(w[:-1]**2)/2;gw=xg.T@grad;gw[:-1]+=.001*w[:-1]
    return loss+penalty,gw.ravel()
   started=time.monotonic();opt=minimize(objective,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-10,'gtol':1e-7})
   w=opt.x.reshape(shape);pred=np.argmax(xv@w,axis=1)
   reports[kind]={'selection_balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'selection_macro_f1':float(f1_score(yv,pred,average='macro')),'selection_confusion_matrix':confusion_matrix(yv,pred,labels=range(4)).tolist(),'fit_balanced_accuracy':float(balanced_accuracy_score(y,np.argmax(xf@w,axis=1))),'optimizer_converged':bool(opt.success),'optimizer_message':str(opt.message),'iterations':int(opt.nit),'elapsed_seconds':time.monotonic()-started}
   joblib.dump({'scaler':scaler,'pca':pca,'normalizer':normalizer,'weights':w,'classes':CLASSES,'declaration':declaration},a.out/(kind+'.joblib'))
   np.savez_compressed(a.out/(kind+'_validation.npz'),record_ids=z['selection_record_ids'],true_class_indices=yv,predictions=pred,decision_scores=xv@w)
   (a.out/'results.json').write_text(json.dumps(reports,indent=2));print(kind,json.dumps(reports[kind]),flush=True)
if __name__=='__main__':main()
