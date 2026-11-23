"""One predeclared GCE(q=.7) versus CE frozen-feature screen; no label editing."""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'robust_loss_readout_v1';O.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy/accuracy_development')
from ultrasound_manifest import rows_for,fitting_weights,CLASSES

def loss_gradient(flat,x,y,weights,q,regularization):
 w=flat.reshape(x.shape[1],4);logp=x@w;logp-=logsumexp(logp,axis=1,keepdims=True);p=np.exp(logp);logpy=logp[np.arange(len(y)),y]
 if q is None:loss=-logpy;factor=np.ones(len(y))
 else:factor=np.exp(q*logpy);loss=-np.expm1(q*logpy)/q
 grad=p.copy();grad[np.arange(len(y)),y]-=1;grad*=((weights*factor)/weights.sum())[:,None];gw=x.T@grad;gw[:-1]+=regularization*w[:-1]
 return float(np.dot(weights,loss)/weights.sum()+regularization*np.sum(w[:-1]**2)/2),gw.ravel()
# Finite difference verification before processing project features.
rng=np.random.default_rng(2627);tx=np.c_[rng.normal(size=(7,3)),np.ones(7)];ty=np.array([0,1,2,3,0,1,2]);tw=np.arange(1,8,dtype=float);theta=rng.normal(scale=.1,size=16)
for q in [None,.7]:
 _,g=loss_gradient(theta,tx,ty,tw,q,.01)
 for k in range(len(theta)):
  plus=theta.copy();minus=theta.copy();plus[k]+=1e-6;minus[k]-=1e-6;numeric=(loss_gradient(plus,tx,ty,tw,q,.01)[0]-loss_gradient(minus,tx,ty,tw,q,.01)[0])/2e-6;np.testing.assert_allclose(g[k],numeric,atol=1e-7,rtol=1e-4)
fit=rows_for(B/'Classification','fit');val=rows_for(B/'Classification','select');assert len(fit)==12763 and len(val)==2112
p=D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz';z=np.load(p,allow_pickle=False);np.testing.assert_array_equal(z['fit_record_ids'],[r['record_id'] for r in fit]);np.testing.assert_array_equal(z['selection_record_ids'],[r['record_id'] for r in val]);np.testing.assert_array_equal(z['fit_y'],[CLASSES.index(r['class_name']) for r in fit]);assert not set(z['pixel_sha256'][z['fit_indices']])&set(z['pixel_sha256'][z['selection_indices']])
# Reuse fixed, FIT-only preprocessing from the already matched source-balance control.
control=D/'source_balance_readout_v1/control.joblib';bundle=joblib.load(control);declaration={'before_fitting':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'features_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'preprocessing_control_sha256':hashlib.sha256(control.read_bytes()).hexdigest(),'roles':['fit','select'],'fit_records':len(fit),'selection_records':len(val),'objectives':['cross_entropy','generalized_cross_entropy_q0.7'],'source':'https://proceedings.neurips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html','optimizer':'L-BFGS-B,maxiter300,zero initialization,identical L2 for both; GCE nonconvex, one fixed start.','preprocessing':'Same FIT-only pool+head/PCA128/L2 as previous control.','labels_changed':False,'original_positive_weights_retained':True,'test_calibration_access':False,'gradient_tests':'32 finite-difference coordinates passed','scope':'Frozen-representation hypothesis screen, not end-to-end CNN replication. GCE changes loss scale and implicit regularization as well as hard-sample influence. No assertion that high-loss source labels are wrong.'};(O/'declaration.json').write_text(json.dumps(declaration,indent=2))
features=np.concatenate([z['encoder_pool'],z['head_features']],axis=1);weights=fitting_weights(fit);results={}
with threadpool_limits(limits=2):
 x=bundle['normalizer'].transform(bundle['pca'].transform(bundle['scaler'].transform(features)));x=np.c_[x,np.ones(len(x))].astype('float64');xf=x[z['fit_indices']];xv=x[z['selection_indices']];y=z['fit_y'];yv=z['selection_y'];penalty=1/(.1*weights.sum())
 for name,q in [('cross_entropy',None),('gce_q0.7',.7)]:
  opt=minimize(lambda v:loss_gradient(v,xf,y,weights,q,penalty),np.zeros(x.shape[1]*4),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-10,'gtol':1e-7});w=opt.x.reshape(x.shape[1],4);pred=(xv@w).argmax(1);results[name]={'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,average='macro')),'confusion_matrix':confusion_matrix(yv,pred,labels=range(4)).tolist(),'converged':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit)};np.savez_compressed(O/(name+'.npz'),record_ids=z['selection_record_ids'],true_class_indices=yv,logits=xv@w,weights=w);print(name,json.dumps(results[name]),flush=True)
(O/'results.json').write_text(json.dumps(results,indent=2))
