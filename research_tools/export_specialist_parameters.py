from pathlib import Path
import json,joblib,numpy as np
from scipy.special import expit
D=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets/Accuracy_Development_2026_11_22');S=D/'normal_benign_specialist_v1';z=np.load(D/'readout_cpu_v3/Ovarian Continuation Results/frozen_feature_study/frozen_features.npz',allow_pickle=False);f=np.concatenate([z['encoder_pool'],z['head_features']],axis=1)[z['selection_indices']];saved=np.load(S/'selection_predictions.npz',allow_pickle=False);receipt={}
for name in ['plain_specialist','mined_specialist']:
 b=joblib.load(S/(name+'.joblib'));sc=b['scaler'];pc=b['pca'];clf=b['classifier'];assert sc.with_mean and sc.with_std and not pc.whiten and b['normalizer'].norm=='l2';assert list(clf.classes_)==[0,1]
 arrays={'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,'pca_components':pc.components_,'pca_mean':pc.mean_,'coefficient':clf.coef_,'intercept':clf.intercept_,'metadata':np.array(json.dumps({'declaration':b['declaration'],'feature_order':b['feature_order'],'routing_threshold':b['routing_threshold'],'format':'NumPy-only specialist parameters; no sklearn pickle needed at inference'}))}
 out=S/(name+'_portable.npz');np.savez_compressed(out,**arrays)
 x=f.astype('float32').copy();x-=arrays['scaler_mean'];x/=arrays['scaler_scale'];x=x@arrays['pca_components'].T;x-=arrays['pca_mean'][None,:]@arrays['pca_components'].T;norm=np.sqrt(np.sum(x*x,axis=1,keepdims=True));norm[norm==0]=1.;x/=norm;p=expit(x@arrays['coefficient'].T+arrays['intercept']).ravel();reference=saved[name+'_normal_probability'];difference=float(np.max(abs(p-reference)));assert difference<1e-6;np.testing.assert_array_equal(p>=.5,reference>=.5);receipt[name]={'max_probability_difference':difference,'threshold_decisions_identical_all_2112':True}
(S/'portable_parameter_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
