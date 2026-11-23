"""Exploratory image-group cross-fitting of a small class-boundary correction."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,f1_score
import csv
ROOT=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');BASE=ROOT/'Accuracy_Development_2026_11_22';OUT=BASE/'boundary_study';OUT.mkdir(exist_ok=True)
manifest=ROOT/'Classification/metadata/dataset_manifest.csv'
with manifest.open(newline='') as f:rows={r['record_id']:r for r in csv.DictReader(f)}
source=BASE/'validation_selection_v1/selected_validation_probabilities.npz'
with np.load(source,allow_pickle=False) as z:ids=z['record_ids'];y=z['true_class_indices'];p=z['raw_probabilities']
assert len(ids)==2112 and len(set(ids))==len(ids)
for i,rid in enumerate(ids):
 r=rows[str(rid)];assert r['training_role']=='val' and int(hashlib.sha256(r['image_group'].encode()).hexdigest()[:8],16)%3!=0
 assert r['class_name']==['Benign','Malignant','Normal','PCOS'][y[i]]
groups=np.array([rows[str(rid)]['image_group'] for rid in ids]);logp=np.log(np.clip(p,1e-8,1));original=p.argmax(1);oof=np.full(len(y),-1);folds=[]
def objective(bias,indices):
 pred=(logp[indices]+np.r_[bias,0.]).argmax(1)
 return -balanced_accuracy_score(y[indices],pred)+.001*float(np.mean(np.r_[bias,0.]**2))
for fold,(fit,heldout) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=2627).split(logp,y,groups)):
 assert not set(groups[fit])&set(groups[heldout]);assert len(np.unique(y[fit]))==len(np.unique(y[heldout]))==4
 opt=differential_evolution(lambda b:objective(b,fit),[(-1.5,1.5)]*3,seed=2627+fold,popsize=10,maxiter=50,polish=False)
 bias=opt.x if objective(opt.x,fit)<objective(np.zeros(3),fit) else np.zeros(3)
 oof[heldout]=(logp[heldout]+np.r_[bias,0.]).argmax(1)
 folds.append({'fold':fold,'fit_records':len(fit),'heldout_records':len(heldout),'heldout_record_ids':ids[heldout].tolist(),'bias':np.r_[bias,0.].tolist(),'original_balanced_accuracy':balanced_accuracy_score(y[heldout],original[heldout]),'adjusted_balanced_accuracy':balanced_accuracy_score(y[heldout],oof[heldout])})
assert np.all(oof>=0)
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'Three-fold StratifiedGroupKFold; optimize three bounded additive log-probability offsets on two folds and evaluate on the held-out fold. Identity is retained if fit objective does not improve.','original_balanced_accuracy':balanced_accuracy_score(y,original),'out_of_fold_adjusted_balanced_accuracy':balanced_accuracy_score(y,oof),'original_macro_f1':f1_score(y,original,average='macro'),'adjusted_macro_f1':f1_score(y,oof,average='macro'),'changed_predictions':int((oof!=original).sum()),'folds':folds,'calibration_or_test_used':False,'model_modified':False,'limitations':'Cross-fitting guards bias-fitting overfit only. Base checkpoints and the ensemble were already selected using this validation set; this remains exploratory, not an untouched test estimate. Free offsets are an adaptation motivated by logit-adjustment research, not an exact reproduction of frequency-based logit adjustment.','sources':['https://arxiv.org/abs/2007.07314','https://arxiv.org/abs/1910.09217']}
(OUT/'crossfit_bias_results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'crossfit_predictions.npz',record_ids=ids,true_class_indices=y,original_predictions=original,out_of_fold_predictions=oof)
print(json.dumps({k:v for k,v in result.items() if k!='folds'},indent=2))
