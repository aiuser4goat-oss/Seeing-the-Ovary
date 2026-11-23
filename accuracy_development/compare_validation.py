"""Select among declared validation candidates using their cached probabilities."""
import argparse,json
from pathlib import Path
import numpy as np
from continuation.common import CLASSES,file_hash
from continuation.evaluation import metrics
from .runner import dump,EXPECTED_MANIFEST
ENSEMBLES={
 'new_image_pair':['warm_image','convnext_image'],
 'new_anatomy_pair':['warm_anatomy','convnext_anatomy'],
 'extended_image':['old_baseline','warm_image','convnext_image'],
 'extended_anatomy':['old_baseline','warm_anatomy','convnext_anatomy'],
 'extended_image_with_old_image':['old_baseline','old_image','warm_image','convnext_image'],
 'extended_anatomy_with_old_image':['old_baseline','old_image','warm_anatomy','convnext_anatomy']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True,help='Declared name=run folder');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 arrays={};origins={};first_ids=None;truth=None
 allowed={'old_baseline','old_image','old_fusion','warm_image','warm_anatomy','convnext_image','convnext_anatomy'}
 for item in a.input:
  name,path=item.split('=',1);root=Path(path)
  if name not in allowed or name in arrays:raise ValueError('Unknown or repeated validation candidate')
  receipt=json.loads((root/'selection_receipt.json').read_text());npz=root/'selection_probabilities.npz'
  if receipt['role']!='select' or receipt['manifest_sha256']!=EXPECTED_MANIFEST or file_hash(npz)!=receipt['probabilities_sha256']:raise ValueError('Validation receipt does not match')
  if name.startswith(('warm_','convnext_')):
   status=json.loads((root/'status.json').read_text())
   if status['stage']!='complete' or status['smoke_only']:raise ValueError('Incomplete/smoke candidate cannot enter selection')
   if file_hash(root/'config.json')!=receipt['config_sha256']:raise ValueError('Candidate config changed')
  with np.load(npz,allow_pickle=False) as z:ids=z['record_ids'];y=z['true_class_indices'];prob=z['raw_probabilities'];classes=z['classes']
  if list(classes)!=CLASSES or len(ids)!=2112 or len(set(ids))!=len(ids) or prob.shape!=(2112,4) or not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,atol=1e-5):raise ValueError('Unexpected validation arrays')
  if first_ids is None:first_ids=ids;truth=y
  if not np.array_equal(first_ids,ids) or not np.array_equal(truth,y):raise ValueError('Validation IDs/labels differ')
  arrays[name]=prob;origins[name]={'path':str(root),'receipt_sha256':file_hash(root/'selection_receipt.json'),'probabilities_sha256':file_hash(npz),'checkpoint_sha256':receipt.get('checkpoint_sha256'),'config_sha256':receipt.get('config_sha256'),'inference_signature':receipt.get('inference_signature')}
 candidates={name:{'members':[name],'metrics':metrics(truth,p)} for name,p in arrays.items()}
 for name,members in ENSEMBLES.items():
  if all(n in arrays for n in members):candidates[name]={'members':members,'metrics':metrics(truth,np.mean([arrays[n] for n in members],axis=0))}
 winner=max(candidates,key=lambda n:(candidates[n]['metrics']['balanced_accuracy'],candidates[n]['metrics']['macro_f1'],-len(candidates[n]['members']),n))
 a.out.mkdir(parents=True,exist_ok=True);p=np.mean([arrays[n] for n in candidates[winner]['members']],axis=0)
 np.savez_compressed(a.out/'selected_validation_probabilities.npz',record_ids=first_ids,classes=np.array(CLASSES),true_class_indices=truth,raw_probabilities=p)
 dump(a.out/'validation_selection.json',{'selection_metric':'balanced_accuracy','tie_breakers':['macro_f1','fewer_members','name'],'winner':winner,'selected_recipe':candidates[winner],
  'candidates':candidates,'origins':origins,'selected_probabilities_sha256':file_hash(a.out/'selected_validation_probabilities.npz'),
  'target_met_on_validation':candidates[winner]['metrics']['balanced_accuracy']>.93,'calibration_and_test_used':False,
  'scope':'Exploratory validation selection after historical test review. A validation score is not an independently confirmed test accuracy.'})
 print('VALIDATION_SELECTED',winner,candidates[winner]['metrics']['balanced_accuracy'],flush=True)
if __name__=='__main__':main()
