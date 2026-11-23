"""Evaluate only two predeclared equal-weight recipes on selection validation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument('--new-run',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True)
 plan=json.loads(a.plan.read_text());declaration=json.loads((a.new_run/'declaration.json').read_text());status=json.loads((a.new_run/'status.json').read_text());assert not declaration['smoke'] and not declaration['test_access'] and not status['test_evaluated']
 assert declaration['manifest_sha256']=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
 assert hashlib.sha256((a.new_run/'best.weights.pt').read_bytes()).hexdigest()==status['weights_sha256']
 z=np.load(a.new_run/'validation_predictions.npz',allow_pickle=False);ids=z['record_ids'];y=z['true_class_indices'];probs={'tinyusfm':z['probabilities']};assert len(ids)==2112
 for name,info in plan['reference_sources'].items():
  p=Path(info['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'];old=np.load(p,allow_pickle=False)
  np.testing.assert_array_equal(ids,old['record_ids']);np.testing.assert_array_equal(y,old['true_class_indices']);assert list(old['classes'])==declaration['class_order'];probs[name]=old['raw_probabilities']
 for v in probs.values():
  assert v.shape==(2112,4) and np.isfinite(v).all() and np.all(v>=0);np.testing.assert_allclose(v.sum(axis=1),1,atol=1e-5)
 report={'scope':'Selection only, two recipes fixed before new full fitting results; no test evaluation','plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'new_checkpoint_sha256':status['weights_sha256'],'results':{}}
 recipes={'tinyusfm_alone':['tinyusfm'],'old_image_alone':['image_only'],**plan['candidates']}
 for name,members in recipes.items():
  probability=np.mean([probs[n] for n in members],axis=0);pred=probability.argmax(1)
  report['results'][name]={'members':members,'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}
  np.savez_compressed(a.out/(name+'.npz'),record_ids=ids,true_class_indices=y,raw_probabilities=probability)
 (a.out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
