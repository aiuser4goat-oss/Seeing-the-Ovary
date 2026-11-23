"""Descriptive matched selection errors; no fitting or test-image access."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets')
D=B/'Accuracy_Development_2026_11_22'
manifest=B/'Classification/metadata/dataset_manifest.csv'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
rows={r['record_id']:r for r in csv.DictReader(manifest.open())}
old_path=D/'results_v1/warm_b0/Ovarian Continuation Results/accuracy_development/references/image_only/selection_probabilities.npz'
new_path=D/'ultrasound_transfer_v3_results/Ultrasound_Transfer/training/validation_predictions.npz'
old=np.load(old_path,allow_pickle=False);new=np.load(new_path,allow_pickle=False)
np.testing.assert_array_equal(old['record_ids'],new['record_ids'])
np.testing.assert_array_equal(old['true_class_indices'],new['true_class_indices'])
ids=old['record_ids'];y=old['true_class_indices'];classes=list(old['classes']);a=old['raw_probabilities'].argmax(1);b=new['probabilities'].argmax(1)
records=[rows[str(i)] for i in ids]
for r in records:
 assert r['training_role']=='val' and int(hashlib.sha256(r['image_group'].encode()).hexdigest()[:8],16)%3!=0

def summarize(indices):
 idx=np.asarray(indices,dtype=int);truth=y[idx];aa=a[idx]==truth;bb=b[idx]==truth
 return {'records':len(idx),'unique_image_groups':len({records[i]['image_group'] for i in idx}),'old_correct':int(aa.sum()),'pilot_correct':int(bb.sum()),'both_correct':int((aa&bb).sum()),'old_only_correct':int((aa&~bb).sum()),'pilot_only_correct':int((~aa&bb).sum()),'both_wrong':int((~aa&~bb).sum())}
result={'scope':'Descriptive selection-only diagnostic after model selection. No independent confirmation, no clinical label adjudication, no image or target edits. Source fields identify collection folders, not verified scanner or patient origin.','classes':classes,'old_cache_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'pilot_cache_sha256':hashlib.sha256(new_path.read_bytes()).hexdigest(),'overall':summarize(range(len(y))),'by_class':{c:summarize(np.flatnonzero(y==k)) for k,c in enumerate(classes)},'by_collection':{s:summarize([i for i,r in enumerate(records) if r['source']==s]) for s in sorted({r['source'] for r in records})}}
out=D/'ultrasound_transfer_error_diagnostic';out.mkdir(exist_ok=True)
(out/'selection_error_transitions.json').write_text(json.dumps(result,indent=2))
with (out/'selection_errors.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['record_id','image_group','collection','provided_class','old_prediction','pilot_prediction','old_correct','pilot_correct'])
 writer.writeheader()
 for i,r in enumerate(records):
  if a[i]!=y[i] or b[i]!=y[i]:writer.writerow({'record_id':r['record_id'],'image_group':r['image_group'],'collection':r['source'],'provided_class':classes[y[i]],'old_prediction':classes[a[i]],'pilot_prediction':classes[b[i]],'old_correct':bool(a[i]==y[i]),'pilot_correct':bool(b[i]==y[i])})
print(json.dumps(result,indent=2))
