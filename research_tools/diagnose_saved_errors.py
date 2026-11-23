from pathlib import Path
import csv,json,collections
import numpy as np
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'saved_error_diagnosis';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((B/'Classification/metadata/dataset_manifest.csv').open()));byid={r['record_id']:r for r in rows};classes=['Benign','Malignant','Normal','PCOS']
def report(records):
 cm=np.zeros((4,4),int);sources={}
 for r,truth,pred in records:
  i=classes.index(truth);j=classes.index(pred);cm[i,j]+=1;sources.setdefault(r['source'],np.zeros((4,4),int))[i,j]+=1
 def summary(m):
  supports=m.sum(1);return {'records':int(m.sum()),'confusion_matrix':m.tolist(),'recall':{c:(float(m[i,i]/supports[i]) if supports[i] else None) for i,c in enumerate(classes)},'support':dict(zip(classes,map(int,supports))),'errors':int(m.sum()-np.trace(m))}
 return {'overall':summary(cm),'by_source':{s:summary(m) for s,m in sources.items()}}
test=list(csv.DictReader((B/'Preserved_Best_Completed_Version_2026_11_22/results/ensemble_control/evaluation/test_predictions.csv').open()));out={'scope':'Descriptive cached predictions only. Historical TEST not used to select new settings. Collection IDs are not verified scanners.','test_preserved_ensemble':report([(byid[x['record_id']],x['true_class'],x['predicted_class']) for x in test])}
z=np.load(D/'ultrasound_transfer_v3_results/Ultrasound_Transfer/training/validation_predictions.npz');old=np.load(D/'results_v1/warm_b0/Ovarian Continuation Results/accuracy_development/references/image_only/selection_probabilities.npz');np.testing.assert_array_equal(z['record_ids'],old['record_ids']);np.testing.assert_array_equal(z['true_class_indices'],old['true_class_indices']);y=z['true_class_indices'];a=old['raw_probabilities'].argmax(1);b=z['probabilities'].argmax(1)
for name,p in [('old_image',a),('tiny_pilot',b)]:out[name+'_selection']=report([(byid[str(id)],classes[t],classes[v]) for id,t,v in zip(z['record_ids'],y,p)])
shared=np.flatnonzero((a!=y)&(b!=y));selected=[byid[str(z['record_ids'][i])] for i in shared]
labels=collections.defaultdict(set);group_labels=collections.defaultdict(set)
for r in rows:
 if r['training_role']=='fit' or (r['training_role']=='val' and r['record_id'] in set(map(str,z['record_ids']))):labels[r['pixel_sha256']].add(r['class_name']);group_labels[r['image_group']].add(r['class_name'])
out['shared_error_pool']={'records':len(selected),'unique_pixel_hashes':len({r['pixel_sha256'] for r in selected}),'detected_similarity_groups':len({r['image_group'] for r in selected}),'records_with_multiple_supplied_classes_in_exact_pixel_group':sum(len(labels[r['pixel_sha256']])>1 for r in selected),'records_with_multiple_supplied_classes_in_detected_similarity_group':sum(len(group_labels[r['image_group']])>1 for r in selected),'meaning':'Wrong in BOTH old image-only and TinyUSFM pilot; not established always-wrong across all models. Exact/similarity metadata associations do not adjudicate true diagnoses. No labels changed; near-duplicate detection not exhaustive.'}
with (O/'shared_selection_review.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['record_id','relative_path','class_name','image_group','pixel_sha256','exact_group_classes','similarity_group_classes']);w.writeheader()
 for r in selected:w.writerow({**{k:r[k] for k in w.fieldnames[:5]},'exact_group_classes':'|'.join(sorted(labels[r['pixel_sha256']])),'similarity_group_classes':'|'.join(sorted(group_labels[r['image_group']]))})
(O/'diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps({'shared_error_pool':out['shared_error_pool'],'test_by_source':out['test_preserved_ensemble']['by_source']},indent=2))
# All unique shared-error images are arranged into local-only contact sheets for visual review.
from PIL import Image,ImageOps,ImageDraw
unique=list({r['pixel_sha256']:r for r in selected}.values())
for start in range(0,len(unique),24):
 sheet=Image.new('RGB',(1000,6*190),'white');draw=ImageDraw.Draw(sheet)
 for index,r in enumerate(unique[start:start+24]):
  x=(index%4)*250;y0=(index//4)*190
  with Image.open(B/'Classification'/r['relative_path']) as image:im=ImageOps.pad(image.convert('RGB'),(245,160))
  sheet.paste(im,(x,y0));draw.text((x+2,y0+162),r['record_id'][:10]+' '+r['class_name'],fill='black')
 sheet.save(O/f'shared_review_{start//24+1:02}.jpg')
print('Contact sheets created; visual review is not claimed complete.')
