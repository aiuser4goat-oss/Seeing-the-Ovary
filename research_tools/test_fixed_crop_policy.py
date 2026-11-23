"""One locked-policy evaluation on the separate annotated anatomy test split."""
from pathlib import Path
import sys,json,csv,hashlib,datetime
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy')
from continuation.common import runtime,evaluation_role
from continuation.anatomy import build_segmenter
from continuation.morphology import instances_from_predictions,instance_matches
from accuracy_development.anatomy_safety import crop_decision
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');A=B/'Anatomy/USOVA3D';S=B/'Colab_Transfer/recovered/ovarian_continuation/segmentation';O=B/'Accuracy_Development_2026_11_22/anatomy_fixed_test';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (O/'metrics.json').exists():raise ValueError('Locked anatomy test already scored; do not rerun for tuning')
policies={'original':dict(margin=.1,min_crop_fraction=0,min_mask_fraction=0),'fixed_guard':dict(margin=.25,min_crop_fraction=.15,min_mask_fraction=.01)}
freeze={'locked_before_test_inference':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':sha(S/'best.weights.h5'),'crop_source_sha256':sha('/Users/ik-mac/ovarian_ai_project copy/accuracy_development/anatomy_safety.py'),'worker_sha256':sha(__file__),'policies':policies,'selection':'Fixed from prior anatomy validation; no test-based threshold selection permitted','scope':'USOVA supplied test split, r1 annotations, slice_step4, correlated slices from two volumes. No classification test/calibration inference. Not classification accuracy or patient-independent evaluation.'}
assert freeze['checkpoint_sha256']=='cc4286142cf2edf27a6d0d962d8c9e2772cf4f8015188c276869b84bc7980040';(O/'locked_policy.json').write_text(json.dumps(freeze,indent=2))
with (B/'Classification/metadata/dataset_manifest.csv').open() as f:protected={r['pixel_sha256'] for r in csv.DictReader(f) if evaluation_role(r) in ['calibrate','test']}
with (A/'image_mask_pairs.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['split']=='test' and int(Path(r['image']).stem.split('_')[-1])%4==0]
c=json.loads((S/'config.json').read_text());runtime(c);model=build_segmenter(c);model.load_weights(S/'best.weights.h5');results=[];skipped=[]
def pad(p):return np.asarray(ImageOps.pad(Image.open(p),(256,256),method=Image.Resampling.NEAREST))
for start in range(0,len(rows),4):
 batch=[];images=[]
 for r in rows[start:start+4]:
  with Image.open(A/r['image']) as f:raw=f.convert('L');array=np.asarray(raw)
  rgb=np.repeat(array[...,None],3,-1);key=hashlib.sha256(f'{raw.width}x{raw.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
  if key in protected:skipped.append(r['image']);continue
  images.append(np.asarray(ImageOps.pad(raw,(256,256),method=Image.Resampling.BILINEAR),dtype='float32')[...,None]/255);batch.append(r)
 if not batch:continue
 pred=model(np.array(images),training=False)
 for i,r in enumerate(batch):
  ov,ins=instances_from_predictions(pred['ovary'][i,...,0].numpy(),pred['follicle'][i,...,0].numpy(),pred['distance'][i,...,0].numpy(),follicle_threshold=.8,min_area=30,method='connected_components');gt=pad(A/r['follicle_r1_labels']);truth=pad(A/r['ovary_r1'])>0;truth|=gt>0
  for name,policy in policies.items():
   d=crop_decision(ov,**policy);y0,y1,x0,x1=d['bounds'];crop=np.zeros(ov.shape,bool);crop[y0:y1,x0:x1]=True
   results.append({'image':r['image'],'volume':r['volume_id'],'policy':name,'fallback':d['fallback'],'ovary_crop_coverage':float((crop&truth).sum()/truth.sum()) if truth.any() else None,'follicle_crop_coverage':float((crop&(gt>0)).sum()/(gt>0).sum()) if (gt>0).any() else None,'crop_fraction':float(crop.mean()),'predicted_count_diagnostic_only':int(np.count_nonzero(np.unique(ins))),'true_count':int(np.count_nonzero(np.unique(gt))),'count_absolute_error_diagnostic_only':int(abs(np.count_nonzero(np.unique(ins))-np.count_nonzero(np.unique(gt)))),'numeric_morphology_available':not d['fallback'],**instance_matches(gt,ins)})
 print('ANATOMY_TEST_PROGRESS',min(start+4,len(rows)),'/',len(rows),flush=True)
summary={}
for name in policies:
 rs=[r for r in results if r['policy']==name];m={'slices':len(rs),'volumes':sorted({r['volume'] for r in rs}),'fallbacks':sum(r['fallback'] for r in rs),'mean_crop_fraction':float(np.mean([r['crop_fraction'] for r in rs])),'diagnostic_count_mae_all_slices':float(np.mean([r['count_absolute_error_diagnostic_only'] for r in rs]))}
 tp=sum(r['instance_tp'] for r in rs);fp=sum(r['instance_fp'] for r in rs);fn=sum(r['instance_fn'] for r in rs);m['diagnostic_instance_f1_iou50_all_slices']=2*tp/max(2*tp+fp+fn,1)
 for field in ['ovary_crop_coverage','follicle_crop_coverage']:
  values=[r[field] for r in rs if r[field] is not None];m[field]={'mean':float(np.mean(values)),'minimum':float(np.min(values)),'positive_slices':len(values),'below_90pct':sum(v<.9 for v in values)}
 summary[name]=m
(O/'metrics.json').write_text(json.dumps({'locked_policy_sha256':sha(O/'locked_policy.json'),'summary':summary,'classification_protected_overlap_skipped':skipped,'interpretation':'Crop coverage is not segmentation/classification accuracy. Larger crops trade localization for retention. Count metrics summarize the identical diagnostic predictions under both policies, including fallback cases; fallback measurements are unavailable in the guarded pipeline, not zero. Slices are correlated; no statistical independence or clinical generalization claim.'},indent=2))
with (O/'per_slice.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
print(json.dumps(summary,indent=2))
