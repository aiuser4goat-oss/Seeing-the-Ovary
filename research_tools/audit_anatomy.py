"""Audit existing anatomy-validation predictions against supplied r1 annotations."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy')
from continuation.morphology import instances_from_predictions,instance_matches,crop_bounds
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22/anatomy_audit';D.mkdir(exist_ok=True)
A=B/'Anatomy/USOVA3D';C=B/'Colab_Transfer/anatomy_postprocessing_validation';z=np.load(C/'validation_predictions.npz',allow_pickle=False);prov=json.loads((C/'provenance.json').read_text())
assert prov['role']=='anatomy validation only';assert hashlib.sha256((B/'Colab_Transfer/recovered/ovarian_continuation/segmentation/best.weights.h5').read_bytes()).hexdigest()==prov['segmenter_checkpoint_sha256']
allrows=list(csv.DictReader((A/'image_mask_pairs.csv').open()));raw={hashlib.sha256(('usova/'+r['image']).encode()).hexdigest()+'.npz':r for r in allrows if r['split']=='val' and int(Path(r['image']).stem.split('_')[-1])%4==0}
assert set(prov['prepared_files'])<=set(raw)
def pad(p):return np.asarray(ImageOps.pad(Image.open(p),(256,256),method=Image.Resampling.NEAREST))
results=[];images=[];truths=[];ovtruth=[]
recipes={'original_watershed':dict(method='watershed',follicle_threshold=.5,min_area=5,min_peak_distance=3),'refined_components':dict(method='connected_components',follicle_threshold=.8,min_area=30,min_peak_distance=3)}
for i,f in enumerate(prov['prepared_files']):
 r=raw[f];gt=pad(A/r['follicle_r1_labels']);ov=pad(A/r['ovary_r1'])>0;ov|=gt>0
 assert np.array_equal(gt,z['truth'][i]),'Cached truth differs from supplied annotation'
 image=np.asarray(ImageOps.pad(Image.open(A/r['image']).convert('L'),(256,256),method=Image.Resampling.BILINEAR));images.append(image);truths.append(gt);ovtruth.append(ov)
 for name,c in recipes.items():
  po,pi=instances_from_predictions(z['ovary'][i].astype('float32'),z['follicle'][i].astype('float32'),z['distance'][i].astype('float32'),**c)
  t,b,l,rr=crop_bounds(po);crop=np.zeros((256,256),bool);crop[t:b,l:rr]=1
  ntruth=int(np.count_nonzero(np.unique(gt)));npred=int(np.count_nonzero(np.unique(pi)));inter=int((po&ov).sum());both=int((po.sum()+ov.sum()))
  result={'file':f,'volume':r['volume_id'],'recipe':name,'ovary_truth_present':bool(ov.any()),'follicle_truth_present':bool((gt>0).any()),'ovary_dice':float(2*inter/both) if both else None,'ovary_crop_coverage':float((crop&ov).sum()/ov.sum()) if ov.any() else None,'follicle_crop_coverage':float((crop&(gt>0)).sum()/(gt>0).sum()) if (gt>0).any() else None,'true_count':ntruth,'predicted_count':npred,'count_absolute_error':abs(ntruth-npred),'ovary_fraction':float(po.mean()),'crop_fraction':float(crop.mean()),**instance_matches(gt,pi)}
  results.append(result)
summary={}
for name in recipes:
 rs=[r for r in results if r['recipe']==name];tp=sum(r['instance_tp'] for r in rs);fp=sum(r['instance_fp'] for r in rs);fn=sum(r['instance_fn'] for r in rs)
 summary[name]={'samples':len(rs),'ovary_positive_samples':sum(r['ovary_truth_present'] for r in rs),'follicle_positive_samples':sum(r['follicle_truth_present'] for r in rs),'count_mae':float(np.mean([r['count_absolute_error'] for r in rs])),'instance_f1_iou50':2*tp/max(2*tp+fp+fn,1),'instance_tp':tp,'instance_fp':fp,'instance_fn':fn}
 for metric in ['ovary_dice','ovary_crop_coverage','follicle_crop_coverage','crop_fraction']:
  values=[r[metric] for r in rs if r[metric] is not None];summary[name][metric+'_mean']=float(np.mean(values));summary[name][metric+'_min']=float(np.min(values))
 summary[name]['ovary_positive_dice_mean']=float(np.mean([r['ovary_dice'] for r in rs if r['ovary_truth_present']]))
(D/'audit.json').write_text(json.dumps({'scope':'Existing anatomy validation, 100 USOVA slices from two volumes; r1 annotations only; postprocessing already selected on this sample. This is a descriptive audit, not independent confirmation or classification-domain anatomical verification. Cached probabilities are float16. No test cases accessed.','source_cache_sha256':hashlib.sha256((C/'validation_predictions.npz').read_bytes()).hexdigest(),'all_cached_truth_reverified_against_original_files':True,'volumes':sorted({r['volume'] for r in results}),'summary':summary},indent=2))
with (D/'per_slice.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
np.savez_compressed(D/'visual_audit_arrays.npz',images=np.array(images),ovary_truth=np.array(ovtruth),follicle_truth=np.array(truths))
print(json.dumps(summary,indent=2))
