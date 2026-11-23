from pathlib import Path
import sys,json
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy')
from accuracy_development.anatomy_safety import crop_decision
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22/anatomy_audit';z=np.load(B/'Colab_Transfer/anatomy_postprocessing_validation/validation_predictions.npz',allow_pickle=False);truth=np.load(D/'visual_audit_arrays.npz',allow_pickle=False)
policies={'original':dict(margin=.1,min_crop_fraction=0,min_mask_fraction=0),'expanded':dict(margin=.25,min_crop_fraction=0,min_mask_fraction=0),'fallback_only':dict(margin=.1,min_crop_fraction=.15,min_mask_fraction=.01),'expanded_and_fallback':dict(margin=.25,min_crop_fraction=.15,min_mask_fraction=.01)}
reports={}
for name,config in policies.items():
 rs=[]
 for i,prob in enumerate(z['ovary']):
  labels,n=ndi.label(prob>.5)
  if n:
   size=np.bincount(labels.ravel());size[0]=0;mask=ndi.binary_fill_holes(labels==np.argmax(size))
  else:mask=np.zeros(prob.shape,bool)
  decision=crop_decision(mask,**config);y0,y1,x0,x1=decision['bounds'];crop=np.zeros(mask.shape,bool);crop[y0:y1,x0:x1]=1;ov=truth['ovary_truth'][i];fg=truth['follicle_truth'][i]>0
  rs.append({'fallback':decision['fallback'],'crop_fraction':float(crop.mean()),'ovary_coverage':float((crop&ov).sum()/ov.sum()) if ov.any() else None,'follicle_coverage':float((crop&fg).sum()/fg.sum()) if fg.any() else None})
 summary={'fallback_count':sum(r['fallback'] for r in rs),'mean_crop_fraction':float(np.mean([r['crop_fraction'] for r in rs]))}
 for field in ['ovary_coverage','follicle_coverage']:
  values=[r[field] for r in rs if r[field] is not None];summary[field]={'mean':float(np.mean(values)),'minimum':float(np.min(values)),'p10':float(np.quantile(values,.1)),'below_90pct':sum(v<.9 for v in values),'positive_samples':len(values)}
 reports[name]={'configuration':config,'summary':summary}
(D/'crop_guard_comparison.json').write_text(json.dumps({'scope':'Exploratory descriptive comparison on anatomy validation already used for postprocessing. Larger/full-image crops trivially improve coverage while losing localization. This is not evidence of improved classification or follicle identification. No test cases read.','policies':reports},indent=2));print(json.dumps(reports,indent=2))
