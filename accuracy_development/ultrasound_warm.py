"""Longer supervised adaptation from the locked TinyUSFM pilot; FIT/SELECT only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json,time,random,math,importlib.util
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ultrasound_manifest import rows_for,fitting_weights
from ultrasound_warm_args import parse_args
from ultrasound_transfer import Images,evaluate,save,sha,CLASSES,MANIFEST

def atomic_weights(model,path):
 temp=path.with_name(path.stem+'.tmp.pt');torch.save(model.state_dict(),temp);os.replace(temp,path)
def main():
 a=parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'status.json').exists() or (a.out/'best.weights.pt').exists():raise ValueError('Choose a fresh warm run directory')
 start=time.monotonic();seed=2628;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');freeze=json.loads(a.freeze.read_text());old=json.loads((a.warm_run/'declaration.json').read_text());old_status=json.loads((a.warm_run/'status.json').read_text())
 assert sha(a.classification/'metadata/dataset_manifest.csv')==MANIFEST==freeze['manifest_sha256']
 assert sha(a.warm_run/'best.weights.pt')==old_status['weights_sha256']==freeze['weights_sha256']
 assert sha(a.warm_run/'validation_predictions.npz')==freeze['validation_predictions_sha256']
 assert old['worker_sha256']==freeze['pilot_worker_sha256'] and old['class_order']==CLASSES and not old['smoke'] and not old['test_access']
 assert sha(a.model_source)==old['model_source_sha256']==freeze['official_model_source_sha256']
 assert a.epochs>3 and a.max_minutes>0
 fit=rows_for(a.classification,'fit');val=rows_for(a.classification,'select');assert len(fit)==12763 and len(val)==2112
 assert not {r['image_group'] for r in fit}&{r['image_group'] for r in val}
 assert not {r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in val}
 if a.smoke:
  fit=sum(([r for r in fit if r['class_name']==c][:2] for c in CLASSES),[]);val=sum(([r for r in val if r['class_name']==c][:2] for c in CLASSES),[])
 spec=importlib.util.spec_from_file_location('pinned_tinyusfm_warm',a.model_source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 model=module.TinyUSFM(num_classes=4,global_pool=True);model.load_state_dict(torch.load(a.warm_run/'best.weights.pt',map_location='cpu',weights_only=True),strict=True);model.to(device)
 for p in model.parameters():p.requires_grad=True
 train=DataLoader(Images(a.classification,fit,True,fitting_weights(fit)),batch_size=8 if a.smoke else 32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed),pin_memory=device.type=='cuda')
 validation=DataLoader(Images(a.classification,val),batch_size=16 if a.smoke else 64,num_workers=0,pin_memory=device.type=='cuda')
 declaration={'created_before_fitting':True,'architecture':old['architecture'],'class_order':CLASSES,'manifest_sha256':MANIFEST,'source_pilot_freeze':freeze,'worker_sha256':sha(__file__),'model_source_sha256':sha(a.model_source),'fit_records':len(fit),'selection_records':len(val),'roles':['fit','select'],'test_access':False,'smoke':a.smoke,'device':str(device),'seed':seed,'torch_version':torch.__version__,'numpy_version':np.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'precision':'float32; deterministic operations','external_pretraining_overlap':old['external_pretraining_overlap'],'recipe':{'full_encoder_epochs':a.epochs,'peak_lr':3e-5,'minimum_lr':1e-6,'linear_warmup_epochs':3,'cosine_after_warmup':True,'weight_decay':1e-4,'label_smoothing':.03,'gradient_clip':1.,'early_stop_patience':6,'max_minutes':a.max_minutes},'data':'Identical original records/labels/split; cached archive staged locally. Same existing weighting and appearance augmentation; new seed2628 avoids replaying the pilot stream.','selection':'Balanced accuracy; macro F1 tie-break. Keep initial pilot checkpoint if adaptation does not improve it.','scope':'Exploratory further fitting after pilot validation review. No test-based recipe selection. New seed, learning-rate schedule and duration form a combined adaptation recipe, not an isolated causal test.'};save(a.out/'declaration.json',declaration)
 initial,y,prob=evaluate(model,validation,device)
 if not a.smoke:
  original=np.load(a.warm_run/'validation_predictions.npz',allow_pickle=False);np.testing.assert_array_equal(original['record_ids'],[r['record_id'] for r in val]);np.testing.assert_array_equal(original['true_class_indices'],y)
  difference=float(np.max(np.abs(original['probabilities']-prob)))
  same_classes=bool(np.array_equal(original['probabilities'].argmax(1),prob.argmax(1)))
  save(a.out/'initial_snapshot_check.json',{'maximum_probability_difference':difference,'identical_top_classes':same_classes,'fixed_tolerance':1e-5,'device':str(device)})
  if device.type!='cuda' or difference>1e-5 or not same_classes:raise ValueError('Full warm run requires GPU and the locked pilot prediction guard')
 best=(initial['balanced_accuracy'],initial['macro_f1']);atomic_weights(model,a.out/'best.weights.pt');save(a.out/'selected.json',{'epoch':0,**initial});print('WARM_INITIAL',json.dumps(initial),flush=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);history=[];stale=0;reason='epoch_limit';epochs=1 if a.smoke else a.epochs
 for epoch in range(epochs):
  if time.monotonic()-start>a.max_minutes*60:reason='time_budget';break
  lr=3e-5*(epoch+1)/3 if epoch<3 else 1e-6+(3e-5-1e-6)*(.5+.5*math.cos(math.pi*(epoch-3)/max(a.epochs-4,1)))
  for group in optimizer.param_groups:group['lr']=lr
  model.train();loss_total=0.;seen=0
  for x,y,w in train:
   if time.monotonic()-start>a.max_minutes*60:reason='time_budget';break
   x=x.to(device);y=y.to(device);w=w.to(device,dtype=torch.float32);optimizer.zero_grad(set_to_none=True)
   loss=(nn.functional.cross_entropy(model(x),y,reduction='none',label_smoothing=.03)*w).mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite fitting loss')
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_total+=float(loss.detach())*len(y);seen+=len(y)
  result,_,_=evaluate(model,validation,device);value=(result['balanced_accuracy'],result['macro_f1']);record={'epoch':epoch+1,'fit_loss':loss_total/max(seen,1),'fit_records_seen':seen,'learning_rate':lr,'elapsed_seconds':time.monotonic()-start,**result};history.append(record)
  if value>best:best=value;atomic_weights(model,a.out/'best.weights.pt');save(a.out/'selected.json',record);stale=0
  else:stale+=1
  save(a.out/'history.json',history);print('WARM_EPOCH',json.dumps(record),flush=True)
  if reason=='time_budget':break
  if epoch>=2 and stale>=6:reason='early_stopping';break
 model.load_state_dict(torch.load(a.out/'best.weights.pt',map_location=device,weights_only=True));result,y,prob=evaluate(model,validation,device);np.savez_compressed(a.out/'validation_predictions.npz',record_ids=np.array([r['record_id'] for r in val]),true_class_indices=y,probabilities=prob,classes=np.array(CLASSES))
 save(a.out/'status.json',{'stage':'complete','termination':reason,'completed_epochs':sum(r['fit_records_seen']==len(fit) for r in history),'epochs_started':len(history),'partial_epoch_logged':any(r['fit_records_seen']!=len(fit) for r in history),'selected_validation':result,'test_evaluated':False,'weights_sha256':sha(a.out/'best.weights.pt'),'elapsed_seconds':time.monotonic()-start});print('WARM_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
