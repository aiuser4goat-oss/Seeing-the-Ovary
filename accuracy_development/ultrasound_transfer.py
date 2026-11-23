"""Isolated TinyUSFM transfer experiment; original TensorFlow pipeline is untouched."""
import argparse,csv,hashlib,json,math,os,random,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps,ImageEnhance
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix

CLASSES=['Benign','Malignant','Normal','PCOS']
MANIFEST='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
from ultrasound_manifest import rows_for,fitting_weights,parse_args
class Images(Dataset):
 def __init__(self,root,rows,augment=False,weights=None):
  self.root=root;self.rows=rows;self.augment=augment;self.weights=weights if weights is not None else np.ones(len(rows),'float32')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['relative_path']) as raw:im=raw.convert('RGB')
  # Letterbox both phases preserves shape; this is a declared adaptation of the
  # official implementation's square resize, not a bitwise reproduction.
  im=ImageOps.pad(im,(224,224),method=Image.Resampling.BILINEAR,color=0)
  x=np.asarray(im,dtype='float32')/255
  if self.augment:
   if random.random()<.5:x=x[:,::-1,:].copy()
   mask=np.any(x>0,axis=-1,keepdims=True);mean=(x*mask).sum(axis=(0,1),keepdims=True)/max(float(mask.sum()),1.)
   x=np.clip((x-mean)*random.uniform(.9,1.1)+mean,0,1)
   x=np.power(x,random.uniform(.9,1.1))*random.uniform(.85,1.15)
   x*=1+np.random.normal(0,random.uniform(0,.015),x.shape[:2]+(1,)).astype('float32');x=np.where(mask,np.clip(x,0,1),0)
  x=(x-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
  return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1),dtype=np.float32)),CLASSES.index(r['class_name']),float(self.weights[i])
def load_pretrained(model,checkpoint):
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('Expected tensor-only checkpoint')
 fixed={}
 for k,v in state.items():
  k=('model.'+k[13:]) if k.startswith('module.model.') else ('model.'+k[7:]) if k.startswith('module.') else k
  # Released weights use norm; the official GAP variant names it fc_norm.
  if k in ['model.norm.weight','model.norm.bias']:k=k.replace('model.norm.','model.fc_norm.')
  if not (k.endswith('head.weight') or k.endswith('head.bias')):fixed[k]=v
 info=model.load_state_dict(fixed,strict=False)
 if set(info.missing_keys)!={'model.head.weight','model.head.bias'} or info.unexpected_keys:raise ValueError('Backbone not fully restored: '+str(info))
 return {'missing_classifier_only':info.missing_keys,'unexpected_keys':info.unexpected_keys,'checkpoint_sha256':sha(checkpoint),'normalization_transfer':'model.norm.{weight,bias} -> model.fc_norm.{weight,bias}; same pretrained tensors, official global-average pooling variant'}
def metrics(y,p):
 return {'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'confusion_matrix':confusion_matrix(y,p,labels=range(4)).tolist()}
@torch.no_grad()
def evaluate(model,loader,device):
 model.eval();ys=[];probs=[]
 for x,y,w in loader:
  p=model(x.to(device)).float().softmax(-1).cpu().numpy();ys.append(y.numpy());probs.append(p)
 y=np.concatenate(ys);prob=np.concatenate(probs)
 if not np.isfinite(prob).all():raise ValueError('Nonfinite validation predictions')
 return metrics(y,prob.argmax(1)),y,prob

def main():
 a=parse_args()
 a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();random.seed(2627);np.random.seed(2627);torch.manual_seed(2627)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 assert sha(a.classification/'metadata/dataset_manifest.csv')==MANIFEST
 fit=rows_for(a.classification,'fit');val=rows_for(a.classification,'select');assert len(fit)==12763 and len(val)==2112
 assert not {r['image_group'] for r in fit}&{r['image_group'] for r in val}
 assert not {r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in val}
 if a.smoke:
  fit=sum(([r for r in fit if r['class_name']==c][:2] for c in CLASSES),[]);val=sum(([r for r in val if r['class_name']==c][:2] for c in CLASSES),[])
 import importlib.util
 spec=importlib.util.spec_from_file_location('pinned_tinyusfm',a.model_source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 model=module.TinyUSFM(num_classes=4,global_pool=True);receipt=load_pretrained(model,a.checkpoint);model.to(device)
 train=DataLoader(Images(a.classification,fit,True,fitting_weights(fit)),batch_size=8 if a.smoke else 32,shuffle=True,num_workers=0,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(2627))
 validation=DataLoader(Images(a.classification,val),batch_size=16 if a.smoke else 64,num_workers=0,pin_memory=device.type=='cuda')
 declaration={'created_before_training':True,'architecture':'Official TinyUSFM; 192-dimensional, 12 blocks, 12 attention heads','model_source_sha256':sha(a.model_source),'worker_sha256':sha(__file__),'manifest_sha256':MANIFEST,'pretrained':receipt,'roles':['fit','select'],'fit_records':len(fit),'selection_records':len(val),'class_order':CLASSES,'device':str(device),'external_pretraining_overlap':'Unknown. Teacher/pretraining images are not exhaustively available for an image-level overlap audit. This is an exploratory pretrained-model comparison; no uncontaminated-external-pretraining claim.','supervision':'All supplied fitting labels/records retained. Original duplicate/class weighting applied once. No oversampling.','augmentation':'Fit only: horizontal reflection, moderate gain/contrast/gamma and multiplicative speckle. No vertical reflection or artificial test copies.','preprocessing':'RGB 224px bilinear letterbox and official ImageNet mean/std normalization','stages':[{'name':'head','epochs':4,'lr':.0003},{'name':'last6','epochs':8,'lr':.00001},{'name':'full','epochs':4,'lr':.000003}],'early_stopping_patience':3,'weight_decay':.0001,'label_smoothing':.03,'selection_metric':'balanced_accuracy; macro F1 tie-break','time_budget_minutes':a.max_minutes,'smoke':a.smoke,'test_access':False,'precision':'float32; no AMP','scope':'Exploratory after historical test review. Different architecture/pretraining than the old CNN; gains cannot be attributed to pretraining alone.'}
 save(a.out/'declaration.json',declaration)
 initial,y,prob=evaluate(model,validation,device);best=(initial['balanced_accuracy'],initial['macro_f1']);torch.save(model.state_dict(),a.out/'best.weights.pt');save(a.out/'selected.json',{'stage':'initial',**initial})
 print('TINY_USFM_INITIAL',json.dumps(initial),flush=True)
 history=[];timed_out=False
 for stage in declaration['stages']:
  for param in model.parameters():param.requires_grad=False
  if stage['name']=='full':
   for param in model.parameters():param.requires_grad=True
  elif stage['name']=='last6':
   for block in model.model.blocks[6:]:
    for param in block.parameters():param.requires_grad=True
   for param in model.model.fc_norm.parameters():param.requires_grad=True
  for param in model.model.head.parameters():param.requires_grad=True
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=stage['lr'],weight_decay=.0001)
  epochs=1 if a.smoke else stage['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=stage['lr']*.1);stale=0
  # Stage starts from the best checkpoint scored by this same eager evaluator.
  model.load_state_dict(torch.load(a.out/'best.weights.pt',map_location=device,weights_only=True))
  for epoch in range(epochs):
   if time.monotonic()-started>a.max_minutes*60:timed_out=True;break
   model.train();loss_sum=0.;seen=0
   # Frozen transformer blocks have zero configured dropout/drop-path.
   for x,y,w in train:
    if time.monotonic()-started>a.max_minutes*60:timed_out=True;break
    x=x.to(device);y=y.to(device);w=w.to(device,dtype=torch.float32);opt.zero_grad(set_to_none=True)
    logits=model(x);loss=(nn.functional.cross_entropy(logits,y,label_smoothing=.03,reduction='none')*w).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite fitting loss')
    loss.backward();nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
   result,_,_=evaluate(model,validation,device);value=(result['balanced_accuracy'],result['macro_f1']);record={'stage':stage['name'],'epoch':epoch+1,'fit_loss':loss_sum/max(seen,1),'fit_records_seen':seen,'elapsed_seconds':time.monotonic()-started,**result};history.append(record)
   if value>best:
    best=value;torch.save(model.state_dict(),a.out/'best.weights.pt');save(a.out/'selected.json',record);stale=0
   else:stale+=1
   save(a.out/'history.json',history);print('TINY_USFM_EPOCH',json.dumps(record),flush=True);scheduler.step()
   if timed_out or stale>=3:break
  if timed_out:break
 model.load_state_dict(torch.load(a.out/'best.weights.pt',map_location=device,weights_only=True));result,y,prob=evaluate(model,validation,device)
 np.savez_compressed(a.out/'validation_predictions.npz',record_ids=np.array([r['record_id'] for r in val]),true_class_indices=y,probabilities=prob)
 save(a.out/'status.json',{'stage':'complete','time_budget_reached':timed_out,'selected_validation':result,'test_evaluated':False,'weights_sha256':sha(a.out/'best.weights.pt'),'elapsed_seconds':time.monotonic()-started})
 print('TINY_USFM_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
