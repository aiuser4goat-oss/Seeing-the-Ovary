"""Bounded validation-only development; never evaluates the old test split."""
import argparse,csv,datetime,gc,json,math,os,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from continuation.common import CLASSES,file_hash,validate_manifest,runtime,role_rows
from continuation.evaluation import metrics
from continuation.models import ConfusionMetric
from continuation.features import FeatureTable,fit_normalizer
from .data import dataset
from .models import build,warm_start,set_trainable,residual_anatomy

EXPECTED_MANIFEST='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
RECIPES={
 'warm_b0':{'backbone':'efficientnetv2b0','image_size':320,'warm_start':True,'stages':[
  {'name':'adapt','size':288,'fraction':.5,'epochs':6,'lr':8e-6,'power':.5,'mixup':.10},
  {'name':'refine','size':320,'fraction':1.,'epochs':8,'lr':3e-6,'power':.5,'mixup':.05},
  {'name':'rebalance','size':320,'fraction':0.,'epochs':4,'lr':3e-5,'power':1.,'mixup':0.}]},
 'convnext':{'backbone':'convnexttiny','image_size':288,'warm_start':False,'stages':[
  {'name':'head','size':224,'fraction':0.,'epochs':3,'lr':3e-4,'power':1.,'mixup':0.},
  {'name':'fine','size':256,'fraction':.5,'epochs':8,'lr':1e-5,'power':.5,'mixup':.10},
  {'name':'full','size':288,'fraction':1.,'epochs':6,'lr':3e-6,'power':.5,'mixup':.10},
  {'name':'rebalance','size':288,'fraction':0.,'epochs':3,'lr':3e-5,'power':1.,'mixup':0.}]}}

def read(p):return json.loads(Path(p).read_text())
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)
def save_status(out,**kw):dump(Path(out)/'status.json',{'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
class TimeBudget(tf.keras.callbacks.Callback):
 def __init__(self,deadline,state=None):super().__init__();self.deadline=deadline;self.state=state;self.last_saved=0.
 def record(self):
  now=time.monotonic()
  if self.state:
   path,started,spent=self.state;dump(path,{'elapsed_seconds':spent+now-started,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  self.last_saved=now
 def on_train_batch_end(self,batch,logs=None):
  if time.monotonic()-self.last_saved>=30:self.record()
  if time.monotonic()>=self.deadline:self.model.stop_training=True
 def on_train_end(self,logs=None):self.record()

def eager_validation(model,ds):
 probabilities=[];labels=[]
 for x,y,_ in ds:
  probabilities.append(model(x,training=False).numpy());labels.append(y.numpy().argmax(1))
 return metrics(np.concatenate(labels),np.concatenate(probabilities))

class EagerValidation(tf.keras.callbacks.Callback):
 def __init__(self,ds):super().__init__();self.ds=ds
 def on_epoch_end(self,epoch,logs=None):
  values=eager_validation(self.model,self.ds)
  if logs is not None:
   for name in ['accuracy','balanced_accuracy','macro_f1']:logs['val_'+name]=values[name]
   logs['val_loss']=values['nll']

def compile_model(model,lr,epochs,steps):
 schedule=tf.keras.optimizers.schedules.CosineDecay(lr,max(steps*epochs,1),alpha=.15)
 optimizer=tf.keras.optimizers.AdamW(schedule,weight_decay=1e-4,clipnorm=1.,use_ema=True,ema_momentum=.99)
 model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.01),
  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),ConfusionMetric(),ConfusionMetric('macro_f1')])

def fit_stage(model,encoder,root,out,c,stage,deadline,smoke=False,table=None,norm=None,budget_state=None):
 if encoder is not None:set_trainable(encoder,stage['fraction'])
 fit,rows=dataset(root,'fit',c,stage['size'],stage['power'],stage['mixup'],table,norm,smoke)
 val,_=dataset(root,'select',c,c['image_size'],table=table,normalizer=norm,smoke=smoke)
 epochs=1 if smoke else stage['epochs'];compile_model(model,stage['lr'],epochs,math.ceil(len(rows)/c['batch_size']))
 target=out/(stage['name']+'.weights.h5')
 prior_score=None
 if target.exists():
  model.load_weights(target)
  prior_score=eager_validation(model,val)['balanced_accuracy']
 callbacks=[tf.keras.callbacks.BackupAndRestore(str(out/('backup_'+stage['name'])),save_freq='epoch',delete_checkpoint=True),
  tf.keras.callbacks.SwapEMAWeights(swap_on_epoch=True),EagerValidation(val),
  tf.keras.callbacks.ModelCheckpoint(str(target),monitor='val_balanced_accuracy',mode='max',save_best_only=True,save_weights_only=True,initial_value_threshold=prior_score),
  tf.keras.callbacks.EarlyStopping(monitor='val_balanced_accuracy',mode='max',patience=4,restore_best_weights=False),
  tf.keras.callbacks.TerminateOnNaN(),tf.keras.callbacks.CSVLogger(str(out/(stage['name']+'_history.csv')),append=True),TimeBudget(deadline,budget_state)]
 # Pinned Keras BackupAndRestore leaves this trainer field after a recovered
 # stage. Clear it; this stage's callback restores its own epoch if needed.
 model._initial_epoch=None
 history=model.fit(fit,epochs=epochs,callbacks=callbacks,verbose=2,shuffle=False)
 if not target.exists():raise RuntimeError('Stage did not save a validation checkpoint')
 model.load_weights(target)
 with (out/(stage['name']+'_history.csv')).open(newline='') as f:history_rows=list(csv.DictReader(f))
 score=eager_validation(model,val)['balanced_accuracy']
 return {'name':stage['name'],'selection_balanced_accuracy':score,'epochs_logged':len(history_rows),'planned_epochs':epochs,'epochs_in_this_invocation':len(history.epoch),'checkpoint_sha256':file_hash(target),'budget_reached':time.monotonic()>=deadline,'epoch_budget_completed':len(history_rows)>=epochs}

def cache_selection(model,root,out,c,table=None,norm=None,smoke=False):
 ds,rows=dataset(root,'select',c,c['image_size'],table=table,normalizer=norm,smoke=smoke)
 p=np.concatenate([model(x,training=False).numpy() for x,_,_ in ds]);y=np.array([CLASSES.index(r['class_name']) for r in rows])
 if not np.isfinite(p).all():raise ValueError('Nonfinite validation probabilities')
 path=out/'selection_probabilities.npz';np.savez_compressed(path,record_ids=np.array([r['record_id'] for r in rows]),classes=np.array(CLASSES),true_class_indices=y,raw_probabilities=p)
 stats=metrics(y,p);dump(out/'selection_metrics.json',stats)
 report=[]
 for i,r in enumerate(rows):report.append({'record_id':r['record_id'],'image_group':r['image_group'],'source':r['source'],'provided_class':r['class_name'],'predicted_class':CLASSES[p[i].argmax()],'confidence':float(p[i].max()),**{n:float(p[i,j]) for j,n in enumerate(CLASSES)}})
 with (out/'selection_predictions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
 dump(out/'selection_receipt.json',{'role':'select','records':len(rows),'manifest_sha256':file_hash(Path(root)/'metadata/dataset_manifest.csv'),
  'checkpoint_sha256':file_hash(out/'best.weights.h5'),'config_sha256':file_hash(out/'config.json'),'probabilities_sha256':file_hash(path),
  'metrics_sha256':file_hash(out/'selection_metrics.json'),'predictions_sha256':file_hash(out/'selection_predictions.csv'),
  'normalizer_sha256':file_hash(out/'normalizer.json') if (out/'normalizer.json').exists() else None,
  'runtime':{'tensorflow':tf.__version__,'keras':tf.keras.__version__,'deterministic_operations':True},
  'scope':'One deterministic validation prediction pass; no test or calibration fitting. All later selection analyses reuse this cache.'})
 return stats

def fit_anatomy(base,root,out,c,features,deadline,smoke=False,budget_state=None):
 out.mkdir(exist_ok=True);table=FeatureTable(features);provenance=read(Path(features)/'provenance.json')
 if provenance['classifier_manifest_sha256']!=EXPECTED_MANIFEST:raise ValueError('Wrong anatomy provenance')
 norm=fit_normalizer(table,role_rows(root,'fit'))
 c={**c,'kind':'residual_anatomy','feature_provenance':provenance,'base_checkpoint_sha256':file_hash(out.parent/'best.weights.h5')}
 if (out/'config.json').exists() and read(out/'config.json')!=c:raise ValueError('Existing anatomy run has different base/config/provenance')
 if (out/'normalizer.json').exists() and read(out/'normalizer.json')!=norm:raise ValueError('Existing anatomy normalizer changed')
 dump(out/'normalizer.json',norm);dump(out/'config.json',c)
 if (out/'status.json').exists() and read(out/'status.json').get('stage')=='complete':
  receipt=read(out/'selection_receipt.json')
  if file_hash(out/'best.weights.h5')!=receipt['checkpoint_sha256'] or file_hash(out/'selection_probabilities.npz')!=receipt['probabilities_sha256']:raise ValueError('Completed anatomy artifacts changed')
  return read(out/'selection_metrics.json')
 model=residual_anatomy(base)
 val,_=dataset(root,'select',c,c['image_size'],table=table,normalizer=norm,smoke=smoke)
 for x,_,_ in val.take(1):
  if not np.allclose(base({'image':x['image']},training=False).numpy(),model(x,training=False).numpy(),atol=2e-5,rtol=2e-5):raise ValueError('Residual branch does not start at the base prediction')
 initial_receipt=out/'initial_state_receipt.json'
 if initial_receipt.exists():
  receipt=read(initial_receipt)
  if file_hash(out/'initial_selection_metrics.json')!=receipt['metrics_sha256'] or file_hash(out/'initial_selection_probabilities.npz')!=receipt['probabilities_sha256']:raise ValueError('Initial anatomy baseline changed')
  model.load_weights(out/'best.weights.h5');initial=read(out/'initial_selection_metrics.json')
 else:
  if (out/'anatomy_correction.weights.h5').exists():raise ValueError('Trained anatomy checkpoint has no initial baseline receipt')
  # No fitting can start before this baseline marker commits. Rebuilding an
  # interrupted initialization therefore cannot discard a trained candidate.
  model.save_weights(out/'best.weights.h5');initial=cache_selection(model,root,out,c,table,norm,smoke)
  dump(out/'initial_selection_metrics.json',initial)
  import shutil
  shutil.copy2(out/'selection_probabilities.npz',out/'initial_selection_probabilities.npz')
  dump(initial_receipt,{'metrics_sha256':file_hash(out/'initial_selection_metrics.json'),'probabilities_sha256':file_hash(out/'initial_selection_probabilities.npz')})
 stage={'name':'anatomy_correction','size':c['image_size'],'fraction':0.,'epochs':4,'lr':1e-4,'power':1.,'mixup':0.}
 result=fit_stage(model,None,root,out,c,stage,deadline,smoke,table,norm,budget_state)
 # Keep a separately documented zero-correction candidate if training is worse.
 if result['selection_balanced_accuracy']>initial['balanced_accuracy']:model.save_weights(out/'best.weights.h5')
 model.load_weights(out/'best.weights.h5');final=cache_selection(model,root,out,c,table,norm,smoke)
 dump(out/'selection.json',{'initial_image_only_balanced_accuracy':initial['balanced_accuracy'],'final_balanced_accuracy':final['balanced_accuracy'],'stage':result,'scope':'Validation-only optional anatomy correction; no assumption that predicted anatomy improves classification.'})
 save_status(out,stage='complete',smoke_only=smoke,budget_reached=result['budget_reached'],epochs_logged=result['epochs_logged'],planned_epochs=result['planned_epochs'])
 return final

def main():
 p=argparse.ArgumentParser()
 for k in ['classification','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--candidate',choices=list(RECIPES),required=True);p.add_argument('--frozen-root',type=Path);p.add_argument('--features',type=Path)
 p.add_argument('--max-minutes',type=float,default=100.);p.add_argument('--smoke',action='store_true');p.add_argument('--skip-anatomy',action='store_true');a=p.parse_args()
 if a.max_minutes<=0:raise ValueError('Time budget must be positive')
 started=time.monotonic();prior_spent=read(a.out/'runtime_budget.json')['elapsed_seconds'] if (a.out/'runtime_budget.json').exists() else 0.
 deadline=started+a.max_minutes*60-prior_spent-90;budget_state=(a.out/'runtime_budget.json',started,prior_spent)
 c={'seed':2627,'batch_size':16,'mixed_precision':True,'data_threads':4,'dropout':.25,'kind':'image','appearance':{'gain':.12,'contrast':.08,'gamma':.08,'speckle':.015},'deterministic_operations':True,**RECIPES[a.candidate]}
 if a.smoke:c={**c,'image_size':64,'batch_size':4,'stages':[{**s,'size':64} for s in c['stages']]}
 tf=runtime(c);tf.config.experimental.enable_op_determinism()
 audit=validate_manifest(a.classification)
 if audit['manifest_sha256']!=EXPECTED_MANIFEST:raise ValueError('Unexpected dataset; split remains fixed')
 for role in ['fit','select']:
  if set(r['class_name'] for r in role_rows(a.classification,role))!=set(CLASSES):raise ValueError('Missing development class')
 out=a.out;out.mkdir(parents=True,exist_ok=True)
 if (out/'config.json').exists() and read(out/'config.json')!=c:raise ValueError('Existing run has different config')
 if (out/'development_protocol.json').exists():
  previous=read(out/'development_protocol.json')
  sources={f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')}
  if previous['model_sources']!=sources or previous['max_minutes']!=a.max_minutes:raise ValueError('Resume source or declared budget changed; use a new experiment directory')
 if (out/'status.json').exists() and read(out/'status.json').get('stage')=='complete':
  receipt=read(out/'selection_receipt.json')
  for name,key in [('best.weights.h5','checkpoint_sha256'),('config.json','config_sha256'),('selection_probabilities.npz','probabilities_sha256'),('selection_metrics.json','metrics_sha256')]:
   if file_hash(out/name)!=receipt[key]:raise ValueError('Completed development artifacts changed')
  print('Development already complete and verified.',flush=True);return
 dump(out/'config.json',c);save_status(out,stage='initializing',candidate=a.candidate,smoke_only=a.smoke)
 progress=read(out/'progress.json') if (out/'progress.json').exists() else {'stages':[],'best_score':-1.}
 resume=(out/'best.weights.h5').exists();model,encoder=build(c,weights=None if resume or a.smoke or c['warm_start'] else 'imagenet')
 if resume:model.load_weights(out/'best.weights.h5')
 elif c['warm_start'] and not a.smoke:
  if a.frozen_root is None:raise ValueError('Frozen root required for warm start')
  run=a.frozen_root/'image_only';freeze=read(a.frozen_root/'experiment_freeze.json')['freeze']
  from continuation.artifacts import inference_signature
  if inference_signature(run)!=freeze['models']['image_only']['signature']:raise ValueError('Frozen warm-start model changed')
  receipt=warm_start(model,encoder,run);dump(out/'warm_start_receipt.json',receipt)
 dump(out/'development_protocol.json',{'manifest_sha256':audit['manifest_sha256'],'allowed_roles':['fit','select'],
  'calibration_and_test_evaluated':False,'prior_test_seen':True,'scope':'Exploratory follow-up after prior test review. New choices are selected on validation only. An untouched confirmation set is needed for a new unbiased accuracy claim.',
  'target_metric':'balanced_accuracy','target':.93,'no_target_guarantee':True,'max_minutes':a.max_minutes,'selection_resolution':c['image_size'],'model_sources':{f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')}})
 try:
  if progress['best_score']<0:
   model.save_weights(out/'best.weights.h5');initial=cache_selection(model,a.classification,out,c,smoke=a.smoke)
   dump(out/'initial_selection_metrics.json',initial)
   import shutil
   shutil.copy2(out/'selection_probabilities.npz',out/'initial_selection_probabilities.npz')
   progress['best_score']=initial['balanced_accuracy'];progress['initial_balanced_accuracy']=initial['balanced_accuracy'];dump(out/'progress.json',progress)
  for s in c['stages']:
   if s['name'] in {r['name'] for r in progress['stages']}:continue
   if time.monotonic()>=deadline:break
   save_status(out,stage='training',candidate=a.candidate,training_stage=s['name'],smoke_only=a.smoke)
   result=fit_stage(model,encoder,a.classification,out,c,s,deadline,a.smoke,budget_state=budget_state);progress['stages'].append(result)
   if result['selection_balanced_accuracy']>progress['best_score']:
    progress['best_score']=result['selection_balanced_accuracy'];model.save_weights(out/'best.weights.h5')
   model.load_weights(out/'best.weights.h5');dump(out/'progress.json',progress)
  if not (out/'best.weights.h5').exists():raise RuntimeError('No completed development checkpoint')
  selected=cache_selection(model,a.classification,out,c,smoke=a.smoke)
  anatomy=None
  if a.features and not a.skip_anatomy and time.monotonic()+360<deadline:
   anatomy=fit_anatomy(model,a.classification,out/'anatomy_correction',c,a.features,deadline,a.smoke,budget_state)
  TimeBudget(deadline,budget_state).record()
  save_status(out,stage='complete',candidate=a.candidate,smoke_only=a.smoke,budget_reached=time.monotonic()>=deadline,full_schedule_completed=len(progress['stages'])==len(c['stages']) and not any(s['budget_reached'] for s in progress['stages']),selection_metrics=selected,anatomy_selection_metrics=anatomy,elapsed_seconds=prior_spent+time.monotonic()-started,calibration_and_test_evaluated=False)
  print('DEVELOPMENT_COMPLETE',a.candidate,'validation_balanced_accuracy',selected['balanced_accuracy'],flush=True)
 except BaseException as e:
  TimeBudget(deadline,budget_state).record();save_status(out,stage='failed',error=repr(e),candidate=a.candidate,smoke_only=a.smoke);raise
if __name__=='__main__':main()
