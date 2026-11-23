from pathlib import Path
import json,hashlib,base64,datetime
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';C=Path('/Users/ik-mac/ovarian_ai_project copy/accuracy_development');K=B/'Kaggle_Launch/ultrasound-warm';K.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=D/'ultrasound_transfer_v3_results/Ultrasound_Transfer/training';status=json.loads((r/'status.json').read_text());old=json.loads((r/'declaration.json').read_text());assert sha(r/'best.weights.pt')==status['weights_sha256']
freeze={'manifest_sha256':old['manifest_sha256'],'weights_sha256':status['weights_sha256'],'validation_predictions_sha256':sha(r/'validation_predictions.npz'),'pilot_worker_sha256':old['worker_sha256'],'official_model_source_sha256':old['model_source_sha256'],'pilot_selection_balanced_accuracy':status['selected_validation']['balanced_accuracy'],'pilot_kernel':'ishaankhera/ovarian-ultrasound-transfer-validation','pilot_version':3,'scope':'Warm start selected by selection data only; no test-based tuning. Full source/checkpoint frozen before the new fitting run.'};(D/'ultrasound_warm_source_freeze.json').write_text(json.dumps(freeze,indent=2))
files={n:(C/n).read_text() for n in ['ultrasound_transfer.py','ultrasound_manifest.py','ultrasound_warm.py','ultrasound_warm_args.py','training_cache.py']};files['tinyusfm.py']=(D/'ultrasound_research/model/tinyusfm.py').read_text();payload={n:{'text':s,'sha256':hashlib.sha256(s.encode()).hexdigest()} for n,s in files.items()};(D/'ultrasound_warm_payload_v1.json').write_text(json.dumps(payload,indent=2))
script='''import os,sys,json,hashlib,subprocess,zipfile,concurrent.futures
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='4'
OUT=Path('/kaggle/working/Ultrasound_Warm_Validation');OUT.mkdir(exist_ok=True)
SOURCE=OUT/'source';SOURCE.mkdir(exist_ok=True)
PAYLOAD=__PAYLOAD__
FREEZE=__FREEZE__
for name,item in PAYLOAD.items():
 p=SOURCE/name;p.write_text(item['text']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
freeze_path=OUT/'pilot_freeze.json';freeze_path.write_text(json.dumps(FREEZE,indent=2))
subprocess.run([sys.executable,'-m','pip','install','--no-deps','timm==0.5.4','--quiet'],check=True)
sys.path.insert(0,str(SOURCE))
from training_cache import prepare,sha
from ultrasound_manifest import rows_for
root,receipt=prepare('/kaggle/input','/tmp/ultrasound_warm_classification')
(OUT/'input_cache_receipt.json').write_text(json.dumps(receipt,indent=2));print('WARM_CACHE_READY',json.dumps(receipt),flush=True)
for role in ['fit','select']:
 rows=rows_for(root,role)
 if os.environ.get('WARM_PROBE_ONLY')=='1':rows=sum(([r for r in rows if r['class_name']==n][:2] for n in ['Benign','Malignant','Normal','PCOS']),[])
 def verify(r):assert sha(root/r['relative_path'])==r['sha256'];return 1
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:count=sum(pool.map(verify,rows))
 print('WARM_FILES_VERIFIED',role,count,flush=True)
candidates=[]
for status in Path('/kaggle/input').rglob('status.json'):
 if status.parent.name!='training':continue
 run=status.parent
 if not (run/'best.weights.pt').is_file() or not (run/'declaration.json').is_file():continue
 s=json.loads(status.read_text());d=json.loads((run/'declaration.json').read_text())
 if s.get('weights_sha256')==FREEZE['weights_sha256'] and d.get('worker_sha256')==FREEZE['pilot_worker_sha256']:
  assert sha(run/'best.weights.pt')==FREEZE['weights_sha256'] and sha(run/'validation_predictions.npz')==FREEZE['validation_predictions_sha256'];candidates.append(run)
assert candidates,'Locked completed pilot output must be attached'
warm=sorted(candidates,key=lambda p:(len(p.parts),str(p)))[0];print('LOCKED_PILOT_RESTORED',str(warm),flush=True)
base=[sys.executable,'-u',str(SOURCE/'ultrasound_warm.py'),'--classification',str(root),'--model-source',str(SOURCE/'tinyusfm.py'),'--warm-run',str(warm),'--freeze',str(freeze_path)]
subprocess.run(base+['--out',str(OUT/'smoke'),'--smoke','--epochs','40','--max-minutes','10'],check=True)
print('WARM_SMOKE_PASSED',flush=True)
if os.environ.get('WARM_PROBE_ONLY')!='1':subprocess.run(base+['--out',str(OUT/'training'),'--epochs','40','--max-minutes','90'],check=True)
with zipfile.ZipFile(OUT/'Warm_Validation_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.npz','.py','.pt'] and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(OUT)))
print('WARM_JOB_COMPLETE',flush=True)
'''.replace('__PAYLOAD__',repr(payload)).replace('__FREEZE__',repr(freeze))
(K/'launch.py').write_text(script)
cell="import base64,subprocess,sys\nfrom pathlib import Path\np=Path('/kaggle/working/warm_launch.py')\np.write_bytes(base64.b64decode("+repr(base64.b64encode(script.encode()).decode())+"))\nsubprocess.run([sys.executable,'-u',str(p)],check=True)\n"
nb={'cells':[{'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':cell.splitlines(True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5};(K/'warm.ipynb').write_text(json.dumps(nb))
oldmeta=json.loads((B/'Kaggle_Launch/ultrasound-transfer/verified_v3/kernel-metadata.json').read_text());metadata={'id':'ishaankhera/ovarian-ultrasound-warm-validation','title':'Ovarian Ultrasound Warm Validation','code_file':'warm.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'docker_image':oldmeta['docker_image'],'dataset_sources':['ishaankhera/ovarian-training-cache-private'],'kernel_sources':['ishaankhera/ovarian-ultrasound-transfer-validation'],'competition_sources':[]};(K/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
P=B/'Kaggle_Launch/ultrasound-warm-probe';P.mkdir(exist_ok=True);nb['cells'][0]['source']=["import os\nos.environ['WARM_PROBE_ONLY']='1'\n"]+nb['cells'][0]['source'];(P/'probe.ipynb').write_text(json.dumps(nb));metadata.update(id='ishaankhera/ovarian-ultrasound-warm-preflight',title='Ovarian Ultrasound Warm Preflight',code_file='probe.ipynb',enable_gpu=False);(P/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
plan=json.loads((D/'ultrasound_transfer_ensemble_plan.json').read_text());plan.pop('before_new_full_training_results',None);plan['declared_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();plan['before_warm_fitting']=True;plan['pilot_validation_reviewed']=True;plan['new_member']='Selected checkpoint from locked warm continuation of TinyUSFM pilot; 40 epoch maximum, 90minute cap';plan['source_freeze_sha256']=sha(D/'ultrasound_warm_source_freeze.json');(D/'ultrasound_warm_ensemble_plan.json').write_text(json.dumps(plan,indent=2));print('Warm CPU preflight and private GPU recipe prepared; initial snapshot guard and old pilot freeze included.')
