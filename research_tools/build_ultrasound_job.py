from pathlib import Path
import json,hashlib,base64
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';K=B/'Kaggle_Launch/ultrasound-transfer';K.mkdir(exist_ok=True)
C=Path('/Users/ik-mac/ovarian_ai_project copy/accuracy_development');files={n:(C/n).read_text() for n in ['ultrasound_transfer.py','ultrasound_manifest.py']};files['tinyusfm.py']=(D/'ultrasound_research/model/tinyusfm.py').read_text();payload={n:{'text':s,'sha256':hashlib.sha256(s.encode()).hexdigest()} for n,s in files.items()}
(D/'ultrasound_transfer_payload_v3.json').write_text(json.dumps(payload,indent=2))
script='''import os,sys,json,hashlib,subprocess,zipfile,urllib.request
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='4'
OUT=Path('/kaggle/working/Ultrasound_Transfer');OUT.mkdir(exist_ok=True)
SOURCE=OUT/'source';SOURCE.mkdir(exist_ok=True)
PAYLOAD=__PAYLOAD__
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,entry in PAYLOAD.items():
 p=SOURCE/name;p.write_text(entry['text']);assert sha(p)==entry['sha256']
# Install only the small official-model dependency; retain Kaggle's paired torch/torchvision.
print('INSTALLING_TIMM',flush=True)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','timm==0.5.4','--quiet'],check=True)
print('TIMM_READY; DOWNLOADING_PRETRAINED_WEIGHTS',flush=True)
checkpoint=OUT/'TinyUSFM.pth'
url='https://drive.google.com/uc?export=download&id=1P_wDuU-StALLJEBaymioaYu52CScW-l-'
with urllib.request.urlopen(url,timeout=90) as source,checkpoint.open('wb') as target:
 import shutil;shutil.copyfileobj(source,target)
assert sha(checkpoint)=='776a50eeff402802c8cd06e00fe9de73df4161504fa733284bc240904b48252c'
print('CHECKPOINT_VERIFIED',flush=True)
expected='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
roots=[p.parent.parent for p in Path('/kaggle/input').rglob('dataset_manifest.csv') if p.parent.name=='metadata' and sha(p)==expected]
if roots:root=roots[0]
else:
 archives=list(Path('/kaggle/input').rglob('Classification.zip'));assert len(archives)==1
 dest=Path('/kaggle/working/classification_data');dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(archives[0]) as z:
  assert all((dest/n).resolve().is_relative_to(dest.resolve()) for n in z.namelist());z.extractall(dest)
 root=dest/'Classification'
assert sha(root/'metadata/dataset_manifest.csv')==expected
print('CLASSIFICATION_ROOT_READY',str(root),flush=True)
sys.path.insert(0,str(SOURCE))
from ultrasound_manifest import rows_for
resolved_root=root.resolve();checked_parents=set()
for role in ['fit','select']:
 rows=rows_for(root,role)
 if os.environ.get('TINY_USFM_PROBE_ONLY')=='1':
  rows=sum(([r for r in rows if r['class_name']==name][:2] for name in ['Benign','Malignant','Normal','PCOS']),[])
 for i,r in enumerate(rows):
  relative=Path(r['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts
  p=root/relative
  if p.parent not in checked_parents:
   assert p.parent.resolve().is_relative_to(resolved_root);checked_parents.add(p.parent)
  assert not p.is_symlink();assert sha(p)==r['sha256']
  if (i+1)%1000==0:print('VERIFY_PROGRESS',role,i+1,'/',len(rows),flush=True)
 print('VERIFIED_ROLE',role,len(rows),flush=True)
base=[sys.executable,'-u',str(SOURCE/'ultrasound_transfer.py'),'--classification',str(root),'--model-source',str(SOURCE/'tinyusfm.py'),'--checkpoint',str(checkpoint)]
subprocess.run(base+['--out',str(OUT/'smoke'),'--smoke','--max-minutes','10'],check=True)
print('TINY_USFM_SMOKE_PASSED',flush=True)
if os.environ.get('TINY_USFM_PROBE_ONLY')!='1':
 subprocess.run(base+['--out',str(OUT/'training'),'--max-minutes','100'],check=True)
with zipfile.ZipFile(OUT/'Ultrasound_Transfer_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.npz','.py','.pt'] and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(OUT)))
print('ULTRASOUND_TRANSFER_JOB_COMPLETE',flush=True)
'''.replace('__PAYLOAD__',repr(payload))
(K/'launch.py').write_text(script)
cell="import base64,subprocess,sys\nfrom pathlib import Path\np=Path('/kaggle/working/ultrasound_launch.py')\np.write_bytes(base64.b64decode("+repr(base64.b64encode(script.encode()).decode())+"))\nsubprocess.run([sys.executable,'-u',str(p)],check=True)\n"
nb={'cells':[{'cell_type':'markdown','metadata':{},'source':['Private exploratory TinyUSFM transfer experiment. Fit/select only. External pretraining overlap unknown; no test-based model selection.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(K/'transfer.ipynb').write_text(json.dumps(nb))
metadata={'id':'ishaankhera/ovarian-ultrasound-transfer-validation','title':'Ovarian Ultrasound Transfer Validation','code_file':'transfer.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':['ishaankhera/ovarian-continuation-2627-private'],'kernel_sources':[],'competition_sources':[]}
(K/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2));print('Created private job with source hashes and exact manifest/checkpoint guards.')
