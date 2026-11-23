"""NumPy-only role and weighting adapter; mirrors the frozen role definition."""
import csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
CLASSES=['Benign','Malignant','Normal','PCOS']
def rows_for(root,role):
 if role not in ['fit','select']:raise ValueError('Only fitting and selection roles permitted')
 with (Path(root)/'metadata/dataset_manifest.csv').open() as f:rows=list(csv.DictReader(f))
 if role=='fit':return [r for r in rows if r['training_role']=='fit']
 return [r for r in rows if r['training_role']=='val' and int(hashlib.sha256(r['image_group'].encode()).hexdigest()[:8],16)%3!=0]
def fitting_weights(rows):
 count=Counter(r['pixel_sha256'] for r in rows);base=np.array([1/np.sqrt(count[r['pixel_sha256']]) for r in rows]);y=np.array([CLASSES.index(r['class_name']) for r in rows]);effective=np.bincount(y,weights=base,minlength=4);cw=np.minimum(base.sum()/np.maximum(4*effective,1),4);w=base*cw[y];return (w/w.mean()).astype('float32')

def parse_args(argv=None):
 import argparse
 p=argparse.ArgumentParser()
 for name in ['classification','model_source','checkpoint','out']:
  options=['--'+name.replace('_','-')]
  if '_' in name:options.append('--'+name)
  p.add_argument(*options,dest=name,type=Path,required=True)
 p.add_argument('--smoke',action='store_true');p.add_argument('--max-minutes',type=float,default=100)
 return p.parse_args(argv)
