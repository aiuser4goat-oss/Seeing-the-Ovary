"""Cache validation predictions from frozen references; never reads test images."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import tensorflow as tf
from continuation.common import CLASSES,file_hash,dump
from continuation.evaluation import predict_runs,metrics
from continuation.artifacts import inference_signature

def main():
 p=argparse.ArgumentParser()
 for k in ['classification','frozen-root','features','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);tf.config.experimental.enable_op_determinism()
 freeze=json.loads((a.frozen_root/'experiment_freeze.json').read_text())['freeze'];assert file_hash(a.classification/'metadata/dataset_manifest.csv')==freeze['manifest_sha256']
 for name in ['original_baseline','image_only','fusion_refined']:
  run=a.frozen_root/name;assert inference_signature(run)==freeze['models'][name]['signature']
  folder=a.out/name;folder.mkdir(exist_ok=True);feature_root=a.features.parent/'features_refined' if name=='fusion_refined' else a.features
  rows,p,_,_=predict_runs(a.classification,[run],'select',feature_root,False)
  y=np.array([CLASSES.index(r['class_name']) for r in rows]);np.savez_compressed(folder/'selection_probabilities.npz',record_ids=np.array([r['record_id'] for r in rows]),classes=np.array(CLASSES),true_class_indices=y,raw_probabilities=p)
  dump(folder/'selection_metrics.json',metrics(y,p));dump(folder/'selection_receipt.json',{'role':'select','manifest_sha256':freeze['manifest_sha256'],'inference_signature':inference_signature(run),'probabilities_sha256':file_hash(folder/'selection_probabilities.npz'),'deterministic_operations':True,'source':'Existing frozen reference; validation cache only. Primary outputs unchanged.'})
  print('FROZEN_VALIDATION_REFERENCE',name,metrics(y,p)['balanced_accuracy'],flush=True);tf.keras.backend.clear_session();gc.collect()
if __name__=='__main__':main()
