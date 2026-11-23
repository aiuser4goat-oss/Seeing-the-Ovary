"""Executable research inference for the frozen-feature routing experiment (uncalibrated)."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.special import expit

def main():
 p=argparse.ArgumentParser()
 for name in ['image','code','global-run','bundle','out']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.code));import tensorflow as tf
 from continuation.common import runtime,CLASSES
 from continuation.models import build_classifier
 from continuation.data import load_classifier_image
 sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 parameters=np.load(a.bundle,allow_pickle=False);bundle=json.loads(str(parameters['metadata']));assert sha(a.global_run/'best.weights.h5')==bundle['declaration']['global_checkpoint_sha256'];assert bundle['declaration']['class_order']==CLASSES;assert bundle['feature_order']==['encoder_pool','pre_layernorm_head']
 config=json.loads((a.global_run/'config.json').read_text());assert config['mode']=='image' and config['domain_weight']==0;config.update(weights=None,mixed_precision=False);runtime(config);tf.config.experimental.enable_op_determinism();tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(4)
 model,_=build_classifier(config);model.load_weights(a.global_run/'best.weights.h5');layers=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense) and l.units==256];assert len(layers)==1
 extractor=tf.keras.Model(model.input,[model.output,layers[0].input,layers[0].output]);image=tf.expand_dims(load_classifier_image(a.image,config['image_size']),0);prob,pool,head=extractor({'image':image},training=False);features=np.concatenate([pool.numpy(),head.numpy()],axis=1);x=features.astype('float32').copy();x-=parameters['scaler_mean'];x/=parameters['scaler_scale'];x=x@parameters['pca_components'].T;x-=parameters['pca_mean'][None,:]@parameters['pca_components'].T;norm=np.sqrt(np.sum(x*x,axis=1,keepdims=True));norm[norm==0]=1.;x/=norm;normal=float(expit(x@parameters['coefficient'].T+parameters['intercept']).ravel()[0]);base=int(prob.numpy()[0].argmax());final=2 if base==0 and normal>=bundle['routing_threshold'] else base
 result={'base_prediction':CLASSES[base],'final_prediction':CLASSES[final],'routed':final!=base,'specialist_normal_score':normal,'base_model_probabilities_uncalibrated':dict(zip(CLASSES,map(float,prob.numpy()[0]))),'confidence_calibration':'No calibration of the routed system has been performed. Scores are not calibrated final-class probabilities.','global_checkpoint_sha256':sha(a.global_run/'best.weights.h5'),'bundle_sha256':sha(a.bundle),'tensorflow':tf.__version__,'keras':tf.keras.__version__,'numpy':np.__version__,'scope':'Research prototype. Cached-feature experiments and runtime precision must be distinguished; no clinical validation claim.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
