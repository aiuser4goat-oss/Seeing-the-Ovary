"""Fit/select only, deterministic decoding, training-only weighted Mixup."""
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
from continuation.common import CLASSES,role_rows
from continuation.data import load_classifier_image
from continuation.features import normalized

def fitting_weights(rows,power):
    if not 0<=power<=1:raise ValueError('Class-weight power must be in [0,1]')
    counts=Counter(r['pixel_sha256'] for r in rows)
    base=np.array([1/np.sqrt(counts[r['pixel_sha256']]) for r in rows],np.float64)
    effective=np.bincount([CLASSES.index(r['class_name']) for r in rows],weights=base,minlength=4)
    cw=np.minimum((base.sum()/np.maximum(4*effective,1))**power,4.)
    w=base*np.array([cw[CLASSES.index(r['class_name'])] for r in rows]);return (w/w.mean()).astype('float32')

def mix_batch(x,y,w,seed,alpha):
    n=tf.shape(y)[0];seed=tf.cast(seed,tf.int32)
    s=tf.random.experimental.stateless_split(tf.stack([seed,tf.bitwise.bitwise_xor(seed,tf.constant(1729,tf.int32))]),3)
    a=tf.random.stateless_gamma([n],s[0],alpha=alpha);b=tf.random.stateless_gamma([n],s[1],alpha=alpha)
    lam=tf.maximum(a/tf.maximum(a+b,1e-8),1-a/tf.maximum(a+b,1e-8));order=tf.random.experimental.stateless_shuffle(tf.range(n),s[2])
    mixed={k:v*tf.reshape(lam,[n]+[1]*(len(v.shape)-1))+tf.gather(v,order)*tf.reshape(1-lam,[n]+[1]*(len(v.shape)-1)) for k,v in x.items()}
    wa=lam*w;wb=(1-lam)*tf.gather(w,order);sw=wa+wb
    target=(y*wa[:,None]+tf.gather(y,order)*wb[:,None])/tf.maximum(sw[:,None],1e-8)
    return mixed,target,sw

def dataset(root,role,c,size,power=1.,mixup=0.,table=None,normalizer=None,smoke=False):
    if role not in ['fit','select']:raise ValueError('Development pipeline permits fit/select only; test/calibration access is forbidden')
    rows=role_rows(root,role)
    if smoke:
        selected=[]
        for name in CLASSES:selected.extend([r for r in rows if r['class_name']==name][:2])
        rows=selected
    x={'image':np.array([str(Path(root)/r['relative_path']) for r in rows])}
    if table is not None:
        if mixup:raise ValueError('Image Mixup is not used for measured-anatomy correction fitting')
        x['morphology']=np.array([normalized(table.vector(r),normalizer) for r in rows])
    y=np.eye(4,dtype=np.float32)[[CLASSES.index(r['class_name']) for r in rows]]
    weights=fitting_weights(rows,power) if role=='fit' else np.ones(len(rows),np.float32)
    ds=tf.data.Dataset.from_tensor_slices((x,y,weights))
    if role=='fit':ds=ds.shuffle(len(rows),seed=c['seed'],reshuffle_each_iteration=True)
    def decode(x,y,w):return {k:(load_classifier_image(v,size) if k=='image' else v) for k,v in x.items()},y,w
    ds=ds.map(decode,num_parallel_calls=c.get('data_threads',4),deterministic=True).batch(c['batch_size'])
    if role=='fit' and mixup:
        # Dataset.random emits reproducible but different seeds across epochs.
        random=tf.data.Dataset.random(seed=c['seed'],rerandomize_each_iteration=True)
        ds=tf.data.Dataset.zip((ds,random)).map(lambda b,s:mix_batch(*b,s,mixup),num_parallel_calls=1,deterministic=True)
    options=tf.data.Options();options.deterministic=True;options.threading.private_threadpool_size=c.get('data_threads',4)
    return ds.with_options(options).prefetch(1),rows
