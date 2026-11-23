from pathlib import Path
from collections import Counter
import numpy as np
from .common import CLASSES,role_rows
from .features import normalized

def resize_classifier_image(image,size):
    """The identical float32 letterbox operation for fitting and prediction."""
    import tensorflow as tf
    return tf.image.resize_with_pad(tf.cast(image,tf.float32),size,size)

def load_classifier_image(path,size):
    import tensorflow as tf
    image=tf.io.decode_image(tf.io.read_file(str(path) if isinstance(path,Path) else path),channels=3,expand_animations=False)
    image.set_shape((None,None,3))
    return resize_classifier_image(image,size)

def training_weights(rows,c):
    duplicates=Counter(r['pixel_sha256'] for r in rows)
    base=np.array([1/np.sqrt(duplicates[r['pixel_sha256']]) if c['duplicate_weighting'] else 1 for r in rows])
    if c['class_balance']:
        counts=np.bincount([CLASSES.index(r['class_name']) for r in rows],weights=base,minlength=4)
        cw=np.minimum(base.sum()/np.maximum(4*counts,1),4.)
        base*=np.array([cw[CLASSES.index(r['class_name'])] for r in rows])
    return (base/base.mean()).astype('float32')

def make_data(root,role,c,table=None,normalizer=None,domains=None,limit=None):
    import tensorflow as tf
    rows=role_rows(root,role)
    if limit:rows=rows[:limit]
    mode=c['mode'];needs_anatomy=mode in ['roi','localized','fusion']
    if needs_anatomy and (table is None or normalizer is None):raise ValueError('Anatomy feature cache required for '+mode)
    x={'image':np.array([str(Path(root)/r['relative_path']) for r in rows])}
    if needs_anatomy:
        x['roi']=np.array([table.roi(r) for r in rows])
        if mode=='fusion':
            x['morphology']=np.array([normalized(table.vector(r),normalizer) for r in rows])
            if c['morphology_ablation']=='zeros':x['morphology']=np.zeros_like(x['morphology'])
    y=np.eye(4,dtype='float32')[[CLASSES.index(r['class_name']) for r in rows]]
    weights=training_weights(rows,c) if role=='fit' else np.ones(len(rows),'float32')
    if c['domain_weight']>0:
        target={'class':y,'domain':np.eye(len(domains),dtype='float32')[[domains.get(r['source'],0) for r in rows]]}
        sw={'class':weights,'domain':np.ones(len(rows),'float32')}
    else:target=y;sw=weights
    ds=tf.data.Dataset.from_tensor_slices((x,target,sw))
    if role=='fit':ds=ds.shuffle(len(rows),seed=c['seed'],reshuffle_each_iteration=True)
    def decode(features,label,weight):
        result={}
        for key,value in features.items():
            if key=='morphology':result[key]=value;continue
            result[key]=load_classifier_image(value,c['image_size'])
        if mode=='roi':result={'image':result['roi']}
        elif mode in ['legacy','image']:result={'image':result['image']}
        return result,label,weight
    ds=ds.map(decode,num_parallel_calls=c['data_threads'])
    options=tf.data.Options();options.threading.private_threadpool_size=c['data_threads']
    return ds.batch(c['batch_size']).with_options(options).prefetch(1),rows
