"""Experimental classifiers; existing continuation model definitions are untouched."""
import tensorflow as tf
from continuation.models import build_classifier as build_original
from continuation.common import file_hash

@tf.keras.utils.register_keras_serializable(package='AccuracyDevelopment')
class AppearanceAugment(tf.keras.layers.Layer):
    def __init__(self,gain=.12,contrast=.08,gamma=.08,speckle=.015,seed=2627,**kw):
        kw.setdefault('dtype','float32');super().__init__(**kw)
        self.gain=gain;self.contrast=contrast;self.gamma=gamma;self.speckle=speckle;self.seed=seed
        self.generator=tf.keras.random.SeedGenerator(seed)
    def call(self,x,training=None):
        if training is None or training is False:return x
        def augment():
            x32=tf.cast(x,tf.float32)/255.;shape=tf.stack([tf.shape(x32)[0],1,1,1])
            def uniform(lo,hi):return tf.keras.random.uniform(shape,minval=lo,maxval=hi,seed=self.generator)
            # Horizontal reflection preserves ultrasound depth orientation.
            x32=tf.where(uniform(0.,1.)<.5,tf.reverse(x32,[2]),x32)
            foreground=tf.reduce_any(x32>0,axis=-1,keepdims=True);weight=tf.cast(foreground,tf.float32)
            mean=tf.math.divide_no_nan(tf.reduce_sum(x32*weight,[1,2],keepdims=True),tf.reduce_sum(weight,[1,2],keepdims=True))
            x32=tf.clip_by_value((x32-mean)*uniform(1-self.contrast,1+self.contrast)+mean,0,1)
            x32=tf.pow(x32,uniform(1-self.gamma,1+self.gamma))*uniform(1-self.gain,1+self.gain)
            if self.speckle:
                noise=tf.keras.random.normal(tf.concat([tf.shape(x32)[:-1],[1]],0),seed=self.generator)
                x32=x32*(1+noise*uniform(0.,self.speckle))
            return tf.where(foreground,tf.clip_by_value(x32,0,1),0)*255.
        return augment() if training is True else tf.cond(tf.cast(training,tf.bool),augment,lambda:x)
    def get_config(self):return {**super().get_config(),**{k:getattr(self,k) for k in ['gain','contrast','gamma','speckle','seed']}}

def build(c,weights='imagenet'):
    L=tf.keras.layers;image=L.Input((None,None,3),name='image')
    constructors={'efficientnetv2b0':tf.keras.applications.EfficientNetV2B0,'efficientnetv2s':tf.keras.applications.EfficientNetV2S,'convnexttiny':tf.keras.applications.ConvNeXtTiny}
    encoder=constructors[c['backbone']](include_top=False,weights=weights,input_shape=(None,None,3),pooling='avg')
    encoder.trainable=False
    x=AppearanceAugment(seed=c['seed'],**c.get('appearance',{}),name='appearance_augmentation')(image)
    x=encoder(x,training=False);x=L.Dense(256,activation='gelu',name='head_dense')(x)
    x=L.LayerNormalization(name='head_norm')(x);x=L.Dropout(c.get('dropout',.25),name='head_dropout')(x)
    logits=L.Dense(4,dtype='float32',name='class_logits')(x)
    probability=L.Activation('softmax',dtype='float32',name='class')(logits)
    return tf.keras.Model({'image':image},probability,name='development_'+c['backbone']),encoder

def warm_start(model,encoder,run):
    """Transfer named encoder/head parameters, not RNG state or input dimensions."""
    import json
    from pathlib import Path
    run=Path(run);c=json.loads((run/'config.json').read_text())
    if c['mode']!='image' or c['backbone']!='efficientnetv2b0':raise ValueError('Warm start requires the declared image-only B0 checkpoint')
    old,old_encoder=build_original({**c,'weights':None});old.load_weights(run/'best.weights.h5')
    encoder.set_weights(old_encoder.get_weights())
    dense=[l for l in old.layers if isinstance(l,tf.keras.layers.Dense) and l.units==256]
    norm=[l for l in old.layers if isinstance(l,tf.keras.layers.LayerNormalization)]
    if len(dense)!=1 or len(norm)!=1:raise ValueError('Unexpected original head topology')
    model.get_layer('head_dense').set_weights(dense[0].get_weights());model.get_layer('head_norm').set_weights(norm[0].get_weights())
    model.get_layer('class_logits').set_weights(old.get_layer('class').get_weights())
    # Check transfer at the SAME resolution before any training or resizing change.
    x=tf.reshape(tf.linspace(0.,255.,c['image_size']*c['image_size']*3),[1,c['image_size'],c['image_size'],3])
    a=old({'image':x},training=False).numpy();b=model({'image':x},training=False).numpy()
    import numpy as np
    if not np.allclose(a,b,atol=2e-5,rtol=2e-5):raise ValueError('Transferred classifier is not equivalent before fine tuning')
    return {'checkpoint_sha256':file_hash(run/'best.weights.h5'),'config_sha256':file_hash(run/'config.json'),'synthetic_transfer_max_abs':float(np.max(np.abs(a-b)))}

def set_trainable(encoder,fraction):
    encoder.trainable=fraction>0;start=int(len(encoder.layers)*(1-fraction))
    for i,l in enumerate(encoder.layers):l.trainable=fraction>0 and i>=start and not isinstance(l,tf.keras.layers.BatchNormalization)

def residual_anatomy(base,feature_count=13):
    """Zero-initialized bounded correction: initial output equals the image model."""
    L=tf.keras.layers;image=L.Input((None,None,3),name='image');m=L.Input((2*feature_count,),name='morphology')
    frozen=tf.keras.Model(base.input,base.get_layer('class_logits').output,name='frozen_image_logits');frozen.trainable=False
    logits=frozen({'image':image},training=False)
    h=L.Dense(64,activation='gelu',name='anatomy_hidden')(m);h=L.Dropout(.2)(h)
    delta=L.Dense(4,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',dtype='float32',name='anatomy_correction')(h)
    gate=L.Dense(1,activation='sigmoid',kernel_initializer='zeros',bias_initializer=tf.keras.initializers.Constant(-2.),dtype='float32',name='anatomy_gate')(m)
    valid=1-tf.keras.ops.clip(m[:,feature_count:feature_count+1],0,1)
    p=L.Activation('softmax',dtype='float32',name='class')(logits+delta*gate*valid)
    return tf.keras.Model({'image':image,'morphology':m},p,name='bounded_residual_anatomy')
