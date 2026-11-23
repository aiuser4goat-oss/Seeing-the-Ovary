import tensorflow as tf
from .morphology import FEATURES
from .augmentation import make_augmentation

@tf.keras.utils.register_keras_serializable(package='Continuation')
class GradientReverse(tf.keras.layers.Layer):
    def call(self,x):
        @tf.custom_gradient
        def reverse(value):return value,lambda dy:-dy
        return reverse(x)

@tf.keras.utils.register_keras_serializable(package='Continuation')
class ConfusionMetric(tf.keras.metrics.Metric):
    def __init__(self,kind='balanced_accuracy',name=None,**kwargs):
        super().__init__(name=name or kind,**kwargs);self.kind=kind
        self.cm=self.add_weight(name='matrix',shape=(4,4),initializer='zeros')
    def update_state(self,y_true,y_pred,sample_weight=None):
        y=tf.argmax(y_true,axis=-1);p=tf.argmax(y_pred,axis=-1)
        self.cm.assign_add(tf.math.confusion_matrix(y,p,num_classes=4,dtype=tf.float32))
    def result(self):
        tp=tf.linalg.diag_part(self.cm);support=tf.reduce_sum(self.cm,axis=1)
        if self.kind=='balanced_accuracy':values=tf.math.divide_no_nan(tp,support)
        else:values=tf.math.divide_no_nan(2*tp,support+tf.reduce_sum(self.cm,axis=0))
        present=tf.cast(support>0,tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(values*present),tf.reduce_sum(present))
    def reset_state(self):self.cm.assign(tf.zeros((4,4)))
    def get_config(self):return {**super().get_config(),'kind':self.kind}

@tf.keras.utils.register_keras_serializable(package='Continuation')
class SmoothedFocalCE(tf.keras.losses.Loss):
    def __init__(self,smoothing=.03,gamma=0.,name='smoothed_focal_ce',**kwargs):
        super().__init__(name=name,**kwargs);self.smoothing=smoothing;self.gamma=gamma
    def call(self,y,p):
        p=tf.clip_by_value(p,1e-7,1-1e-7);smooth=y*(1-self.smoothing)+self.smoothing/4
        ce=-tf.reduce_sum(smooth*tf.math.log(p),axis=-1)
        factor=tf.pow(1-tf.reduce_sum(y*p,axis=-1),self.gamma)
        return factor*ce
    def get_config(self):return {**super().get_config(),'smoothing':self.smoothing,'gamma':self.gamma}

def build_classifier(c,domain_count=0):
    L=tf.keras.layers;size=c['image_size'];inp=L.Input((size,size,3),name='image')
    if c['mode']=='legacy':
        if c['domain_weight']:raise ValueError('Use image/fusion mode for domain experiments')
        from model.hybrid_model import build_hybrid_model
        original=build_hybrid_model((size,size,3),4,False,weights=c['weights'])
        augmented=make_augmentation(c)(inp) if c.get('augmentation',True) else inp
        model=tf.keras.Model({'image':inp},original(augmented),name='original_hybrid_rerun')
        wrapper=original.get_layer('EffB0_Backbone');encoder=next(l for l in wrapper.layers if isinstance(l,tf.keras.Model))
        wrapper.trainable=True;encoder.trainable=False
        return model,encoder
    constructors={'efficientnetb0':tf.keras.applications.EfficientNetB0,
        'efficientnetv2b0':tf.keras.applications.EfficientNetV2B0,
        'efficientnetv2s':tf.keras.applications.EfficientNetV2S,
        'convnexttiny':tf.keras.applications.ConvNeXtTiny}
    if c['backbone'] not in constructors:raise ValueError('Unsupported backbone')
    encoder=constructors[c['backbone']](include_top=False,weights=c['weights'],input_shape=(size,size,3),pooling='avg')
    encoder.trainable=False
    inputs={'image':inp};images=[inp]
    if c['mode'] in ['localized','fusion']:
        roi=L.Input((size,size,3),name='roi');inputs['roi']=roi;images.append(roi)
    augmented=make_augmentation(c)(images) if c.get('augmentation',True) else images
    # All supported applications include their own input preprocessing. RGB
    # reaches them in [0,255]; it must NOT be ImageNet-normalized a second time.
    global_feature=encoder(augmented[0],training=False)
    parts=[global_feature]
    if c['mode'] in ['localized','fusion']:
        parts.append(encoder(augmented[1],training=False))
    if c['mode']=='fusion':
        morphology=L.Input((2*len(FEATURES),),name='morphology');inputs['morphology']=morphology
        m=L.Dense(64,activation='gelu')(morphology);m=L.Dropout(.15)(m);parts.append(m)
    feature=parts[0] if len(parts)==1 else L.Concatenate()(parts)
    x=L.Dense(256,activation='gelu')(feature);x=L.LayerNormalization()(x);x=L.Dropout(c['dropout'])(x)
    probability=L.Dense(4,activation='softmax',dtype='float32',name='class')(x)
    outputs=probability
    if c['domain_weight']>0:
        if domain_count<2:raise ValueError('Domain training requires >=2 supplied sources')
        domain=L.Dense(64,activation='relu')(GradientReverse()(global_feature))
        outputs={'class':probability,'domain':L.Dense(domain_count,activation='softmax',dtype='float32',name='domain')(domain)}
    return tf.keras.Model(inputs,outputs,name='anatomy_guided_'+c['mode']),encoder

def unfreeze(encoder,fraction):
    encoder.trainable=fraction>0
    start=int(len(encoder.layers)*(1-fraction))
    for i,layer in enumerate(encoder.layers):
        layer.trainable=fraction>0 and i>=start and not isinstance(layer,tf.keras.layers.BatchNormalization)

def compile_classifier(model,c,lr,steps):
    lr=tf.keras.optimizers.schedules.CosineDecay(lr,decay_steps=max(steps,1),alpha=.05)
    optimizer=tf.keras.optimizers.AdamW(lr,weight_decay=c['weight_decay'],clipnorm=1.)
    loss=SmoothedFocalCE(c['label_smoothing'],c['focal_gamma'])
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),ConfusionMetric(),ConfusionMetric('macro_f1')]
    if c['domain_weight']>0:
        model.compile(optimizer,loss={'class':loss,'domain':'categorical_crossentropy'},
            loss_weights={'class':1.,'domain':c['domain_weight']},metrics={'class':metrics})
    else:model.compile(optimizer,loss=loss,metrics=metrics)
