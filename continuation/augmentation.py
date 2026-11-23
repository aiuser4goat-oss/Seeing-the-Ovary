"""Conservative image-only appearance changes; never changes annotation labels."""
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Continuation')
class UltrasoundAugment(tf.keras.layers.Layer):
    def __init__(self,gain=.15,contrast=.10,gamma=.10,speckle=.025,
                 flips=True,value_range=255.,seed=2627,**kwargs):
        kwargs.setdefault('dtype','float32')
        super().__init__(**kwargs)
        for name,value in [('gain',gain),('contrast',contrast),('gamma',gamma),('speckle',speckle)]:
            if not 0<=value<.5:raise ValueError(name+' must be in [0, 0.5)')
        self.gain=gain;self.contrast=contrast;self.gamma=gamma;self.speckle=speckle
        self.flips=flips;self.value_range=value_range;self.seed=seed
        self.generator=tf.keras.random.SeedGenerator(seed)

    def get_config(self):
        return {**super().get_config(),'gain':self.gain,'contrast':self.contrast,
            'gamma':self.gamma,'speckle':self.speckle,'flips':self.flips,
            'value_range':self.value_range,'seed':self.seed}

    def call(self,inputs,training=None):
        # Identity at validation/test/prediction, including no RNG advancement.
        if training is None or training is False:return inputs
        def augment():
            views=list(inputs) if isinstance(inputs,(list,tuple)) else [inputs]
            shape=tf.stack([tf.shape(views[0])[0],1,1,1])
            def uniform(lo,hi):
                if lo==hi:return tf.ones(shape,tf.float32)*lo
                return tf.keras.random.uniform(shape,minval=lo,maxval=hi,seed=self.generator)
            gain=uniform(1-self.gain,1+self.gain)
            contrast=uniform(1-self.contrast,1+self.contrast)
            gamma=uniform(1-self.gamma,1+self.gamma)
            horizontal=uniform(0,1)<.5;vertical=uniform(0,1)<.5
            sigma=uniform(0,self.speckle)
            outputs=[]
            for view in views:
                x=tf.cast(view,tf.float32)/self.value_range
                if self.flips:
                    x=tf.where(horizontal,tf.reverse(x,[2]),x)
                    x=tf.where(vertical,tf.reverse(x,[1]),x)
                foreground=tf.reduce_any(x>0,axis=-1,keepdims=True)
                weight=tf.cast(foreground,tf.float32)
                mean=tf.math.divide_no_nan(tf.reduce_sum(x*weight,axis=(1,2),keepdims=True),tf.reduce_sum(weight,axis=(1,2),keepdims=True))
                x=tf.clip_by_value((x-mean)*contrast+mean,0,1)
                x=tf.pow(x,gamma)*gain
                if self.speckle:
                    # Shared-channel noise avoids synthetic color artifacts.
                    noise_shape=tf.concat([tf.shape(x)[:-1],[1]],axis=0)
                    noise=tf.keras.random.normal(noise_shape,seed=self.generator)*sigma
                    x=x*(1+noise)
                x=tf.where(foreground,tf.clip_by_value(x,0,1),0)
                outputs.append(x*self.value_range)
            return outputs if isinstance(inputs,(list,tuple)) else outputs[0]
        if training is True:return augment()
        return tf.cond(tf.cast(training,tf.bool),augment,lambda:inputs)

def make_augmentation(c,flips=True,value_range=255.,seed_offset=0):
    return UltrasoundAugment(gain=c.get('augment_gain',.15),contrast=c.get('augment_contrast',.10),
        gamma=c.get('augment_gamma',.10),speckle=c.get('augment_speckle',.025),
        flips=flips,value_range=value_range,seed=c['seed']+seed_offset,name='train_only_augmentation')
