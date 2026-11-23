from pathlib import Path
from collections import Counter
import hashlib, io, json, zipfile
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from .common import read_csv, write_csv, dump, runtime, file_hash
from .morphology import instances_from_predictions,instance_matches

class DataArchive:
    """Read FollicleFinder whether a cloud platform kept or expanded its ZIP."""
    def __init__(self,folder):
        self.folder=Path(folder);self.archive=None
    def __enter__(self):
        if (self.folder/'Data.zip').is_file():self.archive=zipfile.ZipFile(self.folder/'Data.zip')
        else:
            # Kaggle expands Data.zip into Data/, preserving its internal Data/.
            indices=[p for p in self.folder.rglob('image_metadata.csv')
                     if p.parent.name=='BiCycleStudy' and p.parent.parent.name=='Data']
            if len(indices)!=1:
                raise ValueError('Expected exactly one FollicleFinder Data/BiCycleStudy/image_metadata.csv; found '+str(len(indices)))
            self.folder=indices[0].parent.parent.parent
        return self
    def __exit__(self,*args):
        if self.archive:self.archive.close()
    def read(self,name):return self.archive.read(name) if self.archive else (self.folder/name).read_bytes()
    def namelist(self):
        return self.archive.namelist() if self.archive else [str(p.relative_to(self.folder)) for p in (self.folder/'Data').rglob('*') if p.is_file()]

def padded(array,size,labels=False):
    im=Image.fromarray(array)
    return np.asarray(ImageOps.pad(im,(size,size),method=Image.Resampling.NEAREST if labels else Image.Resampling.BILINEAR))

def instance_distance(labels):
    distance=np.zeros(labels.shape,'float32')
    for i in np.unique(labels):
        if i:
            d=ndi.distance_transform_edt(labels==i)
            distance=np.maximum(distance,d/max(d.max(),1))
    return distance

def prepare(anatomy_root,out,c,sources=('usova','folliclefinder'),max_volumes=None,slice_step=4,classification_root=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];excluded=[];root=Path(anatomy_root)
    def save(image,ovary,instances,key,split,group,instance_valid,source):
        image=np.asarray(image)
        if image.ndim==3 and image.shape[-1] in [3,4]:image=np.asarray(Image.fromarray(image).convert('L'))
        if image.ndim!=2:raise ValueError('Expected 2D grayscale slice')
        if not (image.shape==ovary.shape==instances.shape):raise ValueError('Anatomy shape mismatch')
        if image.dtype!=np.uint8:
            finite=image[np.isfinite(image)];lo,hi=np.percentile(finite,[.5,99.5])
            image=np.clip((image-lo)*255/max(hi-lo,1e-8),0,255).astype('uint8')
        original_rgb=np.repeat(image[...,None],3,axis=-1)
        raw_pixel_hash=hashlib.sha256(f'{image.shape[1]}x{image.shape[0]}:RGB:'.encode()+original_rgb.tobytes()).hexdigest()
        image=padded(image,c['seg_size']);ovary=padded(ovary.astype('uint8'),c['seg_size'],True)>0
        instances=padded(instances.astype('int32'),c['seg_size'],True).astype('uint16')
        ovary|=instances>0
        distance=instance_distance(instances) if instance_valid else np.zeros(image.shape,'float32')
        name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
        np.savez_compressed(out/name,image=image,ovary=ovary.astype('uint8'),
                            follicle=(instances>0).astype('uint8'),instances=instances,distance=distance)
        rows.append(dict(file=name,split=split,group=group,source=source,instance_valid=int(instance_valid),raw_pixel_sha256=raw_pixel_hash))
    if 'usova' in sources:
        base=root/'USOVA3D';pairs=read_csv(base/'image_mask_pairs.csv')
        volumes=list(dict.fromkeys(r['volume_id'] for r in pairs))
        if max_volumes:volumes=volumes[:max_volumes]
        for r in pairs:
            if r['volume_id'] not in volumes:continue
            index=int(Path(r['image']).stem.split('_')[-1])
            if index%slice_step:continue
            raw=np.array(Image.open(base/r['image']).convert('L'))
            ov=np.array(Image.open(base/r['ovary_r1']))>0
            ins=np.array(Image.open(base/r['follicle_r1_labels']))
            save(raw,ov,ins,'usova/'+r['image'],r['split'],'usova/'+r['volume_id'],True,'USOVA3D')
    if 'folliclefinder' in sources:
        import pydicom,h5py
        with DataArchive(root/'FollicleFinder') as z:
            index=list(__import__('csv').DictReader(io.StringIO(z.read('Data/BiCycleStudy/image_metadata.csv').decode('utf-8-sig'))))
            patient_folds={r['PatientID']:r['fold'] for r in index}
            def split(fold):return 'test' if fold=='test_fold' else 'val' if fold=='fold_0' else 'train'
            for r in index[:max_volumes] if max_volumes else index:
                uid=r['SOPInstanceUID'];prefix='Data/BiCycleStudy/'
                raw=pydicom.dcmread(io.BytesIO(z.read(prefix+'raw_ultrasound_images/'+uid+'.dcm')))
                seg=pydicom.dcmread(io.BytesIO(z.read(prefix+'validated_segmentations/'+uid+'.dcm')))
                image=raw.pixel_array;mask=seg.pixel_array
                if image.shape!=mask.shape or image.ndim!=3:raise ValueError('Unsupported DICOM grid: '+uid)
                # This release uses integer semantic labels, not the conventional
                # separate-frame binary DICOM SEG representation. Reject others.
                names={str(s.SegmentLabel).lower():int(s.SegmentNumber) for s in seg.SegmentSequence}
                if not {'ovary','follicle'}<=names.keys():raise ValueError('Missing semantic label map')
                if not set(np.unique(mask))<=set(names.values()):raise ValueError('Unknown DICOM label values')
                for attr in ['FrameOfReferenceUID','ImageOrientationPatient','PixelSpacing']:
                    a,b=getattr(raw,attr,None),getattr(seg,attr,None)
                    if a is not None and b is not None and str(a)!=str(b):raise ValueError('DICOM grid metadata differs: '+attr)
                fg=mask==names['follicle']
                ovarian_names=['ovary','follicle','cl_tissue','cl_cavity','endometrioma','ovarian_cyst']
                ov=np.isin(mask,[names[n] for n in ovarian_names if n in names])
                # Private centroid annotations are NOT claimed as expert instance
                # boundaries. Semantic volumes supervise binary heads only.
                for i in range(0,len(image),slice_step):
                    save(image[i],ov[i],fg[i].astype('uint16'),'ff/'+uid+'/'+str(i),
                         split(r['fold']),'ff/'+r['PatientID'],False,'FollicleFinder_semantic')
                print('Prepared FollicleFinder volume',len(rows),'slices total',flush=True)
            # One observer supplies genuine instance boundaries on six volumes.
            h5_names=[n for n in z.namelist() if '/lf_manual_segmentation/' in n and n.endswith('.h5') and '/._' not in n]
            for name in h5_names[:max_volumes] if max_volumes else h5_names:
                participant=Path(name).stem.split('_')[0]
                if participant not in patient_folds:raise ValueError('Manual case lacks source fold')
                with h5py.File(io.BytesIO(z.read(name)),'r') as h:
                    keys={k.lower():k for k in h}
                    ok=keys.get('ovary',keys.get('ovar'));fk=keys['follicle']
                    raw_key=next(k for k in h if k not in [ok,fk])
                    for i in range(0,h[raw_key].shape[0],slice_step):
                        save(h[raw_key][i],h[ok][i]>0,h[fk][i],'ffmanual/'+name+'/'+str(i),
                             split(patient_folds[participant]),'ff/'+participant,True,'FollicleFinder_manual')
    if not rows:raise ValueError('No anatomy samples prepared')
    overlap_groups=set()
    if classification_root:
        protected={r['pixel_sha256'] for r in read_csv(Path(classification_root)/'metadata/dataset_manifest.csv') if r['training_role']!='fit'}
        overlap_groups={r['group'] for r in rows if r['raw_pixel_sha256'] in protected}
        excluded=[r for r in rows if r['group'] in overlap_groups]
        if excluded:write_csv(out/'excluded_classification_overlap.csv',excluded)
        rows=[r for r in rows if r['group'] not in overlap_groups]
        if not rows:raise ValueError('All prepared anatomy groups overlap protected classification cases')
    partitions={}
    for r in rows:partitions.setdefault(r['group'],set()).add(r['split'])
    if any(len(s)>1 for s in partitions.values()):raise ValueError('Anatomy group overlap')
    write_csv(out/'manifest.csv',rows)
    dump(out/'preparation.json',{'samples':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
        'config':c,'slice_step':slice_step,'instance_supervision':'USOVA r1 + one FollicleFinder manual observer; DICOM semantic volumes have zero distance-head weight.',
        'protected_classification_overlap_groups_removed':sorted(overlap_groups),
        'cross_pipeline_overlap_screen':'Exact original grayscale RGB pixels only; crops, color overlays and transformed variants may be missed.' if classification_root else 'Not performed; pass --classification for the recommended pipeline.',
        'scope':'2D slice adaptation, not reproduction of FollicleFinder 3D LSD network.'})
    return rows

def build_segmenter(c):
    import tensorflow as tf
    L=tf.keras.layers
    inp=L.Input((c['seg_size'],c['seg_size'],1),name='image')
    def block(x,n):
        for _ in range(2):
            x=L.Conv2D(n,3,padding='same',kernel_initializer='he_normal')(x)
            x=L.LayerNormalization()(x);x=L.Activation('relu')(x)
        return x
    x=inp;skips=[]
    for mult in [1,2,4,8]:
        x=block(x,c['seg_filters']*mult);skips.append(x);x=L.MaxPool2D()(x)
    x=block(x,c['seg_filters']*16);x=L.Dropout(.2)(x)
    for mult,skip in zip([8,4,2,1],reversed(skips)):
        x=L.UpSampling2D(interpolation='bilinear')(x);x=L.Concatenate()([x,skip]);x=block(x,c['seg_filters']*mult)
    heads={n:L.Conv2D(1,1,activation='sigmoid',dtype='float32',name=n)(x) for n in ['ovary','follicle','distance']}
    return tf.keras.Model(inp,heads,name='ovary_follicle_unet')

def segmentation_data(root,role,c,limit=None):
    import tensorflow as tf
    rows=[r for r in read_csv(Path(root)/'manifest.csv') if r['split']==role]
    if limit:rows=rows[:limit]
    if not rows:raise ValueError('No anatomy samples for '+role)
    def generator():
        for r in rows:
            with np.load(Path(root)/r['file']) as z:
                x=z['image'].astype('float32')[...,None]/255
                y={n:z[n].astype('float32')[...,None] for n in ['ovary','follicle','distance']}
            weights={n:np.float32(1 if n!='distance' else int(r['instance_valid'])) for n in y}
            yield x,y,weights
    shape=(c['seg_size'],c['seg_size'],1)
    ds=tf.data.Dataset.from_generator(generator,output_signature=(tf.TensorSpec(shape,tf.float32),
        {n:tf.TensorSpec(shape,tf.float32) for n in ['ovary','follicle','distance']},
        {n:tf.TensorSpec((),tf.float32) for n in ['ovary','follicle','distance']}))
    ds=ds.apply(tf.data.experimental.assert_cardinality(len(rows)))
    if role=='train':
        ds=ds.shuffle(min(len(rows),512),seed=c['seed'])
        from .augmentation import make_augmentation
        photo=make_augmentation(c,flips=False,value_range=1.,seed_offset=17)
        def augment(x,y,w):
            joined=tf.concat([x,y['ovary'],y['follicle'],y['distance']],axis=-1)
            if c.get('augmentation',True):joined=tf.image.random_flip_left_right(tf.image.random_flip_up_down(joined))
            x=photo(joined[None,...,:1],training=True)[0] if c.get('augmentation',True) else joined[...,:1]
            return x,{n:joined[...,i+1:i+2] for i,n in enumerate(['ovary','follicle','distance'])},w
        ds=ds.map(augment,num_parallel_calls=c['data_threads'])
    options=tf.data.Options();options.threading.private_threadpool_size=c['data_threads']
    return ds.batch(c['batch_size']).with_options(options).prefetch(1)

def train(root,out,c,resume=False):
    tf=runtime(c);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest_hash=file_hash(Path(root)/'manifest.csv')
    if any((out/name).exists() for name in ['config.json','best.weights.h5']) and not resume:
        raise ValueError('Anatomy run artifacts exist; use --resume or a new run directory')
    if resume and (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=c:raise ValueError('Anatomy resume config differs')
        if json.loads((out/'training_data.json').read_text())['manifest_sha256']!=manifest_hash:raise ValueError('Anatomy resume data differs')
    model=build_segmenter(c)
    if resume and (out/'best.weights.h5').exists():model.load_weights(out/'best.weights.h5')
    def dice_bce(y,p):
        axes=(1,2,3);dice=(2*tf.reduce_sum(y*p,axis=axes)+1)/(tf.reduce_sum(y+p,axis=axes)+1)
        bce=tf.reduce_mean(tf.keras.backend.binary_crossentropy(y,p),axis=axes)
        return bce+1-dice
    def distance_loss(y,p):return tf.reduce_mean(tf.square(y-p),axis=(1,2,3))
    model.compile(tf.keras.optimizers.AdamW(c['seg_lr'],weight_decay=c['weight_decay'],clipnorm=1),
        loss={'ovary':dice_bce,'follicle':dice_bce,'distance':distance_loss},loss_weights={'ovary':1.,'follicle':1.,'distance':.5})
    train_ds=segmentation_data(root,'train',c);val_ds=segmentation_data(root,'val',c)
    prior_loss=float(model.evaluate(val_ds,return_dict=True,verbose=0)['loss']) if resume and (out/'best.weights.h5').exists() else None
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True,
            initial_value_threshold=prior_loss),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=c['patience'],restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-6),tf.keras.callbacks.CSVLogger(str(out/'history.csv'),append=resume)]
    dump(out/'config.json',c)
    dump(out/'training_data.json',{'manifest_sha256':manifest_hash})
    model.fit(train_ds,validation_data=val_ds,epochs=c['seg_epochs'],callbacks=callbacks,shuffle=False,verbose=2)
    model.load_weights(out/'best.weights.h5')
    dump(out/'artifact.json',{'checkpoint_sha256':file_hash(out/'best.weights.h5'),
        'training_manifest_sha256':file_hash(Path(root)/'manifest.csv'),'thresholds_selected':'defaults; optional postprocessing selection must use validation only'})
    return model

def evaluate(root,run,out,role='test'):
    if json.loads((Path(run)/'training_data.json').read_text())['manifest_sha256']!=file_hash(Path(root)/'manifest.csv'):
        raise ValueError('Anatomy evaluation dataset differs from training data')
    c=json.loads((Path(run)/'config.json').read_text());runtime(c)
    model=build_segmenter(c);model.load_weights(Path(run)/'best.weights.h5')
    rows=[r for r in read_csv(Path(root)/'manifest.csv') if r['split']==role]
    if not rows:raise ValueError('No anatomy samples for '+role)
    results=[]
    for r in rows:
        with np.load(Path(root)/r['file']) as z:
            pred=model(z['image'][None,...,None]/255.,training=False)
            ov,ins=instances_from_predictions(pred['ovary'][0,...,0].numpy(),pred['follicle'][0,...,0].numpy(),pred['distance'][0,...,0].numpy(),
                c['ovary_threshold'],c['follicle_threshold'],c['min_follicle_area'],c['min_peak_distance'],
        method=c.get('follicle_instance_method','watershed'))
            result={'file':r['file'],'group':r['group']}
            for name,mask in [('ovary',ov),('follicle',ins>0)]:
                truth=z[name]>0;intersection=np.count_nonzero(mask&truth)
                result[name+'_dice']=(2*intersection+1)/(mask.sum()+truth.sum()+1)
                result[name+'_iou']=(intersection+1)/(np.count_nonzero(mask|truth)+1)
            if int(r['instance_valid']):
                gt=z['instances'];result['count_absolute_error']=abs(np.count_nonzero(np.unique(ins))-np.count_nonzero(np.unique(gt)))
                result.update(instance_matches(gt,ins))
            results.append(result)
    fields=['file','group','ovary_dice','ovary_iou','follicle_dice','follicle_iou','count_absolute_error','instance_tp','instance_fp','instance_fn']
    write_csv(Path(out)/'anatomy_predictions.csv',results,fields)
    report={k:float(np.mean([r[k] for r in results if k in r])) for k in fields[2:] if any(k in r for r in results)}
    tp=sum(r.get('instance_tp',0) for r in results);fp=sum(r.get('instance_fp',0) for r in results);fn=sum(r.get('instance_fn',0) for r in results)
    valid=any('instance_tp' in r for r in results)
    report.update(instance_precision_iou50=tp/max(tp+fp,1) if valid else None,
        instance_recall_iou50=tp/max(tp+fn,1) if valid else None,instance_f1_iou50=2*tp/max(2*tp+fp+fn,1) if valid else None)
    dump(Path(out)/'anatomy_metrics.json',report)
