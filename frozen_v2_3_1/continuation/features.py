from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps
from .common import read_csv,write_csv,dump,file_hash,runtime
from .anatomy import build_segmenter
from .morphology import FEATURES,instances_from_predictions,measurements,crop_bounds

def postprocess(im,pred,c):
    size=c['seg_size']
    ov,ins=instances_from_predictions(pred['ovary'],pred['follicle'],pred['distance'],
        c['ovary_threshold'],c['follicle_threshold'],c['min_follicle_area'],c['min_peak_distance'],
        method=c.get('follicle_instance_method','watershed'))
    vector=measurements(ov,ins)
    scale=min(size/im.width,size/im.height)
    w=max(1,round(im.width*scale));h=max(1,round(im.height*scale));left=(size-w)//2;top=(size-h)//2
    y0,y1,x0,x1=crop_bounds(ov)
    bounds=(max(0,int((x0-left)/scale)),max(0,int((y0-top)/scale)),
            min(im.width,int(np.ceil((x1-left)/scale))),min(im.height,int(np.ceil((y1-top)/scale))))
    if np.isnan(vector).all() or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:bounds=(0,0,im.width,im.height)
    return im.crop(bounds),vector,ov,ins

def extract(classification_root,seg_run,out,batch_size=16):
    """Inference-only anatomy cache; never uses classification targets."""
    root=Path(classification_root);seg_run=Path(seg_run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=json.loads((seg_run/'config.json').read_text());runtime(c)
    model=build_segmenter(c);model.load_weights(seg_run/'best.weights.h5')
    rows=read_csv(root/'metadata/dataset_manifest.csv')
    unique={}
    for r in rows:unique.setdefault(r['pixel_sha256'],r)
    cached={};items=list(unique.items());size=c['seg_size']
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];images=[];originals=[]
        for key,r in batch:
            im=Image.open(root/r['relative_path']).convert('RGB');originals.append(im)
            images.append(np.array(ImageOps.pad(im.convert('L'),(size,size),method=Image.Resampling.BILINEAR))[...,None]/255.)
        pred=model(np.array(images,dtype='float32'),training=False)
        for i,((key,r),im) in enumerate(zip(batch,originals)):
            crop,vector,ov,ins=postprocess(im,{n:pred[n][i,...,0].numpy() for n in pred},c)
            crop_name='roi/'+key+'.png';target=out/crop_name;target.parent.mkdir(exist_ok=True)
            crop.save(target)
            entry={'pixel_sha256':key,'roi_path':crop_name,'roi_sha256':file_hash(target)}
            entry.update({n:float(v) if np.isfinite(v) else '' for n,v in zip(FEATURES,vector)})
            cached[key]=entry
        print('Anatomy inference',min(start+batch_size,len(items)),'/',len(items),flush=True)
    write_csv(out/'features.csv',list(cached.values()))
    dump(out/'provenance.json',{'cache_version':2,'classifier_manifest_sha256':file_hash(root/'metadata/dataset_manifest.csv'),
        'segmenter_checkpoint_sha256':file_hash(seg_run/'best.weights.h5'),'feature_names':FEATURES,
        'segmenter_config_sha256':file_hash(seg_run/'config.json'),'features_csv_sha256':file_hash(out/'features.csv'),
        'scope':'Visible 2D follicles; relative sizes, not millimetres or total ovarian follicle counts. Missing anatomy remains missing, not zero-count anatomy.'})

class FeatureTable:
    def __init__(self,root):
        self.root=Path(root)
        provenance=json.loads((self.root/'provenance.json').read_text())
        if provenance['features_csv_sha256']!=file_hash(self.root/'features.csv'):raise ValueError('Anatomy feature table changed after extraction')
        self.rows={r['pixel_sha256']:r for r in read_csv(self.root/'features.csv')}
        self.checked_rois=set()
    def vector(self,row):
        r=self.rows[row['pixel_sha256']]
        return np.array([float(r[n]) if r[n]!='' else np.nan for n in FEATURES],'float32')
    def roi(self,row):
        key=row['pixel_sha256'];entry=self.rows[key];path=self.root/entry['roi_path']
        if key not in self.checked_rois:
            if entry.get('roi_sha256')!=file_hash(path):
                raise ValueError('Anatomy ROI changed or cache is outdated; rerun extract-anatomy')
            self.checked_rois.add(key)
        return str(path)

def fit_normalizer(table,rows):
    # One contribution per decoded image to prevent duplicate-heavy features
    # from dominating the fitted scaling statistics.
    unique={r['pixel_sha256']:r for r in rows}
    a=np.array([table.vector(r) for r in unique.values()]);median=[];scale=[]
    for column in a.T:
        values=column[np.isfinite(column)];m=float(np.median(values)) if len(values) else 0.
        median.append(m);scale.append(max(float(np.std(values)),1e-3) if len(values) else 1.)
    return {'median':median,'scale':scale,'features':FEATURES}

def normalized(vector,stats):
    missing=~np.isfinite(vector);x=np.where(missing,np.array(stats['median']),vector)
    x=np.clip((x-stats['median'])/stats['scale'],-10,10)
    return np.concatenate([x,missing.astype('float32')]).astype('float32')
