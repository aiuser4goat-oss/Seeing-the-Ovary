from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps
from .common import CLASSES,runtime,file_hash,dump
from .models import build_classifier
from .anatomy import build_segmenter
from .features import postprocess,normalized
from .morphology import FEATURES
from .evaluation import apply_temperature
from .artifacts import validate_calibration
from .data import resize_classifier_image,load_classifier_image

def predict(image_path,runs,calibration_path,out,seg_run=None,explain=False):
    calibration=json.loads(Path(calibration_path).read_text())
    validate_calibration(calibration,runs,calibration['tta'])
    image=Image.open(image_path).convert('RGB');crop=image;vector=None;ov=None;ins=None
    configs=[json.loads((Path(run)/'config.json').read_text()) for run in runs]
    if any(c['mode'] in ['roi','localized','fusion'] for c in configs):
        if seg_run is None:raise ValueError('An anatomy model --seg-run is required')
        seg_run=Path(seg_run);sc=json.loads((seg_run/'config.json').read_text());runtime(sc)
        signature=file_hash(seg_run/'best.weights.h5')
        for run,c in zip(runs,configs):
            if c['mode'] in ['roi','localized','fusion']:
                expected=json.loads((Path(run)/'data.json').read_text())['feature_provenance']
                if signature!=expected['segmenter_checkpoint_sha256'] or file_hash(seg_run/'config.json')!=expected['segmenter_config_sha256']:
                    raise ValueError('Inference anatomy model/config differs from classifier training')
        segmenter=build_segmenter(sc);segmenter.load_weights(seg_run/'best.weights.h5')
        gray=np.asarray(ImageOps.pad(image.convert('L'),(sc['seg_size'],sc['seg_size']),method=Image.Resampling.BILINEAR))[None,...,None]/255.
        p=segmenter(gray,training=False)
        crop,vector,ov,ins=postprocess(image,{k:v[0,...,0].numpy() for k,v in p.items()},sc)
    all_predictions=[];attributions=[]
    for run,c in zip(runs,configs):
        tf=runtime(c);run=Path(run);meta=json.loads((run/'data.json').read_text())
        model,_=build_classifier({**c,'weights':None},len(meta['domains']));model.load_weights(run/'best.weights.h5')
        def tensor(im):return resize_classifier_image(np.asarray(im),c['image_size'])[None,...]
        x={'image':tensor(crop) if c['mode']=='roi' else load_classifier_image(image_path,c['image_size'])[None,...]}
        if c['mode'] in ['localized','fusion']:x['roi']=tensor(crop)
        if c['mode']=='fusion':
            values=normalized(vector,json.loads((run/'normalizer.json').read_text()))
            if c['morphology_ablation']=='zeros':values=np.zeros_like(values)
            x['morphology']=tf.convert_to_tensor(values[None])
        predictions=[]
        axes_list=[None,[2],[1],[1,2]] if calibration['tta'] else [None]
        for axes in axes_list:
            view={k:tf.reverse(v,axis=axes) if axes and k!='morphology' else v for k,v in x.items()}
            p=model(view,training=False);p=p['class'] if isinstance(p,dict) else p;predictions.append(p.numpy())
        mean=np.mean(predictions,axis=0);all_predictions.append(mean)
        if explain:
            target=int(mean.argmax());gradients=[]
            for alpha in np.linspace(0,1,24):
                interpolated={**x,'image':x['image']*alpha}
                with tf.GradientTape() as tape:
                    tape.watch(interpolated['image']);prob=model(interpolated,training=False)
                    if isinstance(prob,dict):prob=prob['class']
                    score=prob[:,target]
                gradients.append(tape.gradient(score,interpolated['image']).numpy())
            heat=np.abs(np.mean(gradients,axis=0)*x['image'].numpy()).mean(-1)[0]
            attributions.append(heat/max(float(heat.max()),1e-8))
    p=apply_temperature(np.mean(all_predictions,axis=0),calibration['temperature'])[0]
    threshold=calibration['referral_confidence_threshold'];confidence=float(p.max())
    result={'predicted_class':CLASSES[int(p.argmax())],'probabilities':dict(zip(CLASSES,map(float,p))),
        'confidence':confidence,'referred':threshold is None or confidence<threshold,
        'referral_threshold':threshold,'entropy':float(-np.sum(p*np.log(np.clip(p,1e-8,1)))/np.log(4)),
        'measurements':{k:float(v) if np.isfinite(v) else None for k,v in zip(FEATURES,vector)} if vector is not None else None,
        'scope':'Research image prediction, not a clinical diagnosis. Referral rule is validation-derived.'}
    out=Path(out);dump(out/'prediction.json',result)
    if ins is not None:
        Image.fromarray(ins).save(out/'follicle_instances.png');Image.fromarray((ov*255).astype('uint8')).save(out/'ovary_mask.png')
        crop.save(out/'ovary_crop.png')
    for i,heat in enumerate(attributions):Image.fromarray((255*heat).astype('uint8')).save(out/f'image_attribution_{i}.png')
    if explain:dump(out/'attribution_scope.json',{'method':'24-step input integrated gradients approximation against black image baseline.',
        'scope':'Global image input only, ROI/morphology held fixed. Per-run predicted class; not a correctness or anatomical-localization validation.'})
    return result
