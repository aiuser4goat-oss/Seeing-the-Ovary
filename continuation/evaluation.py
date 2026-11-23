from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
from .common import CLASSES,dump,write_csv,file_hash,role_rows,runtime
from .features import FeatureTable
from .data import make_data
from .models import build_classifier
from .artifacts import inference_signature,validate_calibration

def apply_temperature(probabilities,temperature=1.):
    return softmax(np.log(np.clip(probabilities,1e-8,1))/temperature,axis=1)

def nll(y,p):return float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-8,1))))

def metrics(y,p):
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=np.arange(4));result={
        'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=np.arange(4),average='macro',zero_division=0)),
        'nll':nll(y,p),'brier':float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1))),
        'confusion_matrix':cm.tolist()}
    confidence=p.max(1);correct=pred==y;ece=0.
    bins=np.minimum((confidence*10).astype(int),9)
    for i in range(10):
        mask=bins==i
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    result['ece_10_bins']=float(ece)
    # Four-class macro AUC is undefined when a source subset lacks a class.
    # scikit-learn 1.6+ can return NaN rather than raise in that situation.
    result['macro_roc_auc_ovr']=None
    if len(np.unique(y))==len(CLASSES):
        try:result['macro_roc_auc_ovr']=float(roc_auc_score(y,p,labels=np.arange(4),multi_class='ovr',average='macro'))
        except ValueError:pass
    result['per_class']={}
    for i,name in enumerate(CLASSES):
        tp=cm[i,i];fn=cm[i].sum()-tp;fp=cm[:,i].sum()-tp;tn=cm.sum()-tp-fn-fp
        result['per_class'][name]={'sensitivity':float(tp/max(tp+fn,1)),
            'specificity':float(tn/max(tn+fp,1)),'precision':float(tp/max(tp+fp,1)),'support':int(cm[i].sum())}
    return result

def predict_runs(root,runs,role,feature_root=None,tta=False,limit=None):
    if not runs:raise ValueError('At least one classifier run is required')
    all_probs=[];checkpoints=[];manifest=file_hash(Path(root)/'metadata/dataset_manifest.csv')
    for run in runs:
        run=Path(run);c=json.loads((run/'config.json').read_text());runtime(c)
        meta=json.loads((run/'data.json').read_text())
        if manifest!=meta['manifest_sha256']:raise ValueError('Evaluation dataset differs from training dataset')
        if json.loads((run/'selection.json').read_text())['smoke_only'] and role=='test':
            raise ValueError('A smoke checkpoint is not eligible for final test reporting')
        table=None;norm=None
        if c['mode'] in ['roi','localized','fusion']:
            path=feature_root or meta['feature_root'];table=FeatureTable(path)
            norm=json.loads((run/'normalizer.json').read_text())
            cache=json.loads((Path(path)/'provenance.json').read_text())
            if cache!=meta['feature_provenance']:raise ValueError('Anatomy feature provenance changed since fitting')
        ds,rows=make_data(root,role,c,table,norm,meta['domains'],limit)
        model,_=build_classifier({**c,'weights':None},len(meta['domains']))
        model.load_weights(run/'best.weights.h5');batches=[]
        for x,_,_ in ds:
            views=[x]
            if tta:
                import tensorflow as tf
                for axes in [[2],[1],[1,2]]:
                    views.append({key:tf.reverse(value,axis=axes) if key!='morphology' else value for key,value in x.items()})
            predictions=[]
            for view in views:
                p=model(view,training=False)
                if isinstance(p,dict):p=p['class']
                predictions.append(p.numpy())
            batches.append(np.mean(predictions,axis=0))
        all_probs.append(np.concatenate(batches));checkpoints.append(file_hash(run/'best.weights.h5'))
    return rows,np.mean(all_probs,axis=0),np.var(all_probs,axis=0).mean(1),checkpoints

def calibrate(root,runs,out,feature_root=None,tta=False,target_accuracy=.95):
    if not 0<target_accuracy<=1:raise ValueError('Target accuracy must be in (0,1]')
    rows,p,variance,checkpoints=predict_runs(root,runs,'calibrate',feature_root,tta)
    y=np.array([CLASSES.index(r['class_name']) for r in rows])
    fit=minimize_scalar(lambda log_t:nll(y,apply_temperature(p,np.exp(log_t))),bounds=(-3.,3.),method='bounded')
    temperature=float(np.exp(fit.x));p=apply_temperature(p,temperature)
    candidates=[]
    for threshold in np.linspace(.25,.99,150):
        keep=p.max(1)>=threshold
        if keep.sum()>=30 and np.mean(p[keep].argmax(1)==y[keep])>=target_accuracy:
            candidates.append((float(keep.mean()),float(threshold)))
    threshold=max(candidates)[1] if candidates else None
    result={'temperature':temperature,'referral_confidence_threshold':threshold,
        'target_calibration_accuracy':target_accuracy,'calibration_cases':len(rows),
        'checkpoints':checkpoints,'inference_signatures':[inference_signature(run) for run in runs],
        'tta':tta,'manifest_sha256':file_hash(Path(root)/'metadata/dataset_manifest.csv'),
        'calibration_metrics':metrics(y,p),'scope':'Validation-derived research referral rule; no clinical reliability guarantee.'}
    dump(Path(out)/'calibration.json',result);return result

def evaluate(root,runs,out,calibration_path,feature_root=None,tta=False,bootstrap=300):
    calibration=json.loads(Path(calibration_path).read_text())
    validate_calibration(calibration,runs,tta,file_hash(Path(root)/'metadata/dataset_manifest.csv'))
    rows,p,variance,checkpoints=predict_runs(root,runs,'test',feature_root,tta)
    if checkpoints!=calibration['checkpoints'] or tta!=calibration['tta']:
        raise ValueError('Calibration does not match checkpoint ensemble/TTA')
    if calibration['manifest_sha256']!=file_hash(Path(root)/'metadata/dataset_manifest.csv'):
        raise ValueError('Calibration belongs to a different manifest')
    y=np.array([CLASSES.index(r['class_name']) for r in rows]);p=apply_temperature(p,calibration['temperature'])
    result=metrics(y,p);threshold=calibration['referral_confidence_threshold']
    keep=p.max(1)>=threshold if threshold is not None else np.zeros(len(p),bool)
    result['retained_coverage']=float(keep.mean())
    result['retained_accuracy']=float(np.mean(p[keep].argmax(1)==y[keep])) if keep.any() else None
    result['test_manifest_sha256']=calibration['manifest_sha256']
    # Resample detected image groups, preserving all supplied file-label records.
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['image_group']].append(i)
    group_indices=list(groups.values());rng=np.random.default_rng(2627);scores=[]
    for _ in range(bootstrap):
        ids=np.concatenate([group_indices[i] for i in rng.integers(0,len(group_indices),len(group_indices))])
        if len(np.unique(y[ids]))==4:
            scores.append([balanced_accuracy_score(y[ids],p[ids].argmax(1)),f1_score(y[ids],p[ids].argmax(1),average='macro',zero_division=0)])
    if scores:
        ci=np.percentile(scores,[2.5,97.5],axis=0)
        result['group_bootstrap_95_ci']={name:ci[:,i].tolist() for i,name in enumerate(['balanced_accuracy','macro_f1'])}
    result['source_metrics']={s:metrics(y[idx],p[idx]) for s in sorted({r['source'] for r in rows})
        if len(idx:=[i for i,r in enumerate(rows) if r['source']==s])}
    dump(Path(out)/'test_metrics.json',result)
    predictions=[]
    for i,r in enumerate(rows):
        predictions.append({'record_id':r['record_id'],'relative_path':r['relative_path'],
            'true_class':r['class_name'],'predicted_class':CLASSES[int(p[i].argmax())],
            'confidence':float(p[i].max()),'entropy':float(-np.sum(p[i]*np.log(np.clip(p[i],1e-8,1)))/np.log(4)),
            'ensemble_variance':float(variance[i]),'referred':bool(not keep[i]),
            **{n:float(p[i,j]) for j,n in enumerate(CLASSES)}})
    write_csv(Path(out)/'test_predictions.csv',predictions)
    curve=[]
    for threshold in np.linspace(0,1,101):
        retained=p.max(1)>=threshold
        curve.append({'threshold':threshold,'coverage':float(retained.mean()),
            'error_rate':float(np.mean(p[retained].argmax(1)!=y[retained])) if retained.any() else ''})
    write_csv(Path(out)/'risk_coverage.csv',curve)
    return result
