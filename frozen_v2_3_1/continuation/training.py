from pathlib import Path
from collections import Counter
import json,math,random
from .common import config,runtime,dump,file_hash,role_rows,validate_manifest,CLASSES
from .features import FeatureTable,fit_normalizer
from .data import make_data
from .models import build_classifier,unfreeze,compile_classifier

def train(root,out,c,feature_root=None,smoke=False,resume=False):
    tf=runtime(c);out=Path(out)
    if any((out/name).exists() for name in ['config.json','best.weights.h5','progress.json']) and not resume:
        raise ValueError('Choose a new run directory or --resume; run artifacts exist')
    out.mkdir(parents=True,exist_ok=True);audit=validate_manifest(root)
    roles={role:role_rows(root,role) for role in ['fit','select','calibrate']}
    for role,rows in roles.items():
        if set(r['class_name'] for r in rows)!=set(CLASSES) and not smoke:raise ValueError('Missing class in '+role)
    needs=c['mode'] in ['roi','localized','fusion']
    table=FeatureTable(feature_root) if needs else None
    if table:
        provenance=json.loads((Path(feature_root)/'provenance.json').read_text())
        if provenance['classifier_manifest_sha256']!=audit['manifest_sha256']:raise ValueError('Feature cache belongs to a different dataset')
    if resume and (out/'config.json').exists():
        old=json.loads((out/'data.json').read_text())
        if json.loads((out/'config.json').read_text())!=c or old['manifest_sha256']!=audit['manifest_sha256']:
            raise ValueError('Resume config or dataset mismatch')
        if table and old['feature_provenance']!=provenance:raise ValueError('Resume anatomy cache mismatch')
    norm=fit_normalizer(table,roles['fit']) if table else None
    if norm:dump(out/'normalizer.json',norm)
    domains={s:i for i,s in enumerate(sorted({r['source'] for r in roles['fit']}))}
    dump(out/'config.json',c);dump(out/'data.json',{'manifest_sha256':audit['manifest_sha256'],
        'feature_root':str(Path(feature_root).resolve()) if feature_root else None,
        'feature_provenance':provenance if table else None,'domains':domains,
        'role_counts':{k:len(v) for k,v in roles.items()},
        'selection_and_calibration_disjoint':True,'scope':'Image-group protocol; no test cases used to select checkpoints.'})
    train_ds,rows=make_data(root,'fit',c,table,norm,domains,limit=8 if smoke else None)
    val_ds,_=make_data(root,'select',c,table,norm,domains,limit=8 if smoke else None)
    model,encoder=build_classifier({**c,'weights':None} if resume else c,len(domains))
    monitor='val_class_balanced_accuracy' if c['domain_weight']>0 else 'val_balanced_accuracy'
    best=-1.;histories=[]
    if resume and (out/'progress.json').exists():
        progress=json.loads((out/'progress.json').read_text())
        if progress['smoke_only']!=smoke:raise ValueError('Cannot resume smoke run as a full experiment')
        best=progress['best_score'];histories=progress['stages']
        model.load_weights(out/'best.weights.h5')
    stages=[('head',0.,c['head_epochs'],c['head_lr']),
            ('fine',c['unfreeze_fraction'],c['fine_epochs'],c['fine_lr']),
            ('full',1.,c['full_epochs'],c['full_lr'])]
    for name,fraction,epochs,lr in stages:
        if name in {h['stage'] for h in histories}:continue
        if not epochs:continue
        stage_path=out/(name+'.weights.h5')
        if resume and stage_path.exists():model.load_weights(stage_path)
        unfreeze(encoder,fraction)
        compile_classifier(model,c,lr,math.ceil(len(rows)/c['batch_size'])*epochs)
        # An interrupted stage already has a valid candidate. Its checkpoint must
        # survive a resumed epoch whose selection score is worse.
        prior_score=float(model.evaluate(val_ds,return_dict=True,verbose=0)[monitor.removeprefix('val_')]) if resume and stage_path.exists() else None
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(stage_path),monitor=monitor,mode='max',save_best_only=True,save_weights_only=True,
                initial_value_threshold=prior_score),
            tf.keras.callbacks.EarlyStopping(monitor=monitor,mode='max',patience=c['patience'],restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN(),tf.keras.callbacks.CSVLogger(str(out/(name+'_history.csv')),append=resume)]
        history=model.fit(train_ds,validation_data=val_ds,epochs=1 if smoke else epochs,callbacks=callbacks,verbose=2,shuffle=False)
        if not stage_path.exists():raise RuntimeError('Training did not produce a valid checkpoint')
        model.load_weights(stage_path)
        score=max(history.history[monitor]+([prior_score] if prior_score is not None else []));histories.append({'stage':name,'score':score})
        if score>best:
            best=score;model.save_weights(out/'best.weights.h5')
        # Always continue fine-tuning from the best state so far.
        model.load_weights(out/'best.weights.h5')
        dump(out/'progress.json',{'best_score':best,'stages':histories,'smoke_only':smoke,
            'resume_scope':'Best weights; optimizer restarts and any interrupted stage restarts its epoch budget.'})
    dump(out/'selection.json',{'best_selection_balanced_accuracy':best,'stages':histories,
        'smoke_only':smoke,'checkpoint_sha256':file_hash(out/'best.weights.h5')})
    return model

def search(root,out,c,feature_root,trials=4):
    rng=random.Random(c['seed']);results=[]
    for i in range(trials):
        trial=dict(c)
        if i:
            trial.update(backbone=rng.choice(['efficientnetv2b0','convnexttiny']),
                dropout=rng.choice([.25,.35,.45]),fine_lr=rng.choice([1e-5,3e-5]),
                label_smoothing=rng.choice([0.,.03,.05]),unfreeze_fraction=rng.choice([.2,.4]))
            if c.get('augmentation',True):
                trial.update(augment_gain=rng.choice([.08,.15]),augment_contrast=rng.choice([.06,.10]),
                    augment_gamma=rng.choice([.05,.10]),augment_speckle=rng.choice([0.,.015,.025]))
        path=Path(out)/('trial_'+str(i))
        if (path/'selection.json').exists():
            previous=json.loads((path/'data.json').read_text())
            if previous['manifest_sha256']!=file_hash(Path(root)/'metadata/dataset_manifest.csv') or json.loads((path/'config.json').read_text())!=trial:
                raise ValueError('Existing search trial has different data/config')
            if json.loads((path/'selection.json').read_text())['smoke_only']:
                raise ValueError('Smoke runs cannot be reused as completed search trials')
            if trial['mode'] in ['roi','localized','fusion']:
                FeatureTable(feature_root)
                if json.loads((Path(feature_root)/'provenance.json').read_text())!=previous['feature_provenance']:
                    raise ValueError('Existing search trial has different anatomy provenance')
        else:train(root,path,trial,feature_root,resume=(path/'config.json').exists())
        score=json.loads((path/'selection.json').read_text())['best_selection_balanced_accuracy']
        results.append({'run':str(path),'score':score,'config':trial})
        dump(Path(out)/'search_results.json',{'trials':results,'best':max(results,key=lambda r:r['score']),
            'selection_scope':'Model-selection validation subset only. Calibration and test are unused.'})
