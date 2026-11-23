#!/usr/bin/env python3
"""Finish predeclared controls, freeze artifacts, evaluate once, and archive reports.
Runs only against the existing fixed classification manifest; it never changes data.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

PLAN_VERSION = '2026-11-21-controls-v2-refined-anatomy'
EXPERIMENTS = [
    ('original_baseline', 'original_baseline.json', 'Original hybrid classifier'),
    ('fusion', 'recommended.json', 'Global image + ovarian ROI + measured morphology'),
    ('fusion_refined', 'recommended.json', 'Global image + ovarian ROI + refined measured morphology'),
    ('image_only', 'image_only.json', 'New encoder, global image only'),
    ('localized_no_morphology', 'localized_no_morphology.json', 'Global image + ovarian ROI'),
    ('fusion_zero_morphology', 'fusion_zero_morphology_control.json', 'Fusion architecture, zero morphology inputs'),
]
PRIMARY = ['fusion_refined', 'original_baseline']

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def paired_group_bootstrap(rows, predictions, classes, comparisons, repetitions=1000):
    """Paired cluster bootstrap, keeping every file-label record in each image group."""
    import numpy as np
    groups = sorted({r['image_group'] for r in rows})
    group_index = {g: i for i, g in enumerate(groups)}
    y = np.array([classes.index(r['class_name']) for r in rows])
    g = np.array([group_index[r['image_group']] for r in rows])
    cms = {}
    for name, pred in predictions.items():
        cm = np.zeros((len(groups), len(classes), len(classes)), dtype=np.int64)
        np.add.at(cm, (g, y, np.asarray(pred, dtype=int)), 1)
        cms[name] = cm
    def scores(cm):
        tp = np.diag(cm)
        support = cm.sum(axis=1)
        if (support == 0).any():
            return None
        ba = np.mean(tp / support)
        den = support + cm.sum(axis=0)
        f1 = np.mean(np.divide(2 * tp, den, out=np.zeros(len(classes)), where=den > 0))
        return np.array([ba, f1])
    draws = {tuple(pair): [] for pair in comparisons}
    rng = np.random.default_rng(2627)
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        s = {name: scores(np.einsum('g,gij->ij', weights, cm)) for name, cm in cms.items()}
        for a, b in comparisons:
            if s[a] is not None and s[b] is not None:
                draws[(a, b)].append(s[a] - s[b])
    result = []
    for a, b in comparisons:
        point = scores(cms[a].sum(0)) - scores(cms[b].sum(0))
        d = np.asarray(draws[(a, b)])
        bounds = np.percentile(d, [2.5, 97.5], axis=0)
        result.append({'model': a, 'reference': b, 'image_groups': len(groups),
            'valid_bootstrap_repetitions': len(d),
            'balanced_accuracy_difference': float(point[0]),
            'balanced_accuracy_difference_95_ci': bounds[:, 0].tolist(),
            'macro_f1_difference': float(point[1]),
            'macro_f1_difference_95_ci': bounds[:, 1].tolist(),
            'interpretation': 'Exploratory paired image-group bootstrap; no multiplicity adjustment or patient-level claim.'})
    return result

def make_reports(results, root, classes):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    report = results / 'final_reports'
    report.mkdir(exist_ok=True)
    models = [e[0] for e in EXPERIMENTS]
    rows_by_id = {r['record_id']: r for r in read_csv(root / 'metadata/dataset_manifest.csv') if r['training_role'] == 'test'}
    metrics, predictions, raw = {}, {}, {}
    ids = None
    summary = []
    for name, _, label in EXPERIMENTS:
        metrics[name] = m = read(results / name / 'evaluation/test_metrics.json')
        table = read_csv(results / name / 'evaluation/test_predictions.csv')
        row_ids = [r['record_id'] for r in table]
        if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(rows_by_id):
            raise ValueError('Incomplete or duplicate evaluation record IDs: ' + name)
        if ids is None:
            ids = row_ids
        lookup = {r['record_id']: r for r in table}
        raw[name] = table = [lookup[i] for i in ids]
        if any(r['true_class'] != rows_by_id[r['record_id']]['class_name'] for r in table):
            raise ValueError('Evaluation labels differ from frozen manifest')
        predictions[name] = np.array([classes.index(r['predicted_class']) for r in table])
        sel = read(results / name / 'selection.json')
        cal = read(results / name / 'calibration/calibration.json')
        summary.append({'model': name, 'description': label,
            'selection_balanced_accuracy': sel['best_selection_balanced_accuracy'],
            **{k: m.get(k) for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr', 'nll', 'brier', 'ece_10_bins', 'retained_coverage', 'retained_accuracy']},
            'temperature': cal['temperature'], 'referral_threshold': cal['referral_confidence_threshold'],
            'test_records': len(table)})
    with (report / 'model_comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    aligned = [rows_by_id[i] for i in ids]
    comparisons = [('fusion_refined', 'original_baseline'), ('fusion_refined', 'fusion'), ('fusion_refined', 'fusion_zero_morphology'), ('fusion', 'original_baseline'), ('fusion', 'image_only'),
        ('localized_no_morphology', 'image_only'), ('fusion', 'localized_no_morphology'),
        ('fusion', 'fusion_zero_morphology')]
    paired = paired_group_bootstrap(aligned, predictions, classes, comparisons)
    save(report / 'paired_comparisons.json', paired)
    save(report / 'all_metrics.json', metrics)
    plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 180, 'font.size': 10})
    labels = ['Original', 'Default fusion', 'Refined fusion', 'Image only', 'Image + ROI', 'Zero morphology']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ['balanced_accuracy', 'macro_f1'], ['Balanced accuracy', 'Macro F1']):
        values = np.array([metrics[n][key] for n in models])
        ci = np.array([metrics[n]['group_bootstrap_95_ci'][key] for n in models])
        ax.bar(np.arange(len(models)), values * 100, color=['#6c757d', '#6c96ac', '#20709b', '#6ca4b6', '#54a083', '#b29462'])
        ax.errorbar(np.arange(len(models)), values * 100,
            yerr=np.maximum(0, np.vstack([values - ci[:, 0], ci[:, 1] - values])) * 100,
            fmt='none', color='black', capsize=4)
        ax.set_xticks(np.arange(len(models)), labels, rotation=20, ha='right')
        ax.set(title=title, ylabel='Percent', ylim=(0, 100))
        for i, v in enumerate(values):
            ax.text(i, min(v * 100 + 3, 98), f'{v*100:.2f}', ha='center', fontsize=8)
    fig.suptitle('Fixed held-out test set; 95% image-group bootstrap intervals')
    fig.tight_layout()
    fig.savefig(report / 'classification_comparison.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ['original_baseline', 'fusion_refined']):
        cm = np.asarray(metrics[name]['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=classes, yticklabels=classes,
               title=name.replace('_', ' ').title(), xlabel='Predicted label', ylabel='Provided label')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    fig.tight_layout()
    fig.savefig(report / 'confusion_matrices.png')
    plt.close(fig)
    y = np.array([classes.index(r['class_name']) for r in aligned])
    for name in ['original_baseline', 'fusion_refined']:
        p = np.array([[float(r[c]) for c in classes] for r in raw[name]])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, label in enumerate(classes):
            truth = (y == i).astype(int)
            fpr, tpr, _ = roc_curve(truth, p[:, i])
            precision, recall, _ = precision_recall_curve(truth, p[:, i])
            axes[0].plot(fpr, tpr, label=f'{label} AUC={auc(fpr,tpr):.3f}')
            axes[1].plot(recall, precision, label=f'{label} AP={average_precision_score(truth,p[:,i]):.3f}')
        axes[0].plot([0, 1], [0, 1], 'k:', alpha=.4)
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='One-versus-rest ROC')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='One-versus-rest precision–recall')
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
        fig.suptitle(name.replace('_', ' ').title())
        fig.tight_layout()
        fig.savefig(report / (name + '_roc_pr.png'))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name in models:
        p = np.array([[float(r[c]) for c in classes] for r in raw[name]])
        confidence = p.max(1)
        correct = p.argmax(1) == y
        bins = np.minimum((confidence*10).astype(int), 9)
        x, z = [], []
        for k in range(10):
            mask = bins == k
            if mask.any():
                x.append(float(confidence[mask].mean()))
                z.append(float(correct[mask].mean()))
        axes[0].plot(x, z, 'o-', label=name, markersize=3)
        curve = read_csv(results / name / 'evaluation/risk_coverage.csv')
        v = sorted([(float(r['coverage']),float(r['error_rate'])) for r in curve if r['error_rate'] != ''])
        axes[1].plot([a for a,b in v], [b for a,b in v], label=name)
    axes[0].plot([0,1],[0,1],'k:',alpha=.5)
    axes[0].set(xlabel='Mean calibrated confidence', ylabel='Observed accuracy', title='Reliability: fixed ten bins', xlim=(0,1), ylim=(0,1))
    axes[1].set(xlabel='Fraction retained', ylabel='Error rate among retained', title='Descriptive risk–coverage curves', xlim=(0,1), ylim=(0,1))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(report / 'calibration_and_referral.png')
    plt.close(fig)
    primary = paired[0]
    text = [
        'OVARIAN ULTRASOUND CONTINUATION: FINAL COMPUTATIONAL RESULTS',
        'Generated: ' + now(), '',
        'Primary comparison: full anatomy-guided fusion with validation-selected follicle postprocessing versus the original hybrid rerun on the same held-out images.',
        'All model recipes were declared before final test evaluation. Checkpoints were selected on selection validation only; temperature and referral thresholds used a separate calibration subset.',
        'All 19,942 supplied classification files, including all 16,729 older files, remain in the dataset. The fixed image-group split contains 12,763 fitting, 2,112 selection, 1,079 calibration, and 3,988 final-test records.',
        'Provided labels and duplicate file-label records are preserved. Image groups, rather than individual duplicate files, are resampled for confidence intervals. No patient independence is claimed.', '',
    ]
    for s in summary:
        text.append(f"{s['model']}: test accuracy {s['accuracy']*100:.2f}%; balanced accuracy {s['balanced_accuracy']*100:.2f}%; macro F1 {s['macro_f1']:.4f}; macro ROC AUC {s['macro_roc_auc_ovr']:.4f}.")
        text.append('95% group-bootstrap balanced-accuracy interval: ' + str(metrics[s['model']]['group_bootstrap_95_ci']['balanced_accuracy']))
    text += ['', 'Primary balanced-accuracy difference, percentage points: ' + str(primary['balanced_accuracy_difference']*100),
        'Paired 95% interval, percentage points: ' + str([v*100 for v in primary['balanced_accuracy_difference_95_ci']]),
        '', 'INTERPRETATION AND LIMITATIONS',
        'A higher test metric supports improvement only on this fixed research dataset. An interval crossing zero does not establish a clear improvement. Secondary control comparisons are exploratory.',
        'The historical reported 91.25% balanced accuracy is a reference from the previous project, not a controlled head-to-head test on the current dataset. The rerun baseline supplies the matched comparison.',
        'Single-seed experiments do not quantify training-seed variability. Repeated independent training and independent external data would strengthen conclusions.',
        'Source-stratified test results describe the included sources; they do not establish unseen-scanner or external-source generalization. Domain-invariant training is not part of this completed comparison.',
        'No measurements are clinical diagnoses. Follicle features represent visible 2D structures and relative image dimensions; there is no verified physical millimeter calibration or full-ovary 3D count.',
        'Original-postprocessing anatomy validation: mean ovary Dice 0.8101 and follicle Dice 0.8264. These semantic overlaps do not demonstrate accurate individual follicle identification. Instance F1 at IoU 0.5 was only 0.0742 and count MAE was 6.35 on the instance-scored subset; these results motivated a separate validation-only postprocessing study. Refined anatomical measurements must be interpreted using the supplementary refined-anatomy validation/test results, when available.',
        'Entropy and calibrated confidence describe prediction confidence. Single-model ensemble variance is zero and is not epistemic uncertainty. A validation-derived referral threshold is not a clinical guarantee; test coverage and retained accuracy are reported without retuning the threshold.',
        'Risk–coverage curves are descriptive. The deployed research referral rule remains the threshold fixed on calibration data.',
        'Attribution maps indicate model sensitivity and do not establish correct anatomy or causal reasoning.',
        '', 'FILES',
        'model_comparison.csv; all_metrics.json; paired_comparisons.json; confusion_matrices.png; classification_comparison.png; ROC/PR plots; calibration_and_referral.png.',
        'Each model directory includes its fixed calibration rule, test_metrics.json, record-level test_predictions.csv, and risk_coverage.csv.',
        'experiment_plan.json and experiment_freeze.json record recipes, data hashes, checkpoint hashes, and calibration hashes before final evaluation.'
    ]
    (report / 'RESULTS_AND_LIMITATIONS.txt').write_text('\n'.join(text) + '\n')
    return report

def main():
    p = argparse.ArgumentParser()
    for key in ['code', 'classification', 'features', 'results']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--seg-run', type=Path, required=True)
    p.add_argument('--refinement', type=Path, required=True)
    a = p.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    lock = (a.results / 'finish_experiments.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(a.code))
    from continuation.common import config, CLASSES
    from continuation.artifacts import inference_signature, validate_calibration
    manifest = sha(a.classification / 'metadata/dataset_manifest.csv')
    source_hashes = {str(f.relative_to(a.code)):sha(f) for folder in ['continuation','model'] for f in sorted((a.code/folder).rglob('*.py'))}
    refinement = read(a.refinement)
    if refinement['selected']['parameters']['method'] not in ['watershed','connected_components']:
        raise ValueError('Unknown validation-selected postprocessing method')
    plan = {'version': PLAN_VERSION, 'controller_sha256':sha(Path(__file__)), 'primary_comparison': PRIMARY,
        'manifest_sha256': manifest, 'source_hashes': source_hashes,
        'refinement_sha256':sha(a.refinement), 'refinement_selection':refinement,
        'segmenter_checkpoint_sha256':sha(a.seg_run/'best.weights.h5'),
        'experiments': [{'name':name, 'description':description, 'config':config(a.code/'configs/continuation'/cfg),'anatomy_variant':'refined' if name=='fusion_refined' else 'original'} for name,cfg,description in EXPERIMENTS],
        'inference_tta':False, 'target_calibration_accuracy':.95, 'bootstrap_repetitions':1000,
        'test_policy':'Fixed recipes, selection by selection validation only, separate calibration, no test-based tuning.',
        'existing_runs_note':'Original baseline and default fusion were trained with version 2.2.1; default anatomy behavior and classifier training code are preserved.',
        'external_generalization':'Not established by this protocol.', 'seed_repeats':1}
    plan_path = a.results / 'experiment_plan.json'
    if plan_path.exists():
        if read(plan_path)['plan'] != plan:
            raise ValueError('Existing experiment plan differs. Refusing to alter a frozen comparison.')
    else:
        if any(a.results.glob('*/evaluation/test_metrics.json')) or any((a.results/n/'test_metrics.json').exists() for n in ['fusion_final_test','baseline_final_test']):
            raise ValueError('Final test results already exist without this predeclared plan')
        save(plan_path, {'created_utc':now(), 'plan':plan})
    status = {'started_utc':now(), 'plan_sha256':sha(plan_path)}
    def stage(name, **detail):
        status.update(stage=name, updated_utc=now(), **detail)
        save(a.results/'completion_status.json', status)
        print(json.dumps(status), flush=True)
    def run(*args):
        cmd = [sys.executable, '-u', str(a.code/'run_continuation.py'), *map(str,args)]
        print('RUN:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=a.code, check=True, env={**os.environ, 'PYTHONUNBUFFERED':'1'})
    refined_features = a.features.parent/'features_refined'
    refined_segmenter = a.features.parent/'checkpoint_refined/segmentation'
    def features_for(name):
        return refined_features if name=='fusion_refined' else a.features
    stage('preparing_refined_anatomy', model='fusion_refined')
    refined_segmenter.mkdir(parents=True,exist_ok=True)
    parameters=refinement['selected']['parameters']
    segmenter_config=read(a.seg_run/'config.json')
    segmenter_config.update(ovary_threshold=parameters['ovary_threshold'],follicle_threshold=parameters['follicle_threshold'],
        min_follicle_area=parameters['min_area'],min_peak_distance=parameters['min_peak_distance'] or 3,
        follicle_instance_method=parameters['method'])
    save(refined_segmenter/'config.json',segmenter_config)
    for filename in ['best.weights.h5','training_data.json']:
        target=refined_segmenter/filename
        if not target.exists():
            shutil.copy2(a.seg_run/filename,target)
    if sha(refined_segmenter/'best.weights.h5')!=sha(a.seg_run/'best.weights.h5'):
        raise ValueError('Refined segmenter must reuse the original learned weights')
    refined_archive=a.results/'features_refined.zip'
    if not (refined_features/'provenance.json').exists() and refined_archive.exists():
        with zipfile.ZipFile(refined_archive) as z:
            destination=refined_features.resolve()
            if any(not (destination/n).resolve().is_relative_to(destination) for n in z.namelist()):
                raise ValueError('Unsafe refined cache archive')
            z.extractall(destination)
    if not (refined_features/'provenance.json').exists():
        run('extract-anatomy','--classification',a.classification,'--seg-run',refined_segmenter,'--out',refined_features)
        local_archive=shutil.make_archive(str(a.features.parent/'features_refined_cache'),'zip',refined_features)
        shutil.copy2(local_archive,refined_archive)
    feature_meta=read(refined_features/'provenance.json')
    if feature_meta['classifier_manifest_sha256']!=manifest or feature_meta['segmenter_config_sha256']!=sha(refined_segmenter/'config.json') or feature_meta['segmenter_checkpoint_sha256']!=sha(a.seg_run/'best.weights.h5'):
        raise ValueError('Refined anatomy cache provenance mismatch')
    freeze_path = a.results / 'experiment_freeze.json'
    try:
        for name,cfg,_ in EXPERIMENTS:
            stage('training_or_verifying', model=name)
            run_dir = a.results/name
            if not (run_dir/'selection.json').exists():
                if freeze_path.exists():
                    raise ValueError('Frozen experiment has lost a selected checkpoint')
                command = ['train','--classification',a.classification,'--features',features_for(name),
                           '--out',run_dir,'--config',a.code/'configs/continuation'/cfg]
                if (run_dir/'config.json').exists():
                    command.append('--resume')
                run(*command)
            if read(run_dir/'config.json') != config(a.code/'configs/continuation'/cfg):
                raise ValueError('Unexpected completed model config: '+name)
            selection = read(run_dir/'selection.json')
            if selection['smoke_only'] or selection['checkpoint_sha256'] != sha(run_dir/'best.weights.h5'):
                raise ValueError('Invalid checkpoint: '+name)
            if read(run_dir/'data.json')['manifest_sha256'] != manifest:
                raise ValueError('Wrong training dataset: '+name)
            stage('calibrating_or_verifying', model=name)
            cal = run_dir/'calibration/calibration.json'
            if not cal.exists():
                legacy_name={'fusion':'fusion_calibration','original_baseline':'baseline_calibration'}.get(name)
                legacy_cal=a.results/legacy_name/'calibration.json' if legacy_name else None
                if legacy_cal is not None and legacy_cal.exists():
                    validate_calibration(read(legacy_cal), [run_dir], False, manifest)
                    cal.parent.mkdir(parents=True,exist_ok=True)
                    cal.write_bytes(legacy_cal.read_bytes())
            if not cal.exists():
                if freeze_path.exists():
                    raise ValueError('Frozen experiment has lost calibration')
                run('calibrate','--classification',a.classification,'--features',features_for(name),
                    '--runs',run_dir,'--out',cal.parent)
            validate_calibration(read(cal), [run_dir], False, manifest)
        freeze = {'plan_sha256':sha(plan_path), 'manifest_sha256':manifest,
            'models':{name:{'signature':inference_signature(a.results/name),
                'selection_sha256':sha(a.results/name/'selection.json'),
                'calibration_sha256':sha(a.results/name/'calibration/calibration.json')} for name,_,_ in EXPERIMENTS}}
        if freeze_path.exists():
            if read(freeze_path)['freeze'] != freeze:
                raise ValueError('Frozen artifacts changed; refusing final evaluation')
        else:
            save(freeze_path, {'frozen_utc':now(), 'freeze':freeze})
        freeze_hash = sha(freeze_path)
        for name,_,_ in EXPERIMENTS:
            if sha(a.classification/'metadata/dataset_manifest.csv') != manifest:
                raise ValueError('Dataset changed after freeze')
            if inference_signature(a.results/name) != freeze['models'][name]['signature']:
                raise ValueError('Model changed after freeze')
            stage('final_test_evaluation', model=name, freeze_sha256=freeze_hash)
            out = a.results/name/'evaluation'
            receipt = out/'evaluation_receipt.json'
            outputs = ['test_metrics.json','test_predictions.csv','risk_coverage.csv']
            if receipt.exists():
                previous = read(receipt)
                if previous['freeze_sha256'] != freeze_hash or any(sha(out/f) != previous['files'][f] for f in outputs):
                    raise ValueError('Stored test artifacts do not match frozen experiment')
            else:
                run('evaluate','--classification',a.classification,'--features',features_for(name),
                    '--runs',a.results/name,'--out',out,'--calibration',a.results/name/'calibration/calibration.json',
                    '--bootstrap',1000)
                save(receipt, {'created_utc':now(), 'freeze_sha256':freeze_hash, 'files':{f:sha(out/f) for f in outputs}})
        stage('writing_reports', model='all')
        report = make_reports(a.results, a.classification, CLASSES)
        for f in [plan_path, freeze_path, Path(__file__)]:
            (report/f.name).write_bytes(f.read_bytes())
        for source in [a.code/'RESEARCH_SOURCES.txt', a.classification/'metadata/DATASET_INFO.json', a.classification/'metadata/all_sources_accounting.csv', a.results.parent/'Kaggle_Anatomy_Validation_Metrics.json',a.refinement,a.results.parent/'Refined_Anatomy_Test_Results.json']:
            if source.is_file():
                (report/source.name).write_bytes(source.read_bytes())
        packages = a.results/'model_packages'
        packages.mkdir(exist_ok=True)
        with zipfile.ZipFile(packages/'refined_anatomy.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in refined_segmenter.iterdir():
                if f.is_file():z.write(f,arcname='segmentation/'+f.name)
            z.write(a.refinement,arcname='Refined_Anatomy_Selection.json')
        for name,_,_ in EXPERIMENTS:
            with zipfile.ZipFile(packages/(name+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
                for file in ['config.json','data.json','best.weights.h5','selection.json','normalizer.json','calibration/calibration.json']:
                    f = a.results/name/file
                    if f.exists():
                        z.write(f, arcname=name+'/'+file)
                z.write(freeze_path, arcname='experiment_freeze.json')
        with zipfile.ZipFile(a.results/'Final_Reports.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(report.rglob('*')):
                if f.is_file():
                    z.write(f,arcname='final_reports/'+str(f.relative_to(report)))
            for name,_,_ in EXPERIMENTS:
                for f in sorted((a.results/name/'evaluation').glob('*')):
                    if f.is_file():
                        z.write(f, arcname=name+'/evaluation/'+f.name)
        save(a.results/'deliverable_hashes.json', {str(f.relative_to(a.results)):sha(f) for f in [a.results/'Final_Reports.zip',*sorted(packages.glob('*.zip'))]})
        stage('complete', model='all', reports=str(a.results/'Final_Reports.zip'), completed_utc=now())
        print((report/'RESULTS_AND_LIMITATIONS.txt').read_text(), flush=True)
    except BaseException as exc:
        stage('failed', error=repr(exc), traceback=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
