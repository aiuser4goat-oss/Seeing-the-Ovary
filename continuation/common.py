from pathlib import Path
import csv, hashlib, json, os, random
import numpy as np

CLASSES = ['Benign', 'Malignant', 'Normal', 'PCOS']

def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def config(path=None, overrides=None):
    c = dict(seed=2627, image_size=224, batch_size=16, backbone='efficientnetv2b0',
        mode='fusion', weights='imagenet', head_epochs=8, fine_epochs=20, full_epochs=8,
        head_lr=0.0005, fine_lr=0.00003, full_lr=0.000005, unfreeze_fraction=0.3,
        dropout=0.35, weight_decay=0.0001, label_smoothing=0.03, patience=6,
        class_balance=True, duplicate_weighting=True, focal_gamma=0.0, mixed_precision=False,
        domain_weight=0.0, data_threads=2, seg_size=256, seg_filters=24,
        seg_epochs=40, seg_lr=0.0003, min_follicle_area=5, min_peak_distance=3,
        ovary_threshold=0.5, follicle_threshold=0.5, morphology_ablation='measured',
        augmentation=True,augment_gain=.15,augment_contrast=.10,augment_gamma=.10,augment_speckle=.025)
    if path: c.update(json.loads(Path(path).read_text()))
    if overrides: c.update(overrides)
    return c

def runtime(c):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(c['seed'])
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError: pass
    gpu = tf.config.list_physical_devices('GPU')
    for device in gpu:
        try: tf.config.experimental.set_memory_growth(device, True)
        except (ValueError, RuntimeError): pass
    tf.keras.mixed_precision.set_global_policy('mixed_float16' if gpu and c['mixed_precision'] else 'float32')
    return tf

def evaluation_role(row):
    if row['training_role']!='val':return row['training_role']
    return 'calibrate' if int(hashlib.sha256(row['image_group'].encode()).hexdigest()[:8],16)%3==0 else 'select'

def role_rows(root, role):
    rows = read_csv(Path(root)/'metadata/dataset_manifest.csv')
    if role in ['select', 'calibrate']:
        rows = [r for r in rows if r['training_role']=='val']
        # Group-level reservation: calibration never selects model checkpoints.
        rows = [r for r in rows if evaluation_role(r)==role]
    else: rows = [r for r in rows if r['training_role']==role]
    if not rows: raise ValueError('No records for role '+role)
    return rows

def validate_manifest(root):
    rows = read_csv(Path(root)/'metadata/dataset_manifest.csv')
    groups = {}; pixels = {}
    for r in rows:
        if r['class_name'] not in CLASSES: raise ValueError('Unknown class')
        if r['training_role'] not in ['fit','val','test']:raise ValueError('Unknown training role')
        if not (Path(root)/r['relative_path']).is_file(): raise FileNotFoundError(r['relative_path'])
        for index,key in [(groups,'image_group'), (pixels,'pixel_sha256')]:
            index.setdefault(r[key],set()).add(evaluation_role(r))
    if any(len(v)>1 for index in [groups,pixels] for v in index.values()):
        raise ValueError('Detected group/image overlaps across training roles')
    return {'images':len(rows),'manifest_sha256':file_hash(Path(root)/'metadata/dataset_manifest.csv')}
