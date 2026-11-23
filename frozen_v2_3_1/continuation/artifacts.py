"""Bind confidence calibration to the complete saved prediction recipe."""
import json
from pathlib import Path
from .common import CLASSES,file_hash

INFERENCE_RECIPE = 'continuation-v3-tensorflow-letterbox'

def inference_signature(run):
    run=Path(run);c=json.loads((run/'config.json').read_text())
    return {'recipe':INFERENCE_RECIPE,'classes':CLASSES,
        'checkpoint_sha256':file_hash(run/'best.weights.h5'),
        'config_sha256':file_hash(run/'config.json'),
        'data_sha256':file_hash(run/'data.json'),
        'normalizer_sha256':file_hash(run/'normalizer.json') if c['mode']=='fusion' else None}

def validate_calibration(calibration,runs,tta,manifest=None):
    if not runs:raise ValueError('At least one classifier run is required')
    signatures=[inference_signature(run) for run in runs]
    if [s['checkpoint_sha256'] for s in signatures]!=calibration['checkpoints']:
        raise ValueError('Calibration/checkpoint ensemble mismatch')
    if calibration.get('inference_signatures')!=signatures or bool(tta)!=calibration['tta']:
        raise ValueError('Calibration prediction recipe changed or is outdated; rerun calibration before evaluation/prediction')
    if manifest is not None and calibration['manifest_sha256']!=manifest:
        raise ValueError('Calibration belongs to a different manifest')
    return signatures
