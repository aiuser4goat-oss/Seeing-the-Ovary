OVARIAN AI CONTINUATION CODEBASE

START HERE
This extends the original TensorFlow/Keras project. Original model modules and commands remain available. The new entry point is run_continuation.py; main.py continuation dispatches to the same CLI before legacy CPU-only settings. No old trained weights are automatically reused for the new evaluation.

CONTENTS
continuation/anatomy.py: USOVA3D and FollicleFinder preparation; multi-head 2D U-Net; binary Dice+BCE and genuine-instance distance supervision; semantic-only volumes do not receive invented instance targets.
continuation/features.py and morphology.py: predicted ovary crop, marker-based watershed instance separation, visible 2D count, normalized follicle size, area fraction, peripheral distribution, nearest-neighbor spacing and missingness flags.
continuation/models.py: EfficientNetV2-B0 default, optional V2-S/ConvNeXtTiny/EfficientNet-B0, old three-branch hybrid rerun, and image/localized/morphology ablations.
continuation/training.py: head-only training -> partial unfreezing -> low-rate full fine-tuning; frozen BatchNorm statistics; AdamW/cosine schedule; gradient clipping; class balancing; reduced weighting of repeat images; early stopping; checkpointing; resumable stages; validation-only search.
continuation/evaluation.py: separate validation calibration, temperature scaling, validation-derived referral threshold, optional flip TTA/equal-weight model ensembles, balanced accuracy/macro F1/AUC/confusion matrix, per-source metrics, group-bootstrap confidence intervals and risk-coverage CSV.
continuation/inference.py: calibrated single-image prediction, masks, crop, morphology and optional input-gradient attribution.

INSTALLATION
Use Python 3.10-3.13. The tested local runtime is Python 3.13, TensorFlow 2.20 and Keras 3.15 (CPU).
pip install -r requirements-continuation.txt
Kaggle/Colab usually includes TensorFlow. Match its supported versions rather than forcing a framework downgrade in a running notebook. ImageNet weights need internet the first time; weights=None is for offline smoke checks, not the recommended final experiment. GPU use is automatic. The recommended preset enables mixed precision when a GPU is present; leave it off if your device has stability or performance problems.

DATA INPUTS
Extract Classification.zip and Anatomy.zip from Model_Datasets into a writable runtime directory. Classification must contain metadata/dataset_manifest.csv and train/test image folders. Anatomy must contain USOVA3D/, FollicleFinder/Data.zip and Supplementary_COCO/. The old/new full dataset already includes all 16,729 older files. No new exclusion or relabeling is applied by this code. Source label differences remain as supplied; the code cannot promise to learn consistent labels from conflicting supervision.

SET VARIABLES IN PYTHON OR REPLACE THESE PLACEHOLDERS
CLASSIFICATION = extracted Classification directory
ANATOMY = extracted Anatomy directory
WORK = writable experiment-output directory (for example /kaggle/working or /content)
All commands below use these placeholders as paths. In shell, define C, A, W appropriately, then use quoted paths.

RECOMMENDED SEQUENCE
1. Validate the classification manifest:
python run_continuation.py audit --classification "$C"

2. Prepare anatomy. This reads the FollicleFinder ZIP directly, avoiding full-volume extraction. Slice step 4 is a compute-saving 2D adaptation; use 1 if resources allow. USOVA rater 1 supplies the default targets; rater 2 remains available in source data for later agreement studies. Original source grouping/splits are preserved. The exact-image overlap screen removes any anatomy group matching classification validation/test image pixels; it is limited to exact original grayscale RGB matches and does not establish that all transformed overlaps are absent.
python run_continuation.py prepare-anatomy --anatomy "$A" --classification "$C" --out "$W/prepared_anatomy" --config configs/continuation/anatomy.json --slice-step 4

3. Train and evaluate anatomy:
python run_continuation.py train-anatomy --prepared "$W/prepared_anatomy" --out "$W/segmentation" --config configs/continuation/anatomy.json
python run_continuation.py eval-anatomy --prepared "$W/prepared_anatomy" --run "$W/segmentation" --out "$W/segmentation_validation" --role val
Review Dice/IoU, true-instance count error and instance F1/precision/recall at IoU 0.5. Native semantic-only volumes do not contribute to instance metrics. Empty masks can inflate averaged Dice, so inspect per-slice CSV and foreground examples. Main FollicleFinder DICOM semantic labels are decoded from SegmentSequence. Private follicle centroid annotations are not claimed as manually drawn instance boundaries. Genuine instance distance targets come from USOVA and one manual HDF5 observer. This is not a reproduction of FollicleFinder's 3D local-shape-descriptor architecture.

4. Cache anatomy predictions for the classification images. This uses predicted masks, never classification class labels or manual test masks:
python run_continuation.py extract-anatomy --classification "$C" --seg-run "$W/segmentation" --out "$W/features" --batch-size 16

5. Train the baseline and controlled comparisons. Each run must use a fresh output directory:
python run_continuation.py train --classification "$C" --out "$W/baseline" --config configs/continuation/original_baseline.json
python run_continuation.py train --classification "$C" --out "$W/image_only" --config configs/continuation/image_only.json
python run_continuation.py train --classification "$C" --features "$W/features" --out "$W/localized" --config configs/continuation/localized_no_morphology.json
python run_continuation.py train --classification "$C" --features "$W/features" --out "$W/fusion" --config configs/continuation/recommended.json
The original baseline preserves its architecture; its new optimization/evaluation protocol is recorded and should not be described as a literal reproduction of the historical 91.25% run. The localized and fusion configurations use the same shared global/ROI image encoder; fusion adds measured anatomy. Baseline/image/ROI models are different capacities and must be interpreted as separate ablations.

6. Optional validation-only search. Four trials can take hours. Stop expanding the search if resources are limited; a larger model is not guaranteed to improve this task. Train additional seeds by changing seed in a config and using a new run directory.
python run_continuation.py search --classification "$C" --features "$W/features" --out "$W/search" --config configs/continuation/recommended.json --trials 4

7. Freeze selected runs before calibration. Validation groups are deterministically divided into model-selection and calibration subsets. Checkpoints use selection balanced accuracy; calibration labels do not pick architectures or early-stopping epochs. Test images are not consumed by fitting or search.
python run_continuation.py calibrate --classification "$C" --features "$W/features" --runs "$W/fusion" --out "$W/calibration"
python run_continuation.py evaluate --classification "$C" --features "$W/features" --runs "$W/fusion" --calibration "$W/calibration/calibration.json" --out "$W/final_test"
For an ensemble, pass multiple run paths to --runs for BOTH commands. For flip TTA, pass --tta to BOTH. Calibration verifies weights, config, run metadata, morphology normalization, class order, preprocessing version, dataset manifest and TTA setting. These checks run before final-test inference. Choose neither models nor referral thresholds using final-test results. Report both accuracy and class-balanced measures; source-specific reports can reveal shortcuts. Temperature scaling adjusts confidence, not top-class accuracy for a single probability vector. A validation referral target is not a guarantee of future safety or clinical accuracy. If no adequately sized calibration subset meets the target, all predictions are referred.

8. Predict an image after calibration:
python run_continuation.py predict --image example.png --runs "$W/fusion" --seg-run "$W/segmentation" --calibration "$W/calibration/calibration.json" --out "$W/example" --explain
Attribution is approximate integrated gradients on the primary image input, with other inputs held fixed; it is not Grad-CAM and does not validate correctness. Legacy Grad-CAM utilities remain available through the old code. Instance masks are in segmenter input coordinates; ROI crop is exported in original-image coordinates. Measurements are relative 2D quantities, not millimetres, clinical AFC or total whole-ovary counts. Missing ovary/follicles remain flagged missing, not confidently normal.

RECOVERY / CHECKPOINTS
Use the same train command with --resume after interruption. It checks the dataset/config/cache match, restores best weights, skips completed stages and restarts an interrupted stage's epoch budget with a fresh optimizer. A resumed stage/anatomy run retains its previous checkpoint unless a new validation score improves on it; recovery logs append instead of erasing earlier epochs. This is weights-based recovery, not exact optimizer-state/step resumption. A smoke run cannot become a full run through --resume. Keras may warn about differing optimizer slot counts when transferring weights between fine-tuning stages; optimizer state is intentionally reset and weight tensors are loaded. Keep trained run config/data/normalizer/checkpoint files together and unchanged.

OPTIONAL METHODS AND LIMITS
Class weighting and inverse-square-root repeat weighting operate on fitting records only. Every supplied fitting record is still used. Source-provided labels are not silently changed. Focal loss is an alternative configurable ablation (focal_gamma), not assumed superior to cross-entropy. Mild flips/contrast changes are training-only; classifier augmentation does not crop away follicles or change scalar morphology. Segmentation applies the same flips to images and masks.
The domain_experiment preset adds a gradient-reversal source head. It is OFF by default because source identifiers are collection labels, not guaranteed scanners, and class/source association can make adversarial learning harmful. Its benefit must be measured against the same architecture without that head. No scanner-invariance claim follows from enabling it.
Supplementary Roboflow COCO data is not automatically converted to expert ovary ground truth. Its annotation quality, categories and cross-source overlap need review before training inclusion. No patient-independent evaluation is required or claimed; the evaluation unit is the image, with detected similarity groups kept together. This does not establish source-invariant or clinical generalization.

TESTS
python -m unittest discover -s tests -p 'test_*.py' -v
The portable code package includes the 27 continuation/augmentation/audit tests. The canonical workspace also has 8 older ablation tests.
PYTHONPATH=. python tests/integration_continuation.py
The integration test trains on tiny synthetic data only. It checks the complete computational path and does not measure medical accuracy. Local checks also decoded sample real USOVA3D and FollicleFinder volumes. No full real-data training or achieved accuracy is claimed in this delivery.

Research rationale and primary references are in RESEARCH_SOURCES.txt.

After all segmentation/postprocessing choices are fixed, run eval-anatomy with --role test once for the final anatomy result. Do not select thresholds from anatomy test results.

Additional matched-capacity control: configs/continuation/fusion_zero_morphology_control.json uses the same fusion architecture and training configuration but zeroes the measured-morphology input in both training and inference. Compare it with recommended.json to test whether the extra measured information helps beyond branch capacity. Both localized/fusion nonlegacy presets use the same mixed-precision setting.
Legacy plotting commands may additionally require pip install -r requirements-legacy-extras.txt. They are separate from the continuation entry point.
Anatomy training also supports --resume: it restores best weights and restarts the epoch budget, with config/manifest checks.

TRAINING APPEARANCE AUGMENTATION UPDATE
The default fitting pipeline now uses per-image brightness/gain factors 0.85-1.15, contrast factors 0.90-1.10, gamma 0.90-1.10, and multiplicative shared-channel noise with relative standard deviation sampled from 0 to 0.025. Existing horizontal/vertical flips remain. These are mild engineering variations, not a validated scanner-physics simulation. No cropping, elastic warping, new anatomic structures, arbitrary resizing of follicles or annotation recoloring is introduced. Black padding/background stays black.

Classification augmentation is inside the model and active only with training=True. Global and ovary-crop views share the photometric factors and flip decisions; noise is independently drawn for each view. Scalar morphology is not altered. Legacy-baseline reruns now receive the same appearance augmentation so comparisons use the same input perturbation policy. Validation, calibration, testing and prediction use identity augmentation and do not advance the augmentation RNG.

Anatomy training applies the same flips to image/masks/distance targets, then changes brightness/contrast/gamma/noise only on the image. The masks and their instance labels are never photometrically augmented. augmentation=False disables both new appearance augmentation and fitting flips.

Augmentation is generated on the fly for each fitting batch. The 19,942-file collection and frozen evaluation memberships are unchanged. No duplicated augmentation files are inserted into train/test. Preview PNGs are demonstrations only and are outside the dataset folders.

Run a preview/audit using real fitting images:
python run_continuation.py preview-augmentation --classification "$C" --out "$W/augmentation_preview" --samples 500

configs/continuation/no_augmentation_control.json gives the same classifier without augmentation for a validation comparison; gentle_augmentation.json offers weaker appearance perturbations. The bounded search can also vary augmentation strength using model-selection validation only. Do not choose augmentation using test results. The initial ranges are defaults, not a claim they maximize accuracy for this task.

This changes the training protocol. Start a new run directory for the augmented experiment; do not silently resume an older run with changed settings. The source image archives need not be rebuilt; use the updated code ZIP with the same data ZIPs.

FULL FITTING-COLLECTION AUGMENTATION AUDIT
To exercise the pipeline on every fitting file, without excluding repeated old files:
python run_continuation.py bulk-augmentation-check --classification "$C" --out "$W/bulk_augmentation_audit" --views-per-image 4 --batch-size 32
The current fitting set has 12,763 records; four draws each produce 51,052 streamed image views. Original checksums, finite pixel ranges, evaluation identity and fitting-only membership are checked. Every fitting file receives the requested number of draws, even if it is a retained old duplicate. The audit writes per-source summaries, not tens of thousands of stored duplicates. The actual model creates new random views during fitting; it does not consume the audit CSV as extra training data. With 36 configured classifier epochs, a run can present up to 459,468 fitting views before considering early stopping. These are image presentations, not additional independent medical cases. This augmentation command does not train model weights.

AUDIT UPDATE: VERSION 2.2
Classifier training, held-out evaluation and single-image prediction use the same TensorFlow RGB decoding and float32 letterbox resizing, including JPEG decoding. Segmentation retains its separate shared grayscale Pillow preprocessing. Anatomy evaluation uses the saved mask thresholds and watershed parameters and checks its training-manifest signature. Instance metrics are null when no genuine instance targets are present.

Feature cache version 2 includes a checksum for each ROI crop, checked before use. Re-extract older feature caches before training a new run. Calibration files from earlier code versions must be regenerated on the reserved calibration subset; do not reuse them with changed preprocessing or run artifacts. Existing trained model weights are not retrained by this update.

The manifest audit checks all four computational roles (fitting, selection, calibration and test), including identical pixel hashes that have different group IDs. Search cannot reuse smoke results or a run from a different anatomy cache. Original images, source labels and dataset memberships are unchanged. Superseded starter notebooks and archives are reference files only; use Continuation_Training.ipynb and Ovarian_Continuation_Code.zip for this continuation.
