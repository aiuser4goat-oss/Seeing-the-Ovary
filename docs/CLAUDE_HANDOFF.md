> Update: the warm run has completed and five new checks were rejected. Read [latest improvement round](LATEST_IMPROVEMENT_ROUND.md) before the earlier snapshot below.

# Seeing the Ovary: technical handoff for Claude

Snapshot: November 23, 2026. Read this before proposing a model change. The student's science-fair deadline is December 2. The objective is to improve **full-set four-class balanced accuracy above 93%** while investigating whether measured ovarian/follicular anatomy helps classification. That target has **not** been reached. The best preserved completed ensemble has **89.5458% test balanced accuracy**. Its raw accuracy is 89.2427%; these are different metrics.

## Read these files in order

1. This document, then MODEL_CARD.md, DATA_PROTOCOL.md, RESULTS_AND_FAILURES.md and NEXT_EXPERIMENTS.md in docs/.
2. results/frozen_completed/ensemble_control/evaluation/test_metrics.json for the actual preserved best aggregate result.
3. continuation/models.py, training.py, data.py, augmentation.py, anatomy.py, morphology.py, features.py and evaluation.py.
4. model/hybrid_model.py and its three branches for the original architecture.
5. accuracy_development/ for experiments. These files are separate from the original completed model. Read experiment_reports/ and provenance/ before suggesting an already rejected method.
6. frozen_v2_3_1/ contains the text code snapshot associated with the preserved results, extracted from its hash-verified original archive. The current working source can contain later development fixes; do not silently substitute it as the historical executed source.

## Project and task

Last-year system: image-level EfficientNet-B0 + Fuzzy-CNN + a branch called FollicleNet. The latter was a learned image-feature branch, not supervised individual-follicle instance segmentation. Historical reported balanced accuracy was 91.25% on the earlier protocol. This year's expanded dataset, split, training and evaluation differ; that number is historical context, not a matched baseline score.

This year adds a supervised anatomy model, predicted ovarian ROI, explicit visible-follicle instances and 13 relative morphology measurements; a newer global encoder; controlled image/localization/morphology comparisons; calibration/referral; and interpretation. Domain-adversarial source learning is implemented as an optional experiment, **off by default**; no source-invariance result is established. The anatomy model is a 2D multi-head U-Net adaptation, not the original FollicleFinder 3D architecture.

Four-class order throughout: **Benign, Malignant, Normal, PCOS**. These are retained source image labels, not clinically reverified diagnoses. An ultrasound image label alone does not establish clinical PCOS. No patient-independent or clinical deployment claim is made.

## Non-negotiable protocol

- Preserve every one of the **19,942 classification records**, including **16,729 previous-year records**. Do not relabel or remove readable older images. Old copies can repeat; new additions were deduplicated during dataset construction.
- Keep current grouping and split. Detected identical/similar image groups stay together. Counts are image records, not independent cases.
- FIT 12,763; model SELECT 2,112; reserved CALIBRATE 1,079; TEST 3,988. Selection and calibration subdivide the metadata's val role by SHA256(image_group) first 8 hex digits modulo3. Remainder0 is calibration; the other two are selection.
- Shuffle fitting batches; do not repartition evaluation membership. Use metadata roles, not the entire train/ folder, for fitting.
- Calibration/test labels must not select epochs, hyperparameters, architectures, augmentation or ensemble weights. Classification test was already examined in completed work; follow-up development is explicitly exploratory. Do not pretend a future repeated test is a fresh untouched confirmation.
- Do not optimize the 93% goal by reporting ordinary accuracy, AUC, confidence-filtered accuracy, a subset, or a validation score as if it were full-set test balanced accuracy.
- Preserve original files and checkpoints. A candidate must improve a matched validation comparison before any promotion is considered. No guarantee that available data/resources support93%.
- Free resources only; no quota evasion, additional accounts, paid machines or public release of private data. The Kaggle jobs and input datasets are private. One requested GitHub destination is public; this repository therefore contains code/aggregate results, no images, credentials, checkpoints or case-level test predictions.

## Actual saved classifier architecture

The default newer image encoder is Keras ImageNet EfficientNetV2-B0. Application preprocessing is inside the model: input RGB is letterboxed float32 [0,255], **not ImageNet-normalized a second time**. Global image and ROI use the **same shared encoder**. Global-only uses one pooled feature vector; localized uses concatenated global+ROI vectors. Fusion also includes 13 normalized morphology values plus13 missingness indicators (26 numbers), Dense64 GELU and Dropout.15. The concatenated representation feeds Dense256 GELU, LayerNormalization, Dropout.35, then float32 Dense4 softmax.

The zero-morphology control has the same fusion capacity with morphology input zeroed, not a different smaller image-only model. The preserved best ensemble is an equal probability average of original_baseline, image_only and fusion_zero_morphology. Temperature scaling follows averaging under its saved calibration recipe. It does not change the top class of a single averaged vector.

Training configuration is copied in results/frozen_completed/*/config.json. Common original stages: head8epochs LR5e-4; last30% backbone20epochs LR3e-5; full8epochs LR5e-6; batch16; AdamW decay1e-4; cosine minimum fraction.05; clipnorm1; label smoothing.03 for newer models; focal_gamma0; patience6; seed2627; mixed precision on GPU for newer models; BatchNorm statistics frozen. The original_baseline exception uses label_smoothing0 and mixed_precision=false, as recorded in its saved config. Early stopping can shorten stages. Stage resume is weights-based with fresh optimizer, not exact step recovery.

All fitting records have positive weight. Base weight is1/sqrt(number of FIT records sharing pixel_sha256); capped class rebalancing is computed from these effective fitting counts and applied once; weights are normalized to mean1. Augmentation is sampled online: shared view flips and photometric gain ±15%, contrast ±10%, gamma ±10%, small speckle<=.025; validation/calibration/test augmentation is identity. The streamed51,052-view audit is four draws per fitting record, **not51k independent images** and not a stored expanded dataset.

## Anatomy supervision and limitations

USOVA3D is available locally despite an earlier access issue:3,419 source slices from16volumes, rater1/rater2 ovary and follicle labels. Source splits are retained. Model input256x256 grayscale, every fourth slice; primary targets rater1. FollicleFinder contributes validated DICOM semantic heads and a manual HDF5 observer's instance labels. Semantic masks do not become fabricated instance targets; their distance-head weight is zero. Counts are visible2D instances, not clinical AFC or whole-ovary counts. No physical-mm measurement is inferred.

Unverified Roboflow candidates remain separate (360digitmg410; Shivani523 in the downloaded version; Rahma1015). Ranaudio was identified as hair-follicle material and excluded. MMOTU lesion masks are not ovary/follicle masks. Its official source folder was checked, but additional annotations were not downloaded/aligned into the new experiment.

A validation-selected connected-components threshold.8/minimum30pixels greatly improved counts over the earlier watershed settings on100USOVA validation slices. This refinement is already part of the saved fusion_refined model, so it is not a newly available improvement. Connected components can merge touching follicles. Only two validation volumes were evaluated; slices correlate.

A new isolated crop guard checks the ORIGINAL10%-margin proposal and mask fraction; tiny proposals use the full image with all morphology missing, otherwise margin expands to25%. It improves anatomical annotation retention by accepting more context, not segmentation accuracy. It has **not** been added to the frozen classifiers or current image-only TinyUSFM jobs. See the locked separate anatomy-test report, not just validation observations.

## Completed and active development

The saved historical-test best remains89.5458%. Comparable selection-only references: old image-only87.4401%, old anatomy ensemble87.5748%; a four-way oldbaseline+oldimage+warmB0+ConvNeXt development ensemble88.0582%. The old best control's zero-morphology selection cache is absent, so don't claim a matched comparison to that control using an anatomy ensemble instead.

New high-resolution warmB0 selection87.2305%; ConvNeXt84.4422%. Frozen-feature linear/RBF readout best87.0271%. Removing PCA best86.9326%. A grouped out-of-fold class-offset correction reduced selection88.0582→87.8070%. Duplicate inverse-count weighting reduced87.0271→86.6818%. Candidate-marginal group-target objective was worse in a matched frozen-head study. All rejected as improvements. These failures do not prove every possible implementation of these ideas will fail; they document what actually ran.

Official ultrasound-pretrained TinyUSFM pilot completed16epochs (head4,last6blocks8,full4), selected BA82.9666%. Both predeclared fixed ensembles failed matched references: Tiny+oldimage86.1117%; Tiny+oldimage+oldbaseline+oldrefined87.2023%. New model corrected52 old-image-only selection errors but lost153 old successes;225 were wrong in both. The pilot is not promoted.

Longer warm TinyUSFM continuation is **QUEUED at snapshot**. It uses the pilot checkpoint (SHA9fcf116544efed994e1918266c79f1cb3c2d101f517d9683d54bff9b0fac8b58), new seed2628, full-encoder AdamW,3epoch warmup to3e-5 then cosine to1e-6, maximum40epochs/90minute soft cap, patience6. Initial predictions must reproduce all2,112 pilot rows within1e-5 with identical argmax on GPU before fitting. Initial weights remain fallback. CPU8+8 smoke passed; its100% score is not project accuracy. The original paper's400epoch downstream schedule is not replicated.

TinyUSFM is an official 192-dim,12-block,12-head GAP transformer; timm0.5.4. Released norm tensors are explicitly mapped to GAP fc_norm. External pretraining-image overlap is unknown, so this is exploratory transfer, not proven clean external generalization or an isolated causal estimate of pretraining benefit. Pinned source and payload hashes are in provenance/.

## Numerical reliability

A separate frozen-model consistency investigation found small process-dependent GPU probability differences; original-runtime reruns changed1/3,988 top classes, while deterministic reruns matched one another. Same tensors/weights were verified; the exact responsible GPU op was not isolated. CPU old-head reconstruction differs slightly from GPU; don't replace frozen reported scores with reconstructed values or loosen historical guards. New runs use deterministic execution and immutable single-pass probability caches. Hardware/precision variants must be labeled.

## What to ask Claude to do

Find the most plausible remaining bottleneck using the source and executed results. Distinguish implemented methods from suggestions. Focus on balanced accuracy and the weaker class recalls. Check actual feature scaling, augmentation, weighting, early stopping/EMA order, shared encoder, missingness, source associations and numerical guards. Propose a small matched validation experiment with a fixed hypothesis, controls, free compute budget, decision rule and inference path. Identify repeated-selection overfitting risk. Do not suggest changing supplied labels, resplitting, training on test/calibration, reporting referral-only accuracy as93%, adding unrelated data without provenance or blindly averaging a weak model. Be candid if the current data/protocol makes the target uncertain.

No trained weights or source images are in GitHub; anyone reproducing training must obtain the corresponding authorized inputs and checkpoints. Hashes identify expected artifacts without making them public. Do not assume a code repository grants access to private Kaggle inputs. No credentials should be added to proposed code.
