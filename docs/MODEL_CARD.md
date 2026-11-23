# Model card

## Intended use and output

Research comparison of four supplied ovarian-ultrasound image classes, ordered Benign/Malignant/Normal/PCOS. Outputs are four class probabilities, optional temperature-calibrated confidence, and a calibration-derived review/referral decision. It is not a clinical diagnostic service. Referral improves the error rate among retained cases by reducing coverage; it is not an improvement in all-case balanced accuracy.

## Original architecture

`model/hybrid_model.py` concatenates three learned image-feature branches. FuzzyConv branch:32/64/128 filters, BatchNorm, max pooling after first two, global average pooling and Dense128. FollicleNet is a small image-feature CNN, not a supervised follicle detector. EfficientNet-B0 is the third pretrained branch. Concatenated features pass Dense512/256/128 with ReLU and BatchNorm, Dropout.35/.3, then4-way softmax. See the actual branch files for operations. The current rerun uses its new saved training/evaluation protocol; it does not literally reproduce the historical91.25% experiment. Its saved configuration uses label_smoothing0 and mixed_precision=false; newer default classifiers use.03 and true.

## New classifier and controls

`continuation/models.py`: Keras EfficientNetV2-B0 pooled global feature; optional ROI encoded by the SAME backbone; optional anatomy feature branch. Supported alternatives include EfficientNet-B0, EfficientNetV2-S, ConvNeXtTiny. Image preprocessing is TF RGB decode and float32 aspect-preserving letterbox to224pixels, application-native [0,255] preprocessing. Do not apply PyTorch mean/std to this Keras branch.

Morphology input is13 values plus13 missingness flags, normalized using FIT only. Morphology branch Dense64/GELU/Dropout.15. Final classifier Dense256/GELU/LayerNorm/Dropout.35/Dense4 float32softmax. Image-only excludes ROI/morphology. Localized adds ROI only. Fusion adds measured morphology. Zero-morphology preserves fusion architecture but supplies zero anatomy input during fitting/inference. The optional gradient-reversal source head is disabled in saved default models; repository collection IDs are not verified scanner IDs.

## Anatomy network

`continuation/anatomy.py`:2D U-Net,256x256x1. Two3x3 Conv/LayerNorm/ReLU operations per block. Encoder filters24,48,96,192; bottleneck384/Dropout.2. Four bilinear-upsample/skip-concatenate decoder levels. Three1x1 sigmoid float32 heads: ovary, follicle, distance. Semantic heads use Dice+BCE; genuine-instance slices supervise normalized per-instance distance. Semantic-only FollicleFinder data has zero distance weight. Source annotations and folds remain distinct; this is not the released3D FollicleFinder network.

## Explicit morphology

`continuation/morphology.py` calculates ovarian area fraction, bounding-box aspect ratio, approximate circularity, visible follicle count, follicle area fraction, mean/std/max follicle diameter relative to ovarian equivalent diameter, peripheral fraction, mean normalized radial position, nearest-neighbor mean/std, centroid dispersion. These are2D pixel-relative measurements. There is no claim of clinical size in millimetres, complete3D count or endocrine diagnosis. Missing anatomy/features remain flagged, not interpreted as healthy.

Instance methods include marker watershed and validation-selected connected components. Refined settings: follicle threshold.8, minimum30pixels; ovary threshold.5. Connected components may merge touching follicles. The isolated new guard expands valid ROI margins to.25; proposals with maskfraction<.01 or original.1-margin cropfraction<.15 fall back to full image and NaN morphology. The guard is not in the saved classifiers or TinyUSFM image-only experiment.

## Saved best inference

Equal probability average of `original_baseline`, `image_only`, `fusion_zero_morphology`, then the saved ensemble calibration/referral recipe. Test BA89.5457687538%; raw accuracy89.2427281846%; macroF1.8605949762. All3,988records are included. Architecture includes anatomy-derived ROI in one member; the best ensemble does not demonstrate that measured morphology improves classification. The anatomy-containing alternative ensemble includes `fusion_refined` and scores89.4225%testBA. See individual aggregate metrics, calibration signatures and exact recipe files.

## New ultrasound-pretrained branch

`accuracy_development/ultrasound_transfer.py`: official TinyUSFM,192-dim transformer,12blocks/12heads, GAP. PyTorch image preprocessing differs: Pillow bilinear224RGBletterbox, ImageNet mean/std; fitting-only horizontal reflection/gain/contrast/gamma/speckle, explicit contiguousfloat32. Official norm tensors map explicitly to fc_norm for GAP. Released backbone initialized, new4-class head learned. Pilot used4headepochsLR3e-4,8last-six-blockepochsLR1e-5,4full-model epochsLR3e-6, AdamW/cosine/label smoothing.03/clip1. Pilot selection82.9666%, rejected.

`ultrasound_warm.py`: strict pilot restore, seed2628, allparameters trainable,40epochmaximum,90minutesoftcap, AdamW,3epochwarmup→3e-5 then cosine→1e-6, decay1e-4, smoothing.03,clip1,patience6. Full GPU run requires reproduction guard <=1e-5 and identicalargmax. FP32 deterministic, T4firstGPU. Longer warm result pending at snapshot. External pretraining overlap unknown. Same-image selection comparison is exploratory after pilot review; no test tuning.

## Uncertainty and interpretation

Temperature scaling measures/calibrates probability reliability on reserved calibration records. The referral rule is chosen there, not from test results, and can reject all cases if its calibration target lacks support. It is not a clinical safety guarantee. Optional attribution is a24-step input-gradient/integrated-gradient approximation on the global input, with ROI/morphology heldfixed. It is not standard Grad-CAM, segmentation groundtruth or causal evidence. Legacy Grad-CAM utilities also exist separately.
