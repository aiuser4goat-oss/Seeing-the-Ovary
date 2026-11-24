> Active dataset update: the user explicitly excluded210test IDs after error review. V3has19732records/3778test; originalV2results remain unchanged. Read [dataset revision](DATASET_REVISION_V3.md). Earlier retention/count statements below describe V2.

# Data and evaluation protocol

The downloaded data remain private inputs. No image archive or individual medical image is part of this repository. Counts refer to the actual organized project export, not a claim that every source image is unique or clinically verified.

## Classification

19,942records: all16,729older readable files retained, plus deduplicated additions. Original folder layout train/test, each Benign/Malignant/Normal/PCOS. Validation metadata resides inside train; fitting must use FIT rows rather than every train image. Zero-byte nonimages were retained separately as references rather than passed to the image model.

Metadata roles: FIT12,763; SELECT2,112; CALIBRATE1,079; TEST3,988. SELECT/CALIBRATE derive from val metadata by SHA256(image_group) first8hex modulo3. Remainder0 reserves calibration. Image groups and identical decoded pixels must not cross roles. Detected similarity grouping does not prove that every transformed or related view has been found, nor establish patient independence.

Repeated older files remain by explicit project constraint; all source labels remain supplied. New PCOS infected/noninfected maps toPCOS/Normal. Five-class healthy/poly_cyst can map toNormal/PCOS; dominant_follicle/simple_cyst/complex_cyst references do not establish benign/malignant and were not silently assigned those labels. Source-matched PCOSGen/MMOTU additions were deduplicated. This is provenance accounting, not a reassessment of last year's labels or clinical findings.

Collection IDs in diagnostics are2526_Data_ALL,Additional_Five_Class,Recovered_MMOTU_2D,Recovered_PCOSGen,data. These are folder-level provenance, not verified institutions/scanners. The pooled classification split does not establish performance on an unseen data source. Several identical-input groups contain more than one supplied class in this combined record protocol; no class was adjudicated/changed. This is a constraint on what a deterministic image-only model can fit, not proof that an original study was mislabeled. The verified fitting-only BA ceiling is90.8796%, selection96.1911%; these are protocol bounds, not achieved scores or a guarantee that >93%testBA is feasible/infeasible. The historical test bound from metadata was not reverified at image-byte level and is not a certified result.

Training: keep every FIT record with positive weight. Inverse-sqrt duplicate count and capped class weights are computed on FIT only, applied once and mean-normalized. Online augmentation creates repeated presentations, not independent data.51,052streamedviews=4x12,763fittingrecords. It is not a52k-image independent dataset.

## Anatomy sources

USOVA3D locally contains3,419slices/16volumes with rater1/rater2 ovary and follicle binary/instance masks. Native split counts:2,593train,398validation,428test source slices;12/2/2volumes. Current2D pipeline samples everyfourthslice, usingrater1 targets and nearest-neighbor label resizing. Keep binary masks, instance IDs and color visualizations distinct. Volume-level source splits are retained to avoid adjacent slices crossing them; this does not create a patient-independent classification claim.

FollicleFinder downloaded archive contains533raw/validated3Dvolume pairs plus supplementary annotations. DICOM label meanings are read from SegmentSequence, grid metadata checked, and onlysemantic heads supervised for semantic masks. Private centroids are not claimed as manual instance boundaries. A manualHDF5observer supplies genuine instances for sixvolumes. Preparation reads the original source fold mapping; exact raw-grayscale pixel matching excludes anatomy groups overlapping protected classification roles. Transformed/color/cropped overlaps may escape this exact screen.

Roboflow supplementary downloaded exports:360digitmg410,Shivani523,Rahma1015. Their original project pages list variable/versioned counts. Annotation provenance/expertise and cross-source overlap remain unverified; none becomes expert whole-ovary groundtruth automatically. Ranaudio was hair-follicle material and is excluded. Supplementary candidates are not claimed as used in current supervised anatomy fitting.

MMOTU's official repository links accessible OTU2D images/annotations. These are ovarian lesion/tumor masks, not whole-ovary/follicle truth. Additional annotations were not downloaded/aligned for the current classifier. Classification source-matched images already used and additional annotation checking are different activities.

## Frozen signatures

Classification manifest SHA256:d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190.
Classification archive SHA256:604df1cc7c9ce8b6ace8e72f31c0a76bcce90dd661a5aa30f78674a46fa2d998.
Private cached .bin is byte-identical to that archive, not a rebuilt dataset. Safe staging verifies archive/manifest signatures, path bounds, symlinks, counts and existing-destination preservation. The training launcher separately verifies all14,875FIT/SELECT file hashes. Cache extraction includes stored evaluation bytes but does not fit/evaluate their pixels; fitting workers role-filter strictly.

## Analysis and claims

Balanced accuracy is the unweighted mean of four class recalls. MacroF1 averages four classF1values. Also report rawaccuracy, per-class recall/precision/specificity, confusionmatrix,ROC/PR where applicable,calibration,NLL/Brier/ECE,and coverage/error tradeoff. Bootstrap by detectedimagegroup, not each repeated record independently. This is image/group uncertainty, not patient-level validation. Source-specific metrics can expose source association but do not prove scanner invariance.

Test has been examined in completed analyses. New architecture selection uses SELECT only, calibration staysreserved, and further results are exploratory. A convincing future external-generalization study would need genuinely independent source material/labels, but such a study has not been completed or promised here. Preserve this limitation rather than claiming a fresh untouched test.
