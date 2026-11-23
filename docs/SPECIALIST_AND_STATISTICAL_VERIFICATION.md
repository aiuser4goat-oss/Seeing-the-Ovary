# Normal-vs-Benign specialist and statistical verification

November23,2026. The two specialist heads did not improve matched SELECT balanced accuracy; neither is promoted. The preserved89.5458%historical-test ensemble is unchanged. No new classifier TEST or calibration-image inference was performed.

## Specialist experiment

Same frozen image-only encoder features for all comparisons. Main model baseline is deterministic CPUfloat32 identity inference, not the historical89.55%ensemble. Binary specialist heads use Benign/Normal FIT records only (7054records); the main classifier still uses all12763FITrecords. No image, supplied label, or split was changed.

Controls: unchanged main; plain binary head; hard-mined binary head. Both heads have identicalFIT-onlyScaler/PCA128/L2 preprocessing and LogisticRegressionC.1. Routing is fixed: only a main Benign prediction may switch to Normal at specialist score>=.5. No threshold/Cgrid. It can recover Normal recall at the cost of Benign recall; Malignant/PCOS recall is unchanged. It cannot repairPCOS→Benign mistakes.

Five grouped FIT folds generated head-level out-of-fold hard examples. Connected grouping joined declared similarity, bytehash and pixelhash, with disjointfold checks. Fold-specific preprocessing fitted unique training-fold feature rows. Both classes' OOFerrors received2xweights, applied once, every eligibleFITrecord retained with positive weight. These are conditional head-mining folds: the backbone had already seen allFIT, and inherited base weights came fromFIT. This is NOT independent whole-network five-fold validation. Existing project classifiers already underwent staged backbone fine-tuning; this study fits specialist heads, not a new separately fine-tuned binary encoder.

## Matched SELECT results

- Unchanged main:87.4202%balancedaccuracy, macroF1.849755.
- Plain specialist:87.1951%, macroF1.848705;3Normalrecoveries versus4Benignlosses. Delta−.2251pp; paired group-bootstrap95%CI[−.6006,+.0638]pp. Group-swap Holm3p=.1822. No convincing benefit.
- Hard-mined specialist:86.1767%, macroF1.843429;11Normalrecoveries versus20Benignlosses. Delta−1.2435pp; paired95%CI[−1.9456,−.5791]pp. Holm3p=.0012: exploratory evidence of a decline.

Scores cover all2112SELECTrecords. AdditionalPCOSswitches remain incorrect and do not increasePCOSrecall. No selective/retained-subset accuracy is substituted for the primary result. RepeatedSELECTexperiments create optimism; Holm3does not correct prior project search. No93%result achieved.

## Statistical methods and graphs

2500paired whole-image-group bootstrap draws for95%percentile intervals;5000pairedgroup-swap permutations for balanced-accuracy contrasts; Holm correction within three current comparisons. Swapping assumes exchangeability/symmetry, not an assumption-free equality test. Bootstrap groups are detected image-dependency clusters, not patients. Precision, recall, specificity, balancedaccuracy, accuracy, macroF1, binary specialist ROC/PR, full-class historical ROC/PR, calibration/NLL/Brier/ECE, class-by-source counts, risk–coverage and historical model/ablation comparisons are included. Specialist probability curves are two-class subset scores, not calibrated routed four-class outputs.

Historical analyses reread eight frozenTESTprediction files only. They do not retrain or tune anything. The newprimary historical difference refinedfusion−originalbaseline is+.9416ppBA with new group-bootstrap95%CI[−.6879,+2.6792]pp, inconclusive. Its macroF1difference is positive, but that does not prove the primaryBAhypothesis. Individual measured-versus-zero morphologyCI excludes0, while rawp=.04899 becomesHolm4p=.19596; do not call it significant after adjustment. Newbootstrapintervals do not overwrite archived originalintervals.

The frozen best ensemble still has89.5458%testBA. A covariance allocation attributes56.9%of bootstrapBAvariance to Malignant under this resampling scheme; that is not a share of mistakes or a causal attribution. Source classes with n=0areunavailable, and n=3cells are visibly labelled. Retained-caseaccuracy is separately labelled and notfullsetaccuracy.

## Reproducible inference and artifacts

Six computational smoke/parity checks support implementation: record/label/group checks; positiveweights; routing algebra; convergedheads; portable parameter parity on all2112cachedscores (<1e-6, identicalthresholddecisions); two native-image inference smokeexamples (one routed), not a medical performance estimate. Portable NumPy parameters avoid cross-version sklearn pickle loading. The saved routed system has not been probability-calibrated or clinicallyvalidated. Private weights and images are not uploaded.

16numberedfigures are each available inPNG/PDF. Reports/CSVs/JSONs and source scripts accompany them. Private original prediction archives are required to reproduce historical clustered uncertainty; public aggregate results alone do not grant access. Native image examples, case-levelTESTpredictions and specialist weightbundles remain local/private.

Methods references: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Actual custom resampling samples whole connected groups jointly, not independent image rows. Source and runtime hashes are saved.
