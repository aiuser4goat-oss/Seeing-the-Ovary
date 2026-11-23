# Results and failures

## Completed classification test results

All rows use the same3,988records/2,163detectedimagegroups. Values below are aggregate frozen results, not new evaluation runs. Historical test was reviewed; follow-up development is exploratory.

- ensemble_anatomy: test balanced accuracy 89.4225%; accuracy 89.0923%; macro F1 0.863786.
- ensemble_control: test balanced accuracy 89.5458%; accuracy 89.2427%; macro F1 0.860595.
- fusion: test balanced accuracy 87.4471%; accuracy 88.1143%; macro F1 0.836484.
- fusion_refined: test balanced accuracy 87.5164%; accuracy 88.8415%; macro F1 0.851521.
- fusion_zero_morphology: test balanced accuracy 88.4469%; accuracy 89.2177%; macro F1 0.845803.
- image_only: test balanced accuracy 88.4032%; accuracy 88.9418%; macro F1 0.849690.
- localized_no_morphology: test balanced accuracy 88.3379%; accuracy 88.7161%; macro F1 0.840734.
- original_baseline: test balanced accuracy 86.5748%; accuracy 85.0552%; macro F1 0.808465.

The primary refined-fusion vs rerun-baseline difference is approximately+.9416percentage points balanced accuracy, with group-bootstrap95%CI[-.7075,+2.7689]points: inconclusive for a positive gain. The best ensemble is a separate result and not evidence of that primary hypothesis. Its BA95%CI is approximately[88.1384%,90.9634%]. Its retained/confident-caseaccuracy~96.02% covers84.33%ofcases and is NOT full-data accuracy. Historical91.25%used a different dataset/protocol.

## Selection-only comparisons

- Old image-only:87.4401%BA; old original baseline86.7465%; old refinedfusion86.5050%.
- Old anatomy equal3:87.5748%. Its members are oldbaseline+oldimage+oldrefined; this is not the best historical control ensemble whose zero-morphologyselectioncache is absent.
- New warm EfficientNetV2-B0:87.2305%; ConvNeXtTiny:84.4422%.
- Four-way oldbaseline+oldimage+warmB0+ConvNeXt:88.0582%. This is not a test result or an improvement over89.55%test.
- Grouped3fold offset adjustment:88.0582→87.8070%, rejected.
-12frozen-feature logistic/RBFcandidates:best87.0271%, rejected as replacement.
- Six noPCAreadouts:best86.9326%, rejected.
- Inverse-count duplicate/class weighting:87.0271→86.6818%; macroF1rose,butBAfell,rejected.
- Matched frozen linearhead group-objective comparison:softCE81.8805%,candidate-marginal78.3507%,rejected. This was not PRODEN/PiCO replication.
- TinyUSFMpilot:82.9666%,macroF1.786965;16epochs;~57.5minworker/~74.8minjob.
- Fixed Tiny+oldimage equal2:86.1117%; Tiny+oldimage+oldbaseline+oldrefined equal4:87.2023%. Neither improves its matched reference, rejected.
- LongerwarmTinyUSFM:pendingat snapshot.8FIT/8SELECTCPUprobe passed. Its100%smallsmokescore is not a performance estimate.

Repeated selection trials can overfit the development set; these scores are descriptive, not independent confirmations. CPU reconstructed oldhead87.4202%differs from GPUoldimage87.4401%;3topclassdifferences/maxprob.0083501. That precision/runtimevariant does not replace the frozen score or justify relaxing numerical guards.

## Anatomy findings

Validation100slices/twovolumes: earlier threshold.8/minarea30connectedcomponents reduced countMAE6.33→.38 and instanceF1.069945→.756579 compared with initialwatershed. This validation selected the refinement, so it is not independent confirmation; refinedfusionalreadyusedit.

Locked additional anatomytest:108slices/twovolumes (Vol1/Vol106),81positiveovary/73positivefollicle. Originalcropmeanovarycoverage93.7193%,guard97.3937%;under90%retention failures7→3;minimumremained0. Meanimagearea47.6327→74.9830%;fallbacks6→18. This is a larger-context tradeoff, NOT a segmentation or classification gain. RefinedcountMAE.879630,instanceF1@IoU.5.643357,precision.528736,recall.821429;alwayszeroMAE1.555556. Volume-specificresultsareprovided;Vol1modelcountMAEequalszero-countbaseline. Counts remain unavailable when guardrejectsanatomy, even though underlyingpredictionswereincludedindiagnosticaudit.

See experiment_reports/ for exact declarations and machine-readable evidence. No case-level test predictions are shared for optimization.
