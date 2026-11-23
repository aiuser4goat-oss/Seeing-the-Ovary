# Latest improvement round — November23,2026

No93%result has been reached. The saved89.5458%historical test balanced-accuracy ensemble remains unchanged. No new classification test/calibration inference was performed.

The longer TinyUSFM continuation completed12epochs, selected85.3899%SELECTBA. Its GPU initial-prediction guard was exact. Equal-two warmTiny+oldimage scored87.6183%versus87.4401%oldimage, but did not beat the earlier88.0582%developmentensemble. The fixed anatomy-plus-Tinyfour scored87.4186%and failed its matched reference. All results remain exploratory after prior test review.

Five new matched SELECT checks:

- Image-only identity versus four-flipTTA:87.4202%→87.2797%, rejected.
- FIT-only BN running-statistics adaptation:87.4202%→36.8791%, rejected. One cleanFITpass, momentum.9,batch32,no gradients, encoder dropout disabled, all59BNmeans changed. Learned variables unchanged by hash and originalcheckpoint preserved. This is not exact population-statistic estimation or evidence that all BN protocols fail.
- Fixed frozen-feature source-balancing screen:87.0271%→86.7248%, rejected. MMOTU60/121→70/121correct,but main source1483/1737→1478/1737. AllFITrecordspositive, within-class source factors raw-capped4 then classmass restored. Not end-to-end source-balanced training.
- Fixed frozen-feature GCEq.7 versusCE:87.0271%→86.3833%, rejected. Both converged;32finite-difference gradient coordinates passed. Same features, initialization, regularization and optimizer budget. Not end-to-end CNN training; GCE loss scale/regularization interaction remains part of this treatment.
- Fixed equal-five exploratory ensemble after warm-result review:88.0582%→87.8311%, rejected. No weight/subset grid. MacroF1improvement does not override the BAobjective.

TTA identity is a separate deterministic CPUfloat32variant; it does not overwrite the historical GPUscore. Exact byte-identical inputs share computation, expanded back to all2112SELECTrecords. A preliminary pixel-hash version was stopped before scoring after agent review identified a decoder-consistency concern.

Visual engineering review covered all204distinct decoded-pixel shared-error thumbnails. It is not clinical annotation. Observed repeated views/templates, geometry/aspect differences, blackborders, measurementmarks and coloroverlays. Source/class/aspect/color/blackfraction summaries are descriptive, not proof of shortcuts. No images/labels/splits changed. The225records mean errors shared by oldimage and Tinypilot, not always-wrong across every model.

Next research priority is a matched original-versus-guardedROI study with fixed model capacity and versioned cache. It has NOT run and has no demonstrated classification benefit. Blindly stacking failed treatments or optimizing historical test cases is not justified.

See experiment_reports/AUTONOMOUS_IMPROVEMENT_ROUND.json and per-experiment declarations/results. Local image contact sheets, weights and probability arrays are intentionally not published.
