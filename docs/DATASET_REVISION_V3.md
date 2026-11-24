# Active dataset revision V3 — user-requested exclusions

The user's latest explicit instruction supersedes the earlier all-records-retained constraint for the active dataset. OriginalV2data, models and results remain preserved. This is a dataset membership revision, not a new trained model.

Exactly210listed recordIDs were removed from the active classification folder, manifest, testCSV and rebuiltClassification.zip. All210IDs were found; allwereTESTrecords. No labels were changed and no split was shuffled.

ActiveV3counts:19732total;12763FIT;2112SELECT;1079CALIBRATE;3778TEST. Trainingfolder15954unchanged. Excluded supplied classes:49Benign,102Normal,57PCOS,2Malignant.146wereolder-source records, explicitly overriding the previous preservation rule only for activeV3. Readable originals remain quarantined and originalmetadata/archive preserved.26otherTESTrecords share decoded pixels with excluded records but were not in the exact requested list; they are flagged locally for review, not automatically excluded.

The reason is user-reported inaccurate/uncertain class identity, not independent clinical adjudication. All210IDs appear among the previously misclassified historicaltestrecords. Thus this is a post-error-review filtered test population. A higher score after filtering would not establish improved model learning or independent generalization, and must not replace the original89.5458%V2testbalancedaccuracy. No filtered accuracy has been computed and no retraining has occurred in this operation.

The active localClassification.zip isV3. OriginalV2archive is preserved separately asClassification_V2_Original_19942_Records.zip. Prior privateKaggle datasets/cacheinputs remain historicalV2snapshots and were not overwritten or deleted. NoV3cloud upload/newtraining was performed. A new cloud input/version and protocol validation are needed before futureV3jobs. Older pinned launchers retain V2hash/count checks; do not silently repin old experiment declarations to make them accept V3.

To reproduce historical results, extract the preservedV2archive into a separate directory and point historical scripts to it. Their hardcodedClassificationroot now refers to activeV3locally, so it must not be assumed to be the old population. Old prediction files, calibration, confidence intervals and figures remain V2artifacts. Remaining labels and memberships are unchanged, but full-manifest signatures have changed.

The exact IDlist, excluded-record metadata, hashes, quarantinefiles, originalmetadata and related-record review list are stored locally under Dataset_Revisions/User_Exclusions_2026_11_23. Source images and case-level test details are not added to this public review repository.

NewmanifestSHA256: d68249a926657367ca8a584d645f8838109ba772fbac401703fc4354df4bab08

NewexportSHA256: e6989adc9a7ddb6eee2b4b74e62f48c4dab6be676e00190c206d9cde7d95fa4d

OriginalmanifestSHA256: d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190

OriginalarchiveSHA256: 604df1cc7c9ce8b6ace8e72f31c0a76bcce90dd661a5aa30f78674a46fa2d998

UserrequestIDlistSHA256: 40238043d576d7d398e5d37696b8f46598e910c7ad808e322c2dc4dab42c39b6
