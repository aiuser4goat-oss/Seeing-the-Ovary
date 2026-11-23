# Seeing the Ovary

Anatomy-guided ovarian ultrasound classification: a science-fair continuation project by Ishaan Khera.

**For Claude or another technical reviewer: start with [docs/CLAUDE_HANDOFF.md](docs/CLAUDE_HANDOFF.md).** It explains the task, actual architecture, dataset, frozen results, failed experiments and current run.

The goal is >93% full-set four-class **balanced accuracy**. It has not been reached. Best preserved completed ensemble: **89.5458% test balanced accuracy**. New experimental validation scores are reported separately and must not be compared directly to that test score.

- [Model card](docs/MODEL_CARD.md)
- [Data and evaluation protocol](docs/DATA_PROTOCOL.md)
- [Results and failed approaches](docs/RESULTS_AND_FAILURES.md)
- [Remaining research questions](docs/NEXT_EXPERIMENTS.md)
- [Reproduction instructions](docs/REPRODUCING.md)
- [Sources and implementation status](docs/SOURCES.md)
- [Copy/paste review prompt](docs/CLAUDE_PROMPT.txt)

`continuation/` and `model/` implement the TensorFlow/Keras system; `accuracy_development/` contains separate experimental classifiers, guarded anatomy crops, PyTorch ultrasound transfer and the unchanged-data cache adapter. `frozen_v2_3_1/` preserves the historical text source associated with completed results. `results/` and `experiment_reports/` provide aggregate evidence and selection diagnostics.

Snapshot November23,2026: longer TinyUSFM warm continuation completed at85.3899%SELECTbalancedaccuracy. Five additional controlled checks did not beat their matched references; see [latest improvement round](docs/LATEST_IMPROVEMENT_ROUND.md). No new test predictions were generated for these development trials. Check docs/RUN_STATUS.json for snapshot status; a repository is not a live job monitor.

**Data and trained weights are not bundled.** The repository does not grant access to private Kaggle inputs. It includes no source ultrasound images, credentials or case-level test predictions. Author-provided pretrained TinyUSFM weights and source are obtained separately and hash-checked. Early README_CONTINUATION.txt statements such as “no training has run” describe the original delivery; this README, the handoff and saved metrics supersede those historical status statements.

This is research software, not a validated clinical diagnostic system. Source labels, visible2D morphology and predicted masks have the limitations described in the data protocol. Original dataset and third-party source terms remain applicable; this upload does not relicense them.

## Preserved best ensemble: per-class recall

- Benign: 86.61%
- Malignant: 92.52%
- Normal: 84.92%
- PCOS: 94.13%

Full-set test balanced accuracy: **89.5458%** across 3,988 records. These are frozen historical test results, not new evaluations.

- [Full test metrics](test_metrics.json)
- [Next experiments](NEXT_EXPERIMENTS.md)

## Specialist and verification update

[Specialist experiment and statistical report](docs/SPECIALIST_AND_STATISTICAL_VERIFICATION.md): plain andhard-minedNormal-vs-Benignheads didnotimproveSELECTbalancedaccuracy. FivegroupedFITfolds wereusedonlyforheadmining,notwhole-networkCV. [16verificationgraphs,PNG/PDF](verification_graphs/) and [CSV/JSON statistics](experiment_reports/normal_benign_specialist_v1/statistics/) areavailable. No93%result; original89.5458%testbestunchanged.
