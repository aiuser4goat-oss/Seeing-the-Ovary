> Update: the warm run has completed and five new checks were rejected. Read [latest improvement round](docs/LATEST_IMPROVEMENT_ROUND.md) before the earlier snapshot below.

# Next experiments and decisions

## Already submitted

Finish the locked longer TinyUSFM continuation, verify its initial prediction guard, source/checkpoint hashes, final single-pass probabilities and class order. Compare only its two predeclared equal-weight ensembles with their matched SELECT references. Keep the old weights unless a candidate improves the matched selection result. Do not tune ensemble weights or send new predictions through the classification test to chase93%.

The pilot was still improving at its final epochs, which motivates one longer/higher-LR warm adaptation. This is a hypothesis, not evidence of a future gain. The paper's400epoch schedule is much longer than our free-compute budget. If the run fails the initial reproduction guard, investigate instead of widening tolerance after seeing scores.

## Most useful remaining anatomy question

Current predicted ROI can miss relevant ovarian area. A locked crop-guard test improves retention by using larger/fullimage crops; it does not yet show classification benefit. A useful FUTURE trial would compare original ROI vs guarded ROI with identical architecture, initialization, fitting records, optimization budget and SELECT rows. Add a morphology-vs-zero-morphology arm only if capacity and missingness handling match. The guard has not been integrated into the frozen cache. Recompute a new versioned cache and bind its source/segmenter hashes before training; never mix old ROI with new measurements.

A bounded zero-initialized morphology/ROI correction anchored to the stronger image-only classifier is another possible controlled trial. It could fall back to the original global prediction when anatomy is unavailable. This has **not** been implemented or run; it is not a calibrated anatomical-quality estimator. Its control would need the same gating/missingness capacity to isolate added morphology. Existing morphology-only correction code in accuracy_development is a different already-explored method, not proof that a gated ROI-feature correction has been tested.

## Diagnose before adding compute

Examine class-specific recall and selection error transitions, not just overall accuracy. Best completed test recalls:Benign86.61%,Malignant92.52%,Normal84.92%,PCOS94.13%. These historical test recalls explain the gap but must not become a case-by-case optimization target. Selection diagnostics are available separately. The TinyUSFM pilot improved52old-image errors but broke153old successes, supporting rejection of blind averaging. Check record weighting, source association, augmentation strength, finite input tensors and feature/missingness scaling before attributing failure to model size.

A source-robust experiment requires a clear source definition and matched control. Collection IDs are imperfect proxies. Source/class association can make gradient reversal harmful; enabling the optional head is not evidence of domain invariance. A withheld-source analysis must respect existing protected roles and cannot be claimed as already completed.

Distillation, SAM, ArcFace/PiCO-like objectives and stronger foundation encoders were reviewed as possible ideas but not implemented in the active warm run. Distillation from current task classifiers might preserve class boundaries in a new encoder; it needs a fixed teacher trained without protected data, labeled FIT cross-entropy retained, and a matched non-distilled control. Don't claim the generic candidate-marginal objective is PRODEN/PiCO or a tested implementation of all partial-label learning.

## Decision template

For any proposal specify: scientific question; exact source files to change; fitting-only inputs; matched control; predicted mechanism; training budget; fixed selection rule; reproducible inference; failure fallback; evaluation limitations. One well-controlled short trial is more informative than many selection-tuned recipes. Increasing SELECT reuse creates optimism; archive every attempt, including failed ones. No method can honestly guarantee93%.
