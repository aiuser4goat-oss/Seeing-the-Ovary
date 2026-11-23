# Sources and implementation status

This list distinguishes methods implemented in project code from papers reviewed as possible future directions. Presence in a bibliography does not mean an implementation replicated the paper or achieved its reported benchmark.

## Used in implemented components

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML. https://proceedings.mlr.press/v97/tan19a.html . Original EfficientNet-B0 branch and matched rerun.

Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. ICML. https://arxiv.org/abs/2104.00298 . Default newer Keras image encoder. The exact project training schedule differs from the paper.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the2020s. CVPR. https://arxiv.org/abs/2201.03545 . ConvNeXt trial completed; it did not improve the stronger project selection reference.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI. https://arxiv.org/abs/1505.04597 . Multi-head2Dsegmentation adaptation, not exact reproduction of the original architecture.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. ICLR. https://arxiv.org/abs/1711.05101 . AdamW is used in classifier adaptation.

Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. ICLR. https://arxiv.org/abs/1710.09412 . Implemented in the separate development pipeline with mixed sample weights handled explicitly; not used in the TinyUSFM warm worker.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML. https://proceedings.mlr.press/v70/guo17a.html . Temperature scaling informs held-out calibration; calibration does not itself increase single-vector argmax accuracy.

Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. JMLR,17(59),1–35. https://jmlr.org/papers/v17/15-239.html . Optional gradient-reversal source head implemented but OFF in default completed comparisons. No demonstrated scanner invariance.

Ma, C., Jiao, J., Liang, S., Fu, J., Wang, Q., Li, Z., Wang, Y., & Guo, Y. (2026). TinyUSFM: Towards compact and efficient ultrasound foundation models. IEEE Journal of Biomedical and Health Informatics. https://arxiv.org/abs/2510.19239 ; https://doi.org/10.1109/JBHI.2026.3678309 . Official source https://github.com/MacDunno/TinyUSFM pinned f3096e97da3814011c5f59176b5cc19b5477ac64. Released backbone transferred in a16epoch project pilot and a queued longer warm continuation. Paper's400epoch downstream protocol not replicated. Pretraining overlap unknown.

## Implemented exploratory adaptations, not exact paper replicas

Kang, B., et al. (2020). Decoupling representation and classifier for long-tailed recognition. ICLR. https://arxiv.org/abs/1910.09217 . Motivated frozen-feature readout comparisons; project logistic/SVC trials are not a reproduction of all paper methods.

Menon, A. K., et al. (2021). Long-tail learning via logit adjustment. ICLR. https://arxiv.org/abs/2007.07314 . Motivated fixed/grouped out-of-fold class-offset investigation. The project study did not improve its reference and is not a claim of paper replication.

Lv, J., Xu, M., Feng, L., Niu, G., Geng, X., & Sugiyama, M. (2020). Progressive identification of true labels for partial-label learning. ICML. https://proceedings.mlr.press/v119/lv20a.html . Reviewed alongside a separate candidate-marginal group-target comparison. The latter does NOT implement PRODEN's progressive target-estimation method, and it failed its matched frozen-head comparison.

## Reviewed but NOT implemented in the active warm run

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. https://arxiv.org/abs/1503.02531 . Possible future task-teacher distillation; current TinyUSFM worker uses ordinary weighted supervised cross-entropy.

Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. ICLR. https://arxiv.org/abs/2010.01412 . Reviewed, not executed as SAM training.

Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. CVPR. https://arxiv.org/abs/1801.07698 . Reviewed objective; not implemented as a validated ultrasound method.

Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. CVPR. https://arxiv.org/abs/1901.05555 . Reviewed; actual saved weight formula is the explicit project inverse-sqrt duplicate plus capped class weighting, not automatically this loss.

PiCO official project page: https://hbzju.github.io/pico/ . Reviewed, not implemented; do not label the candidate-marginal experiment PiCO.

Less Contouring, More Accuracy: Lesion-Guided ROI Deep Learning for Ovarian Ultrasound Classification. (2026). https://arxiv.org/abs/2608.25965 . Reviewed in support of studying lesion-focused crops; tasks, lesions, labels and reported accuracy differ from our predicted ovary regions and balanced-accuracy goal. No transferable93% expectation follows.

## Dataset provenance and annotation candidates

MMOTU/DS2Net official source: https://github.com/cv516Buaa/MMOTU_DS2Net ; paper https://arxiv.org/abs/2207.06799 . Additional public lesion annotations were checked for availability, not newly aligned/trained in the current trial. Existing classification image recovery is a separate earlier step. Original author links and terms govern downloads.

TinyUSFM official released weights: https://drive.google.com/file/d/1P_wDuU-StALLJEBaymioaYu52CScW-l-/view . Weights SHA256776a50eeff402802c8cd06e00fe9de73df4161504fa733284bc240904b48252c. Not bundled inGitHub.

Supplementary Roboflow pages: https://universe.roboflow.com/360digitmg-d19qm/follicle-instance-segmentation-ersnk ; https://universe.roboflow.com/shivani-jx6hi/follicle-detection-n6jxi ; https://universe.roboflow.com/rahmas-workspace/follicle_instance_segmentation . These are unverified annotation candidates; page/project counts vary by export. Do not certify expert labels or combine native test splits without review. Ranaudio was excluded as hair-follicle material.

USOVA3D, FollicleFinder and PCOSGen acquisition/terms must be checked against their original downloaded source READMEs and metadata. This upload does not invent permanent record URLs or infer a license from public accessibility. Locally verified annotation representation and usage are described in DATA_PROTOCOL.md; original data and full local download metadata are not republished here. Earlier RESEARCH_SOURCES.txt and ADDITIONAL_REPORTING_METHOD_SOURCES.txt retain further primary references used in the core development.

## Official implementation documentation

Keras SwapEMAWeights: https://keras.io/api/callbacks/swap_ema_weights/ . Development callback order matters: swap EMA before validation and checkpoint selection. The current development runner includes an eager-validation correction; older launched payloads remain separately frozen.

TensorFlow deterministic operations: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism . Determinism is runtime/hardware scoped; it does not certify equivalence with older nondeterministic snapshots.
