CONTINUATION EXPERIMENT COMPLETION

The active private notebook is:
https://colab.research.google.com/drive/1rgkjGzdseCRGQ8GmJpiunAN7Xr8w2a45

The final added code cell runs finish_experiments.py from the private Drive project folder.
It follows the current full fusion training and calibration. It verifies the completed original-model rerun, trains image-only, localization, and zero-morphology controls, and calibrates all models using reserved validation images.

Only after all recipes, selected checkpoints, and calibration rules are frozen does it evaluate the held-out test set. The primary comparison is full fusion versus the original rerun. It never changes image labels, files, or the fixed split.

Private Drive results will include:
results/completion_status.json -- current stage or detailed failure
results/experiment_plan.json -- predeclared model recipes and code/data hashes
results/experiment_freeze.json -- selected model and calibration hashes
results/Final_Reports.zip -- test metrics, predictions, statistics, graphs, and limitations
results/model_packages/ -- one selected checkpoint and supporting artifacts per model
results/deliverable_hashes.json -- package integrity checksums

If Colab disconnects, reconnect a GPU runtime, rerun setup and feature-cache restoration, and run the completion cell again. Completed matching runs are reused. Interrupted training restores the best saved weights and restarts the interrupted stage's epoch budget and optimizer; it does not recover an exact optimizer trajectory. Final evaluation receipts prevent silent replacement of a frozen completed comparison.

All 19,942 supplied classification files remain in the fixed dataset. The old 91.25% balanced accuracy is a historical reference; a matched current-dataset baseline is trained separately. Current selection scores are validation results, not final test accuracy.

The anatomy model has usable semantic overlap but weak instance counting on validation. Reports retain that limitation. This run does not establish clinical diagnostic reliability or unseen-source generalization.

For a fresh local or cloud environment, install both requirements-continuation.txt and requirements-analysis.txt before final reports. The live Colab environment already has matplotlib 3.10.0.

Command: python -u finish_experiments.py --code PATH_TO_CODE --classification PATH_TO_CLASSIFICATION --features PATH_TO_EXTRACTED_FEATURES --results PATH_TO_RESULTS
