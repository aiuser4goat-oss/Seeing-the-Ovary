# Reproducing and inspecting the project

## Review without images or GPUs

Start with CLAUDE_HANDOFF.md. Aggregate model results, configs/calibration signatures, source hashes and selection error reports are included. `results/development/matched_selection_probabilities.csv` contains the2,112selected rows for old image-only and TinyUSFM, not classification test predictions. No images/weights are bundled, so the repository alone cannot reproduce medical model inference.

Run source syntax checks without installing training frameworks:

```sh
python -m compileall -q continuation accuracy_development model research_tools
```

## TensorFlow/Keras branch

Use a separate Python environment. `requirements-continuation.txt` describes the original core; `requirements-analysis.txt`/legacyextras are optional. Actual tested local environment Python3.13/TF2.20/Keras3.15; the successful cloud frozen-feature CPU runtime used TF2.20/Keras3.13.2. Do not downgrade base notebook packages in a live interpreter to force version guesses. Execute fitting in a fresh subprocess. Pretrained Keras weights require initial network access.

```sh
python -m pip install -r requirements-continuation.txt
python -m unittest accuracy_development.test_training_cache accuracy_development.test_anatomy_safety accuracy_development.test_development
```

All17these tests passed locally at snapshot. Tests train tiny synthetic models where needed; test scores are not medical results. The separate `tests/` folder includes earlier unit/integration checks. Historical verification files document the source version that ran; including them does not mean every test was rerun for this handoff.

With authorized extracted inputs, define CLASSIFICATION as its directory, ANATOMY as its directory and WORK as a fresh writable experiment path. Core commands:

```sh
python run_continuation.py audit --classification "$CLASSIFICATION"
python run_continuation.py prepare-anatomy --anatomy "$ANATOMY" --classification "$CLASSIFICATION" --out "$WORK/prepared_anatomy" --config configs/continuation/anatomy.json --slice-step 4
python run_continuation.py train-anatomy --prepared "$WORK/prepared_anatomy" --out "$WORK/segmentation" --config configs/continuation/anatomy.json
python run_continuation.py extract-anatomy --classification "$CLASSIFICATION" --seg-run "$WORK/segmentation" --out "$WORK/features" --batch-size 16
python run_continuation.py train --classification "$CLASSIFICATION" --features "$WORK/features" --out "$WORK/fusion" --config configs/continuation/recommended.json
```

These are reproduction entry points, not authorization to tune on or repeatedly evaluate classification TEST. Refined connected-components settings must be taken from the saved refined config/provenance, not assumed to be the starter defaults. Match the intended data, code, configuration, postprocessing, normalization, precision and inference signature before comparing historical scores. Frozen2.3.1text is supplied under frozen_v2_3_1; run from that root when reproducing that exact source.

## PyTorch ultrasound-transfer branch

Use a separate PyTorch environment. Successful pinned Kaggle container digest is in provenance/ULTRASOUND_RUNTIME_PROVENANCE.json. Warm CPU observed torch2.11.0+cu130,numpy2.1.3,CUDA13.0; timm0.5.4 installed --no-deps. Do not force these hardware-specific versions on a Mac. Local PyTorch training was not performed.

`notebooks/Warm_Validation.ipynb` is the exact source-only submitted notebook. It expects the project's private unchanged classification-cache dataset and private completed pilot output attached to Kaggle, has a full worker payload and source hashes, and runs on the firstT4. Its CPU counterpart runs only the8FIT/8SELECT smoke. Neither notebook contains credentials. Input dataset identifiers are not authorization tokens; other users do not gain access by opening it.

Official TinyUSFM source/checkpoint information and hashes are in provenance and the model card. The official model file is embedded in the original submitted notebook's frozen payload, with pinned authorship/provenance; see THIRD_PARTY_NOTICES.md. The external model's exact pretraining-image overlap is not known. Do not interpret a code hash as validation of its medical applicability.

Top-level research_tools builder/diagnostic scripts retain owner workspace paths to document the executed pipeline. Edit path constants or use the portable submitted notebook to reproduce on a different machine. Never rerun a builder against a live frozen payload/ensemble declaration and silently overwrite its historical record. Use a new version/directory for changed recipes.

## Artifact preservation

Private saved checkpoints are identified by hashes in provenance/PRESERVED_VERSION.json and source freezes. The historical final archive and2.3.1code ZIP were hash-verified locally; binaries are not copied to GitHub. Changes in this repository do not modify those originals. Before any future publication, independently check original dataset licensing and access terms; this code upload is not a release of the training datasets.
