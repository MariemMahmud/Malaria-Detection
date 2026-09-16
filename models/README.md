# models/

This folder holds the versioned, exported model files. It is intentionally
empty of actual weights in this deliverable — the notebook only saves the
Custom CNN checkpoint (`best_cnn_model.keras`), not the MobileNetV2 model,
which is the best-performing one.

## To finish setup

1. In the training notebook, after training the MobileNetV2 model, add:

   ```python
   model.save("models/malaria_mobilenetv2_v1.keras")
   ```

2. Copy the resulting `.keras` file into this folder so the path matches
   `model_metadata.json` (`malaria_mobilenetv2_v1.keras`).

## Versioning convention

- File naming: `malaria_{architecture}_v{N}.keras`
- Every new trained version gets a new file (`_v2`, `_v3`, ...) — never
  overwrite an old version in place, so you can roll back.
- `model_metadata.json` always describes the **currently deployed** version.
  Update its `version`, `file_name`, and `validation_metrics` fields whenever
  you promote a new model.
- `app/app.py` reads the active version from the `MODEL_VERSION` environment
  variable (defaults to `v1`), so switching versions in deployment doesn't
  require a code change.
