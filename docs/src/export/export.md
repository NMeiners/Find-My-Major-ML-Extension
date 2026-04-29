# Module: src/export

## Responsibility
`src/export` implements export pipelines for frontend inference artifacts.
It trains the configured experiment model, exports the trained classifier as an ONNX model,
and writes a lightweight career database JSON file for browser-based ranking.

## Status
The export package now supports config-driven frontend artifact export.
Artifacts are written into the experiment `output.directory` run folder.

## Integration
- `main.py` calls `src.export.export_frontend_artifacts()` when `export.export_inference_model` is enabled.
- The module reads the first enabled dataset and model from the experiment config.
- The module relies on `src.models.MODEL_REGISTRY` so the same model contract is used by evaluation and export.

## Configuration
Supported export config keys:
- `export.export_inference_model`: boolean
- `export.format`: string, currently only `onnx`

Example:
```yaml
export:
  export_inference_model: true
  format: onnx
```

## Notes
- ONNX export depends on `skl2onnx`.
- Runtime errors are raised if required data files or model configuration are missing.
