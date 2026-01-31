# AutoML Performance Profiler

A FastAPI backend for profiling PyTorch model performance, converting to ONNX, and analyzing bottlenecks.

## Features

- ✅ PyTorch model upload and profiling
- ✅ Automatic ONNX conversion (stable, non-fatal)
- ✅ Performance bottleneck analysis
- ✅ ONNX Runtime inference optimization

## Quick Start

```bash
pip install -r requirements.txt
python mcp_server/main.py
```

## ONNX Export Stability

**PyTorch 2.x Note:** This project uses the legacy ONNX exporter (`use_dynamo=False`) for stability:

- **Why?** PyTorch 2.x introduced a new dynamo-based ONNX exporter that can fail with "missing onnxscript" errors even when onnxscript is installed.
- **Solution:** We explicitly disable the dynamo exporter and use the stable legacy path with `opset_version=11`.
- **Benefit:** Non-fatal ONNX export failures—the app continues to work even if ONNX export fails.

### Test ONNX Export

```bash
python test_onnx_export.py
```

This validates ResNet18 export with the same configuration used in production.

## Project Structure

```
mcp_server/
├── main.py           # FastAPI endpoints
├── converter.py      # ONNX conversion (legacy exporter)
├── profiler.py       # Model profiling
├── analyzer.py       # Bottleneck analysis
└── context.py        # Model context extraction

ui/
└── app.py            # Streamlit frontend

models/               # Stored model files
utils/                # Utility functions
```

## API Endpoints

- `POST /upload-model` - Upload and analyze PyTorch model
- `POST /run-profile` - Profile a model
- `POST /analyze` - Analyze profiling results
