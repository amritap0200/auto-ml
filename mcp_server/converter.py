import torch
import onnx
import onnxruntime as ort
import os

def export_to_onnx(model_path, input_shape, onnx_path):
    try:
        model = torch.load(model_path, map_location="cpu")
        model.eval()

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        return True, None
    except Exception as e:
        return False, str(e)

def validate_onnx(onnx_path):
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True, None
    except Exception as e:
        return False, str(e)

def get_onnx_session(onnx_path):
    providers = []

    available = ort.get_available_providers()

    # Try TensorRT first
    if "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")

    # Always keep CUDA / CPU as fallback
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session, providers
    except Exception as e:
        # Hard fallback to CPU
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        return session, ["CPUExecutionProvider"]
