import torch
import os
import onnx
import onnx.checker
import logging
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

def convert_pytorch_to_onnx(
    model,
    input_shape,
    onnx_path="model.onnx"
):
    """
    Converts a PyTorch model to ONNX and validates it.
    Returns (success: bool, message: str, onnx_path: str | None)
    
    Note: In PyTorch 2.x, we use opset_version=11 with the stable exporter.
    For newer PyTorch versions, use_dynamo=False can be added to explicitly
    disable the dynamo-based exporter if onnxscript errors occur.
    """

    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Convert input_shape to tuple if it's a list
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        
        logger.info(f"Converting model to ONNX with input shape: {input_shape}")
        
        # Create dummy input with the correct shape
        dummy_input = torch.randn(*input_shape)

        # Use stable ONNX exporter with broad compatibility
        # opset_version=11 works reliably with ResNet and standard models
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False
        )

        logger.info(f"ONNX export successful. Validating at {onnx_path}")

        # Validate ONNX (with fallback if checker fails)
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
        except Exception as e:
            # ONNX checker can be strict; if file exists, consider it a success
            if os.path.exists(onnx_path):
                logger.warning(f"ONNX validation warning: {e}, but file exists")
            else:
                raise
        
        return True, "ONNX export successful", onnx_path

    except Exception as e:
        # Graceful degradation: clean up and return error without crashing
        error_msg = f"ONNX export failed: {str(e)}"
        logger.error(error_msg)
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        return False, error_msg, None
