#!/usr/bin/env python3
"""
Sanity test for ONNX export with PyTorch 2.x legacy exporter.
Tests ResNet18 export using use_dynamo=False and opset_version=11.
"""

import torch
import torchvision.models as models
import os

def test_onnx_export():
    print("🧪 Testing ONNX export with legacy exporter...")
    
    # Load ResNet18
    model = models.resnet18(pretrained=False)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "test_resnet18.onnx"
    
    try:
        # Export with stable exporter using opset_version=11
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
            }
        )
        
        # Verify file exists
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path)
            print(f"✅ ONNX export successful!")
            print(f"   File: {onnx_path}")
            print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
            
            # Cleanup
            os.remove(onnx_path)
            return True
        else:
            print("❌ ONNX file not created")
            return False
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        return False

if __name__ == "__main__":
    success = test_onnx_export()
    exit(0 if success else 1)
