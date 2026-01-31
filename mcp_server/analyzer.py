def analyze_bottleneck(profile_results):
    """
    Analyzes profiling results to identify performance bottlenecks.
    
    Args:
        profile_results: List of dicts, each containing:
            - latency_ms: float, inference latency in milliseconds
            - memory_mb: float, memory usage in MB
            - gpu_util: float, GPU utilization percentage
            - precision: str, precision mode (FP32, FP16, etc.)
            - batch: int, batch size
            - speedup_over_fp32: float, optional speedup ratio
    
    Returns:
        list: List of bottleneck findings with keys:
            - bottleneck: str, name of the bottleneck
            - reason: str, explanation of the bottleneck
            - confidence: float, confidence score (0-1)
    """
    if not profile_results:
        return [{
            "bottleneck": "No data",
            "reason": "Empty profile results provided",
            "confidence": 0.0
        }]

    findings = []

    for result in profile_results:
        # Skip if required fields are missing or invalid
        if not isinstance(result, dict):
            continue
            
        latency = result.get("latency_ms")
        memory = result.get("memory_mb")
        gpu_util = result.get("gpu_util")
        precision = result.get("precision")
        batch = result.get("batch")

        # Skip if critical fields are not numeric
        if not all(isinstance(x, (int, float)) for x in [latency, memory, gpu_util] if x is not None):
            continue

        # Rule 1: High latency + low GPU utilization
        if latency is not None and gpu_util is not None:
            if latency > 50 and gpu_util < 30:
                findings.append({
                    "bottleneck": "Hardware underutilized",
                    "reason": f"High latency ({latency:.1f}ms) with low GPU utilization ({gpu_util:.1f}%)",
                    "confidence": 0.82
                })

        # Rule 2: High memory usage with small batch size
        if memory is not None and batch is not None:
            if memory > 1000 and batch <= 1:
                findings.append({
                    "bottleneck": "Inefficient memory usage",
                    "reason": f"High memory usage ({memory:.1f}MB) at batch size {batch}",
                    "confidence": 0.78
                })

        # Rule 3: FP16 faster than FP32
        if precision == "FP16" and result.get("speedup_over_fp32", 0) > 1.2:
            findings.append({
                "bottleneck": "Precision-sensitive model",
                "reason": "FP16 significantly outperforms FP32",
                "confidence": 0.85
            })

    # Return all findings sorted by confidence
    if findings:
        findings.sort(key=lambda x: x["confidence"], reverse=True)
        return findings
    
    # No bottlenecks detected
    return [{
        "bottleneck": "No clear bottleneck detected",
        "reason": "Model performs within expected parameters",
        "confidence": 0.60
    }]
