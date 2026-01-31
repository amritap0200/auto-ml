from fastapi import FastAPI, UploadFile, File, Form
from mcp_server.context import create_model_context
from mcp_server.profiler import profile_model
from mcp_server.analyzer import analyze_bottleneck, analyze_bottlenecks
import uuid

app = FastAPI()




# In-memory storage for profiling results (for demo purposes)
PROFILE_RESULTS = {}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    input_shape: str = Form(...)
):
    model_context = create_model_context(file, input_shape)
    return model_context


@app.post("/run-profile")
async def run_profile(model_context: dict):
    """Profile a model with the given context"""
    results = profile_model(model_context)
    return results


@app.post("/analyze")
async def analyze(profile_results: list):
    """Direct analysis endpoint for profiling results"""
    return analyze_bottlenecks(profile_results)
