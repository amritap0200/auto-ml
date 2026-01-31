from fastapi import FastAPI, UploadFile, File, Form
from mcp_server.context import create_model_context


app = FastAPI()

@app.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    input_shape: str = Form(...)
):
    model_context = create_model_context(file, input_shape)
    return model_context
