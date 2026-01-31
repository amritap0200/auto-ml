import streamlit as st
import requests

st.title("AutoML Performance Profiler")

uploaded_model = st.file_uploader(
    "Upload PyTorch model (.pt)",
    type=["pt"]
)

input_shape = st.text_input(
    "Enter input shape (e.g. 1,3,224,224)"
)

if uploaded_model and input_shape:
    if st.button("Upload Model"):
        files = {"file": uploaded_model}
        data = {"input_shape": input_shape}

        response = requests.post(
            "http://localhost:8000/upload-model",
            files=files,
            data=data
        )

        st.json(response.json())
