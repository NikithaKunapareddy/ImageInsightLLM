import easyocr
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # Forces CPU usage to avoid meta tensor error
)

st.set_page_config(page_title="Modern Edge Agent", layout="centered")
st.title("Modern Edge Agent: Image Captioning & Generation")
st.write("Upload an image to get a caption, or enter a prompt to generate an image. Runs on edge devices!")

# --- Image Captioning (BLIP) ---
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    )
    return processor, model

# --- Text-to-Image (Stable Diffusion) ---

import requests
import base64
def generate_image_via_api(prompt):
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        # Try to handle both raw image and base64 JSON
        try:
            # Try raw image
            from io import BytesIO
            img = Image.open(BytesIO(response.content))
            return img
        except Exception:
            # Try base64 JSON
            result = response.json()
            if isinstance(result, dict) and "image" in result:
                img_data = base64.b64decode(result["image"])
                from io import BytesIO
                img = Image.open(BytesIO(img_data))
                return img
            else:
                raise RuntimeError(f"API returned unexpected format: {result}")
    else:
        raise RuntimeError(f"Image API Error: {response.status_code} - {response.text}")

tab1, tab2 = st.tabs(["Image Captioning", "Text-to-Image Generation"])

with tab1:
    st.header("Image Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processor, model = load_blip()
        # Short caption
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"**Caption:** {caption}")

        # OCR for text extraction
        st.write("Extracting text from image...")
        import numpy as np
        reader = easyocr.Reader(['en'], gpu=False)
        image_np = np.array(image)
        ocr_result = reader.readtext(image_np)
        # Only keep text with high confidence and filter out short/meaningless results
        filtered_text = [item[1] for item in ocr_result if item[2] > 0.5 and len(item[1].strip()) > 2 and item[1].isalpha()]
        extracted_text = " ".join(filtered_text)
        if extracted_text:
            st.info(f"**Extracted Text:** {extracted_text}")

        # Generate dynamic summary using Hugging Face Transformers (small LLM)
        st.write("Generating dynamic summary with open-source LLM (Transformers)...")
        try:
            if extracted_text:
                summary_input = f"{caption}. {extracted_text}"
            else:
                summary_input = caption
            output = summarizer(summary_input)[0]["summary_text"].strip()
            st.info(f"**Summary:** {output}")
        except Exception as e:
            st.warning(f"Summary generation failed: {e}")
            # Fallback to previous logic
            if extracted_text:
                summary = f"This image contains: {caption}. The following text is present: {extracted_text}"
            else:
                summary = f"This image contains: {caption}. No readable text was detected."
            st.info(f"**Summary:** {summary}")

with tab2:
    st.header("Text-to-Image Generation")
    prompt = st.text_area("Enter a prompt to generate an image:")
    if st.button("Generate Image") and prompt:
        with st.spinner("Generating image via Hugging Face API..."):
            try:
                image = generate_image_via_api(prompt)
                st.image(image, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")