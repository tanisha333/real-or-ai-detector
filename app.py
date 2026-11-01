import streamlit as st
import torch
from PIL import Image
from transformers import CLIPImageProcessor
from src.models.detector import UniversalFakeDetector

st.set_page_config(page_title="Real or AI?", page_icon="??", layout="centered")

st.title("?? Real or AI?")
st.write("Upload an image to check whether it looks real or AI-generated.")

@st.cache_resource
def load_model():
    model = UniversalFakeDetector(
        model_name="openai/clip-vit-base-patch32",
        pretrained=True
    )
    model.load_probe("checkpoints/best_probe.pt")
    model.eval()

    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    return model, processor

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Image", type="primary"):
        with st.spinner("Analyzing image..."):
            model, processor = load_model()
            inputs = processor(images=image, return_tensors="pt")["pixel_values"]

            with torch.no_grad():
                fake_probability = model.predict_proba(inputs).item()

        confidence = max(fake_probability, 1 - fake_probability) * 100

        if fake_probability >= 0.5:
            st.error("?? AI-GENERATED")
        else:
            st.success("?? REAL")

        st.metric("Prediction confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

        st.caption(
            "Experimental model. Confidence is not the same as accuracy. "
            "The current balanced test accuracy is approximately 43.75%, "
            "so predictions may be incorrect."
        )
