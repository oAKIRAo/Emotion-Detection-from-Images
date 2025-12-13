import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# -------------------------------------------------
# PATH SETUP (to import from src/)
# -------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import EmotionEfficientNetB4

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

MODEL_PATH = "src/efficientnetb4_emotion.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_MAP = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger"
}

# -------------------------------------------------
# LOAD MODEL (cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = EmotionEfficientNetB4(num_classes=7)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# TRANSFORMS (same as test.py)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Emotion Recognition from Image")
st.write("Upload a facial image and get emotion probabilities.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    st.subheader("Emotion Probabilities")

    results = {}
    for idx, prob in enumerate(probs):
        results[EMOTION_MAP[idx]] = float(prob * 100)

    # Sort by probability
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # Display results
    for emotion, percentage in results.items():
        st.write(f"**{emotion}** : {percentage:.2f}%")

    # Bar chart
    st.bar_chart(results)

else:
    st.info("Please upload an image to start emotion recognition.")
