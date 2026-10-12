
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="VisionAI | Image Classifier",
    page_icon="🌄",
    layout="wide"
)

# Custom interface styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10172a, #24205b, #173e65);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

.hero {
    text-align: center;
    padding: 35px 15px;
    border-radius: 24px;
    background: linear-gradient(120deg, #6c45e8, #147dba, #16b8a6);
    margin-bottom: 28px;
    box-shadow: 0 12px 35px #00000035;
}

.hero h1 {
    color: white;
    font-size: 43px;
    margin-bottom: 8px;
}

.hero p {
    color: #f0faff;
    font-size: 17px;
}

.info-card {
    background: #ffffff12;
    border: 1px solid #ffffff30;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #184e77, #246b79);
    border: 1px solid #65e6d5;
    border-radius: 20px;
    padding: 25px;
    margin-top: 20px;
}

div[data-testid="stFileUploader"] {
    background: #ffffff0d;
    border: 1px dashed #8b9cff;
    border-radius: 18px;
    padding: 18px;
}

.stButton button {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 24px;
    font-weight: bold;
}

footer {
    visibility: hidden;
}

/* Make the upload text and Browse button visible */
div[data-testid="stFileUploader"] label {
    color: white !important;
}

div[data-testid="stFileUploader"] section {
    background-color: #f8fafc !important;
    border-radius: 14px !important;
}

div[data-testid="stFileUploader"] section button {
    background: linear-gradient(90deg, #7c3aed, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] section button:hover {
    background: #6d28d9 !important;
    color: white !important;
}

div[data-testid="stFileUploader"] small {
    color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("intel_cnn_model.keras")

model = load_model()

classes = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

class_emojis = {
    "buildings": "🏙️",
    "forest": "🌲",
    "glacier": "🧊",
    "mountain": "⛰️",
    "sea": "🌊",
    "street": "🛣️"
}

# Header
st.markdown("""
<div class="hero">
    <h1>🌎 VisionAI</h1>
    <p>Explore the power of Computer Vision</p>
    <p>Upload a landscape image and let AI discover its category.</p>
</div>
""", unsafe_allow_html=True)

# Information cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h2>🧠 CNN</h2>
        <p>Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h2>📸 6 Classes</h2>
        <p>Six scene categories</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h2>⚡ Real-time</h2>
        <p>Instant predictions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## 📤 Upload Your Image")
st.write("Choose a JPG, JPEG, or PNG image to get started.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_col, result_col = st.columns([1, 1], gap="large")

    with image_col:
        st.markdown("### 🖼️ Your Image")
        st.image(image, use_container_width=True)

    with result_col:
        st.markdown("### ✨ AI Prediction")

        resized_image = image.resize((150, 150))
        image_array = np.array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = classes[predicted_index]
        confidence = float(predictions[predicted_index])

        st.markdown(
            f"""
            <div class="result-card">
                <h2>{class_emojis[predicted_class]} {predicted_class.title()}</h2>
                <p>Predicted scene category</p>
                <h1>{confidence * 100:.2f}%</h1>
                <p>Prediction confidence</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("Confidence level")
        st.progress(confidence)

    st.markdown("## 📊 Class-wise Prediction Scores")

    for i, class_name in enumerate(classes):
        score = float(predictions[i])
        st.write(f"{class_emojis[class_name]} {class_name.title()} — {score * 100:.2f}%")
        st.progress(score)

else:
    st.markdown("""
    <div class="info-card">
        <h2>🌈 Ready to Explore?</h2>
        <p>Upload an image to see your AI model in action.</p>
        <p>Buildings · Forest · Glacier · Mountain · Sea · Street</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#b9c5ed;'>"
    "VisionAI | Intel Image Classification Project | Built with TensorFlow & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
