import os
import numpy as np
from PIL import Image
import streamlit as st

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="EE5130 Digit Recognizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE THE MODEL ---
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('models/cnn_mnist.keras')
    except Exception as e:
        return None

model = load_model()

# --- UTILITY FUNCTION ---
def preprocess_and_debug(image):
    """Smart processing that handles any color ink and any background color."""
    image = image.convert('L')
    img_array = np.array(image)

    # 1. Smart Background Detection
    borders = np.concatenate([
        img_array[0, :], img_array[-1, :],
        img_array[:, 0], img_array[:, -1]
    ])
    bg_intensity = np.median(borders)

    if bg_intensity > 127:
        img_array = 255 - img_array

    # 2. Smart Thresholding (Isolates the ink mathematically)
    threshold = np.percentile(img_array, 85)
    threshold = max(threshold, 30)
    img_array = np.where(img_array > threshold, img_array, 0)

    # 3. Find bounding box of the actual digit
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 28, 28, 1)), None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 4. Crop and center strictly to MNIST standards
    cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    cropped_img = Image.fromarray(cropped.astype(np.uint8))
    cropped_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - cropped_img.width) // 2
    paste_y = (28 - cropped_img.height) // 2
    canvas.paste(cropped_img, (paste_x, paste_y))

    # 5. Format for the CNN
    final_array = np.array(canvas) / 255.0
    cnn_input = final_array.reshape(1, 28, 28, 1)

    return cnn_input, canvas


# --- MAIN APP LAYOUT ---
st.title("EE5130: Advanced Machine Learning Exhibition")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Panel")
    st.write("Test the Convolutional Neural Network. Write a single digit (0-9) on paper or screen.")
    
    upload_mode = st.radio("Choose Input Method:", ["📸 Use Mobile Camera", "📂 Upload from Gallery"])
    
    image_file = None
    if upload_mode == "📸 Use Mobile Camera":
        image_file = st.camera_input("Take a photo of a digit")
    else:
        image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

with col2:
    st.subheader("CNN Analysis Dashboard")
    
    if model is None:
        st.error("Fatal Error: CNN Model not loaded. Check your 'models/' folder.")
    elif image_file is not None:
        with st.spinner("Extracting topological features..."):
            
            image = Image.open(image_file)
            processed_img, debug_canvas = preprocess_and_debug(image)
            
            if debug_canvas is None:
                st.warning("Could not detect a clear digit in the image. Please try again.")
            else:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown("**Original Upload**")
                    st.image(image, use_container_width=True)
                with img_col2:
                    st.markdown("**Feature Extracted**")
                    st.image(debug_canvas.resize((150, 150), Image.Resampling.NEAREST), use_container_width=True)
                
                prediction_probs = model.predict(processed_img, verbose=0)
                predicted_digit = np.argmax(prediction_probs)
                confidence = np.max(prediction_probs) * 100
                
                st.markdown("---")
                
                st.markdown(f"<h1 style='text-align: center; font-size: 90px; color: #2c3e50; margin-bottom: 0px;'>{predicted_digit}</h1>", unsafe_allow_html=True)
                st.markdown("<h5 style='text-align: center; color: #7f8c8d; margin-top: 0px; letter-spacing: 2px;'>PREDICTED DIGIT</h5>", unsafe_allow_html=True)
                
                st.write("")
                st.markdown(f"**Model Confidence: {confidence:.2f}%**")
                st.progress(int(confidence))
    else:
        st.info("Awaiting Image Input...")
