import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG

# Setup
np.set_printoptions(suppress=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
IMAGE_PATH = os.path.join(BASE_DIR, "utils", "overview_dataset.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "weights", "leather_model.h5")
DATA_FOLDER = os.path.join(BASE_DIR, "data", "leather")

# UI Layout
st.set_page_config(page_title="Anomaly Detection", page_icon=":camera:")
st.title("Anomaly Detection ML")
st.caption("Quality Control and Anomaly Detection with Static Images of Leather Products.\nBuilt and Hosted by Vivek.")
st.write("Upload or capture a product image to classify it as Good or Anomaly.")

# Sidebar
with st.sidebar:
    try:
        img = Image.open(IMAGE_PATH)
        st.image(img)
    except FileNotFoundError:
        st.warning("Overview image not found.")

    st.subheader("About Anomaly Detection")
    st.write("This app uses AI to automate quality control by detecting defects in leather products.")
    st.write("Even minor scratches or discolorations can be caught using deep learning-based visual inspection.")
    st.write("Model Accuracey is about only 66% On given Dataset Provided")
    st.warning("Application Might be on Sleep as in Streamlit")

# Input Method
st.subheader("Select Image Input Method")
input_method = st.radio("Input Method", ["File Uploader", "Camera Input"])
uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.info("Please upload an image.")
elif input_method == "Camera Input":
    st.info("Allow access to your camera.")
    camera_image_file = st.camera_input("Capture an Image")
    if camera_image_file:
        camera_file_img = Image.open(camera_image_file).convert("RGB")
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image captured successfully!")
    else:
        st.info("Please capture an image.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anomaly Detection Function
def Anomaly_Detection(image_obj, root_path):
    batch_size = 1
    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
    except Exception as e:
        return f"❌ Model loading error: {str(e)}"

    _, test_loader = get_train_test_loaders(root_path, batch_size=batch_size)
    class_names = test_loader.dataset.classes

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_obj).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        predicted_probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = class_names[predicted_class_index]
    confidence = predicted_probabilities[predicted_class_index]

    st.write("Prediction Probabilities:", predicted_probabilities)
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", f"{confidence:.2f}")

    if predicted_class.lower() == "good" and confidence > 0.8:
        return f"✅ Your product is classified as 'Good'. Confidence: {confidence:.2f}"
    elif predicted_class.lower() != "good" and confidence > 0.6:
        return f"⚠️ Anomaly Detected with high confidence ({confidence:.2f})"
    else:
        return f"❓ Uncertain prediction. Class: {predicted_class}, Confidence: {confidence:.2f}. Please recheck."

# Run detection
if st.button("Submit a Leather Product Image"):
    st.subheader("Result")
    with st.spinner("Analyzing image..."):
        if input_method == "File Uploader" and uploaded_file_img:
            prediction = Anomaly_Detection(uploaded_file_img, DATA_FOLDER)
            st.write(prediction)
        elif input_method == "Camera Input" and camera_file_img:
            prediction = Anomaly_Detection(camera_file_img, DATA_FOLDER)
            st.write(prediction)
        else:
            st.warning("Please provide a valid image input.")
