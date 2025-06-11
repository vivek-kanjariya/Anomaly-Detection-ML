# 🧠 Anomaly Detection using CNN (VGG) | Streamlit App

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A simple but powerful image-based anomaly detection system using a VGG-style Convolutional Neural Network, built with PyTorch and deployed with Streamlit.

👉 **Live Demo**: [https://anomaly-detectionn.streamlit.app](https://anomaly-detectionn.streamlit.app)

---

## 📸 Overview

This project is a lightweight anomaly detection system that:

- Classifies whether an image is normal or anomalous.
- Uses a CNN model (based on VGG) trained with PyTorch.
- Deploys a clean and interactive interface using **Streamlit**.

---

## 🔧 Tech Stack

| Category            | Technology Used         |
|---------------------|--------------------------|
| 🧠 ML Framework      | PyTorch, Torchvision     |
| 📊 Data Analysis     | Pandas, NumPy, Seaborn   |
| 📈 Visualization     | Matplotlib               |
| 📦 Deployment        | Streamlit                |
| 📸 Image Handling    | Pillow, OpenCV           |

---

## ⚙️ Features

- 🖼️ Upload an image to check for anomalies
- 📊 Real-time prediction with confidence scores
- 📈 Visualize output using bar plots
- ⚡ Lightweight and responsive UI

---

## 🖥️ Local Setup Instructions

Follow these steps to run the app locally:

### 1. Clone the Repository

```bash
git clone https://github.com/vivek-kanjariya/Anomaly-Detection-ML.git
cd Anomaly-Detection-ML
```

### 2. Create a Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

✅ **Note**: Make sure you're using Python 3.10+. Use virtual environments to avoid dependency conflicts.

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at:  
📍 `http://localhost:8501`

---

## 🧠 How It Works

### 1. Image Upload  
You upload an image via the Streamlit UI.

### 2. Preprocessing  
- Image is resized and normalized.  
- Converted to tensor format for PyTorch.

### 3. Prediction  
- A VGG-style CNN model is loaded.  
- Model classifies the image as `Normal` or `Anomalous`.

### 4. Visualization  
- Results are visualized using a bar plot with Matplotlib.

---

## 📁 Project Structure

```
Anomaly-Detection-ML/
├── app.py                  # Streamlit App UI + Logic
├── model.pth               # Pre-trained model weights
├── utils/
│   └── predict.py          # Model loading and prediction logic
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux apt dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theming
├── README.md               # Project documentation
└── data/                   # Optional: Sample images
```

---

## 🧩 Dependencies

Main Python packages used:

```txt
torch==2.0.0
torchvision==0.15.1
pandas==1.4.2
numpy==1.20.1
opencv-python
matplotlib==3.5.1
seaborn==0.11.2
streamlit==1.19.0
Pillow
```

Install them all using:

```bash
pip install -r requirements.txt
```

---

## 🌐 Deployment

The app is deployed using **Streamlit Cloud**.

👉 Try the app live here:  
🔗 [https://anomaly-detectionn.streamlit.app](https://anomaly-detectionn.streamlit.app)

No installation required. Just open and test.

---

## 📝 License

This project is licensed under the **MIT License**.  
You're free to fork, modify, and build upon it.

---

## 🙋‍♂️ Author

**Vivek Kanjariya**

- GitHub: [@vivek-kanjariya](https://github.com/vivek-kanjariya)

---

## ⭐️ Support

If you find this project useful:

- 🌟 Star the repo  
- 🔁 Share it with others  
- 🛠️ Suggest features via issues or PR  
- 💬 Connect for feedback or collaboration
