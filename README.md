
---

# 🧠 Real-Time Facial Emotion Recognition using TensorFlow.js

This project implements a **real-time facial emotion detection system** using a custom-trained **Convolutional Neural Network (CNN)** model on the FER-2013 dataset. The solution is divided into two main parts:

1. **Model Training** (Node.js backend using TensorFlow.js)
2. **Web Application** (React + TensorFlow.js + BlazeFace)

The final app detects faces in real-time via webcam and predicts facial emotions using the trained model, entirely in the browser — no server required.

---

## 🌐 Demo

👉 **Live Demo**: [Click here to try it out!](https://6807bd70ef796b0893955e42--fabulous-froyo-43dab4.netlify.app/) 

---

## 📦 Project Overview

| Component     | Description |
|---------------|-------------|
| `model-training-tfjs/` | Model training using TensorFlow.js on Node.js |
| `frontend/`  | Real-time web app built with React and TensorFlow.js |

---

## 😃 Emotions Detected

The model detects the following **7 emotions**:
- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Sad 😢
- Surprise 😲
- Neutral 😐

---

## 📁 Project Structure

```bash
📦 root/
├── model-training-tfjs/            # Model training (Node.js + TensorFlow.js)
│   ├── fer2013.csv                 # FER-2013 dataset CSV
│   ├── run.js                      # Main training script
│   └── face_emotion_model_browser/ # Output model (model.json + weights.bin)
│
├── frontend/                       # React web app
│   ├── public/
│   │   └── face_emotion_model_browser/  # Place your model files here
│   └── src/components/EmotionDetector.js # Real-time detection component
```

---

## 🗃 Dataset

- **Dataset**: [FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013)
- **Format**: CSV with grayscale `48x48` images and emotion labels
- **Splits Used**:  
  - Train: `Training`
  - Test: `PublicTest`

---

## 🧠 Model Architecture

A deep CNN architecture similar to VGG-style networks:

- 8 × Conv2D layers with BatchNorm + ReLU
- MaxPooling and Dropout layers
- Dense layers with Dropout before final Softmax
- L2 regularization on Conv layers
- Input: 48x48 grayscale
- Output: 7 emotion classes

> **Parameters**: ~5.9 million  
> **Training Time**: ~12 hours on CPU  
> **Best Test Accuracy**: ~63%

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/045-Joval/Face-Emotion-Detection.git
cd ./Face-Emotion-Detection/model-training-tfjs
```

---

## 🚀 Part 1: Train the Model (Node.js)

### 📦 Requirements
- Node.js v18 (Recommended: Use [NVM](https://github.com/nvm-sh/nvm))
- TensorFlow.js (Node backend)

### 🔧 Steps

```bash
nvm install 18
nvm use 18
npm install
```

### 📥 Add Dataset

Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) and place it in the project root:

```bash
/model-training-tfjs/
└── fer2013.csv
```

### ▶️ Start Training

```bash
node run.js
```

> Final model will be saved to: `./face_emotion_model_browser/` (TensorFlow.js format)

---

## 🌐 Part 2: Run the Web App (React + TensorFlow.js)

### 📦 Requirements
- Node.js v22
- React.js
- TensorFlow.js
- @tensorflow-models/blazeface

### 🔧 Setup

```bash
cd ../frontend
npm install
```

### 📂 Add the Model

Copy the exported model files from training into the React app:

```bash
cp -r ../model-training-tfjs/face_emotion_model_browser ./public/
```

### ▶️ Start the App

```bash
npm run dev
```

---

## 👁️‍🗨️ Live Demo Features

- Webcam face detection using **BlazeFace**
- Emotion classification using **custom FER-2013 CNN**
- Bounding boxes over faces
- Dynamic emotion confidence table with progress bars
- Clean React UI

---

## 📸 Screenshot

> *(Add a screenshot or GIF of the app running here)*

---

## ⚠️ Notes & Tips

- Works best in **good lighting**
- Supports **desktop browsers** with webcam + WebGL
- Mobile support may vary (WebRTC + WebGL support required)
- For multiple face support, loop over BlazeFace detections

---

## 📄 License

MIT License

---