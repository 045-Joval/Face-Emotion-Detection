
---

# 👁️‍🗨️ Real-Time Face Emotion Detection using TensorFlow.js

This is a real-time facial emotion detection web application built with **React**, **TensorFlow.js**, and **BlazeFace**. It detects faces using the webcam and predicts emotions using a pre-trained CNN model trained on the FER-2013 dataset.

---

## 🌐 Demo

👉 **Live Demo**: [Click here to try it out!](https://dulcet-mochi-34cced.netlify.app/) 

---

## 📸 Screenshot

![Screenshot 2025-04-26 143843](https://github.com/user-attachments/assets/4048713a-2152-43e7-b3e4-fc63ffe1728d)
![Screenshot 2025-04-26 143809](https://github.com/user-attachments/assets/70e50aee-00b2-42f0-a00a-6e0bdf34b136)
![Screenshot 2025-04-26 143828](https://github.com/user-attachments/assets/94cdac3e-e983-4b83-a392-894423c4a9c5)
![Screenshot 2025-04-26 144931](https://github.com/user-attachments/assets/f75f689b-2c61-491b-98f9-bef02bad318d)

## 🚀 Features

- Real-time webcam-based face detection using BlazeFace.
- Emotion recognition using a TensorFlow.js model.
- Displays bounding boxes on detected faces.
- Shows a confidence-based table with **all 7 emotions** and progress bars for visual feedback.
- Clean UI with overlayed video feed and live predictions.

## 🧠 Emotions Detected

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Sad 😢  
- Surprise 😲  
- Neutral 😐  

## 📦 Tech Stack

- **React.js** – Frontend framework
- **TensorFlow.js** – Machine learning in the browser
- **@tensorflow-models/blazeface** – Face detection
- **Custom FER-2013 model** – Emotion classification (converted to TensorFlow.js format)

## 📁 Project Structure

```bash
src/
├── components/
│   └── EmotionDetector.js     # Main emotion detection component
├── assets/
│   └── face_emotion_model_browser/  # TensorFlow.js model files (model.json + weights)
public/
└── index.html
```

## 📦 Installation

```bash
git clone https://github.com/045-Joval/Face-Emotion-Detection.git
cd Face-Emotion-Detection/frontend
npm install
npm run dev
```

## 📂 Model Directory

Place your converted TensorFlow.js model inside:

```
public/face_emotion_model_browser/
├── model.json
├── weights.bin
```

> Make sure the model is exported from Keras (`model.save('path', save_format='tfjs')`) or converted using the [TensorFlow.js converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter).

## 🖥️ Usage

Once started:

1. The app loads BlazeFace and your custom emotion detection model.
2. Requests access to your webcam.
3. Detects faces and classifies emotions frame-by-frame.
4. Displays bounding boxes and a dynamic table of emotion probabilities.

## 🔐 Permissions

- Requires access to the user's webcam via `navigator.mediaDevices.getUserMedia`.

## 🧪 Model Training (Optional)

If you want to train your own emotion model:

- Dataset: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013)
- Tools: TensorFlow/Keras, convert with `tensorflowjs_converter`
- Architecture: CNN with grayscale 48x48 input and 7 softmax output classes.

## ⚠️ Notes

- Works best with good lighting and single face in frame.
- Mobile support may vary depending on browser support for WebRTC + WebGL.

## 📃 License

MIT License

---
