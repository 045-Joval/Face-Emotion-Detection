
---
# Face Emotion Detection (FER-2013) using TensorFlow.js (Node.js)

This project implements a deep Convolutional Neural Network (CNN) using TensorFlow.js (Node.js backend) to classify facial emotions from grayscale images. The model is trained on the [FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset and is exported in a browser-compatible format for use in frontend applications.

---

## 😃 Emotion Classes
The model detects the following **7 emotions**:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🧠 Model Architecture

The architecture closely follows a deep CNN similar to VGG-style networks:
- Multiple Conv2D blocks with BatchNorm, MaxPooling, and Dropout
- Final Dense layers with Dropout
- Softmax output for 7 classes

> **Regularization**: L2 regularization is applied to Conv layers  
> **Normalization**: Images are scaled between 0 and 1

---

## 🗃 Dataset

**Dataset**: [FER-2013 Facial Emotion Recognition](https://www.kaggle.com/datasets/deadskull7/fer2013)  
**Format**: CSV file containing 48x48 grayscale pixel values and emotion labels  
**Splits Used**:
- **Training**: `Training`
- **Testing**: `PublicTest`

---

## 🖥️ System Configuration Used for Training

- **CPU**: Intel Core i5 12th Gen
- **RAM**: 16GB DDR4
- **GPU**: ❌ None (CPU-only training)
- **Training Time**: Approx. **12 hours**
- **Operating System**: Tested on Ubuntu 24.04

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/045-Joval/Face-Emotion-Detection.git
cd Face-Emotion-Detection/model-training-tfjs
```

### 2. Install Node.js (Using NVM)
This project uses **Node.js v18**.

If you have [NVM](https://github.com/nvm-sh/nvm) installed:
```bash
nvm install 18
nvm use 18
```

### 3. Install Dependencies
```bash
npm install
```

### 4. Download and Place Dataset

Download the FER-2013 dataset CSV from Kaggle:
👉 [FER-2013 on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013)

Then, place the `fer2013.csv` file into the project root:
```
/face-emotion-tfjs-node
│
├── fer2013.csv 👈 Place it here
```

### 5. Run the Training Script
```bash
node run.js
```

> Training may take several hours on CPU systems. Final model is saved in `./face_emotion_model_browser` folder.

---

## 📦 Output

Once training is complete, the trained model will be saved in a browser-compatible format:
```
./face_emotion_model_browser/
├── model.json
├── weights.bin
├── ...
```

You can use this with TensorFlow.js in a web browser.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements
- TensorFlow.js Team
- [FER-2013 Dataset](https://www.kaggle.com/datasets/deadskull7/fer2013) contributors
- Keras community for model inspiration

---

### ✅ **Training Highlights**
- **Training Accuracy:** Peaked at ~77%
- **Validation Accuracy:** ~65% max
- **Test Accuracy:** ~63%  
  (This is quite normal for FER-2013, as it's a pretty challenging dataset with a lot of noise.)

### 🧠 Model Details
- Deep CNN with 8 convolutional layers + BatchNorm, Dropout, and Dense layers
- Over **5.9 million parameters**
- Early stopping was triggered at epoch 46 (good sign of regularization)

### 💾 Model Export
- Exported successfully in **TensorFlow.js format** to:
  ```bash
  ./face_emotion_model_browser
  ```

That directory should now contain:
- `model.json`
- `weights.bin` files (binary weights)

---

### 🔜 Next Steps?

Now you can load this model in your **React + TensorFlow.js** app like this:

```js
const model = await tf.loadLayersModel('/path/to/model.json');
```
