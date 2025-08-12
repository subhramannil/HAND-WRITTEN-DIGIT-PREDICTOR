# 🖋 Handwritten Digit Recognition (MNIST Dataset)

An interactive deep learning application that recognizes handwritten digits in real time using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

![Demo Screenshot](assets/demo.gif) <!-- Replace with your demo GIF or image -->

---

## ✨ Features
- **High Accuracy** – Achieved **99%** test accuracy through custom CNN architecture and hyperparameter tuning.
- **Interactive Web App** – Built with **Streamlit**, allowing users to draw digits directly on a canvas and receive instant predictions.
- **Optimized Training** – Implemented dropout, data augmentation, and regularization to prevent overfitting.
- **User-Friendly** – Designed for accessibility, making digit recognition intuitive for both technical and non-technical users.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries & Frameworks:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Streamlit
- **Dataset:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

---

## 📊 Model Architecture
```text
Input Layer (28x28 grayscale)
↓
Conv2D → ReLU → MaxPooling2D
↓
Conv2D → ReLU → MaxPooling2D
↓
Flatten
↓
Dense → ReLU → Dropout
↓
Dense → Softmax (10 classes)
```

## 🚀 Installation & Usage
**Clone the repository**:

git clone https://github.com/subhramannil/handwritten-digit-recognition.git
cd handwritten-digit-recognition
**Install dependencies**:

pip install -r requirements.txt
**Run the Streamlit app**:

streamlit run app.py

## 📂 Project Structure

Edit
├── app.py                  # Streamlit app
├── model.py                # CNN model definition and training
├── requirements.txt        # Dependencies
├── assets/                 # Images, GIFs for README
└── README.md               # Project documentation
