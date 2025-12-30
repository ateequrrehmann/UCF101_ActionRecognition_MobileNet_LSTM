# ActionRecognition_MobileNetV2_LSTM

This project implements a video action recognition system using Deep Learning. It combines a Convolutional Neural Network (CNN) with a Long Short-Term Memory (LSTM) network to classify actions in videos.

### Live Demo Links
* **Frontend (User Interface):** [https://ucf101-actionrecognition-mobilenet-lstm.onrender.com](https://ucf101-actionrecognition-mobilenet-lstm.onrender.com)
* **Backend (API):** [https://rehmanateequr501--action-recognition-api-predict-action-api.modal.run](https://rehmanateequr501--action-recognition-api-predict-action-api.modal.run)

### Project Overview
The goal of this project is to detect human actions from video clips, such as archery, applying makeup, or playing instruments. We treat a video as a sequence of images. First, we use a CNN to understand what is in each frame. Then, we use an LSTM to understand the order of frames to guess the action.

The model is deployed on a cloud GPU (Modal.com) and accessed via a web interface hosted on Render.

### Why we selected MobileNetV2?
We chose MobileNetV2 as our feature extractor because it is lightweight and fast. Unlike bigger models (like ResNet), MobileNetV2 uses fewer parameters and calculations. This makes it perfect for web applications where we need quick results without requiring a supercomputer to run it. It strikes the best balance between speed and accuracy for our needs.

### Why Transfer Learning?
Training a deep learning model from scratch requires millions of images and weeks of training time. We used Transfer Learning to save time and resources. By starting with weights pre-trained on ImageNet, our model already knew how to identify edges, shapes, and patterns. We only had to teach it how to recognize specific actions in the UCF101 dataset. This helped us achieve over 80% accuracy in just 15 epochs.

### Why UCF101 Dataset?
UCF101 is a standard benchmark dataset for action recognition. We selected it because it contains realistic videos taken from YouTube, rather than staged studio clips. This variety makes the model more robust and applicable to real-world scenarios. It covers 101 different action categories, providing a challenging yet manageable dataset for this assignment.

### Model Performance
* **Training Accuracy:** 79.24%
* **Validation Accuracy:** 80.27%
* **Hardware Used:** NVIDIA A10G GPU (via Modal)

### Technical Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Deployment:** Modal (Backend), Render (Frontend)

### How to Use the API
The backend operates as a REST API. You send the video file as raw binary data, and the server responds with a JSON object containing the prediction.

**Endpoint:** `POST https://rehmanateequr501--action-recognition-api-predict-action-api.modal.run`

#### 1. Request Format
* **Method:** `POST`
* **Headers:** `Content-Type: video/mp4`
* **Body:** Raw binary bytes of the video file. (Do not wrap in JSON, just send the file stream directly).

#### 2. Response Format (JSON)
The API returns a JSON object with the detected action and confidence score.

**Example Success Response:**
```json
{
  "action": "Biking",
  "confidence": 0.947,
  "message": "Success"
}
```

**Example Error Response:**
```json
{
  "detail": "Error processing video"
}
```
### Acknowledgments
* **Instructor:** Sir Mehdi Hassan
* **Dataset:** UCF101 (Center for Research in Computer Vision)
