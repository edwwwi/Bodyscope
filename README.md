# BODYSCOPE - An App to Maintain Your Health

A comprehensive health tracking application with AI-powered features for obesity prediction, food calorie tracking, face-based BMI detection, and personalized health guidance.

## Features

### 1. Obesity Prediction
A Random Forest model trained on user health data to predict obesity risk. This model is converted to ONNX format for efficient use within the Flutter application. A hybrid deployment strategy uses Antigravity for on-device processing with a FastAPI server fallback.

### 2. Calorie Tracker
A YOLO model trained on an Indian food dataset from Roboflow identifies food items from images. The model runs within the Flutter app using Antigravity, matching detections to a calorie database.

### 3. Face BMI Detector
Uses a pre-existing deep learning model from a GitHub repository to estimate BMI from a user's facial photo. The predicted numerical BMI is mapped to standard health categories (Underweight, Normal, Obese).

### 4. Health Assistant (Chatbot)
An in-app chatbot powered by the Google Gemini API offers personalized health advice. This functionality uses a FastAPI backend for communication between the Flutter app and the AI model.

## Model Architecture

![Model Architecture](Images/BODYSCOPE%20ARCHITECTURE.png)

