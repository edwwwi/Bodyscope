BODYSCOPE - An App to maintain your Health

1. Obesity Prediction
A Random Forest model will be trained on user health data to predict obesity risk. This model will be converted to ONNX format for efficient use within the Flutter application. A hybrid deployment strategy uses Antigravity for on-device processing with a FastAPI server fallback.
2. Calorie Tracker
A YOLO model trained on an Indian food dataset from Roboflow will identify food items from images. The model will run within the Flutter app using Antigravity, matching detections to a calorie database.
3. Face BMI Detector
The app will use a pre-existing deep learning model from a GitHub repository to estimate BMI from a user's facial photo. The predicted numerical BMI is then mapped to standard health categories (Underweight, Normal, Obese).
4. Health Assistant (Chatbot)
The in-app chatbot offers personalized health advice. It is powered by the Google Gemini API. This functionality uses a FastAPI backend for communication between the Flutter app and the AI model.



## Model Architecture 
![image]("C:\Users\ASUS\Downloads\BODYscope\BODYSCOPE ARCHITECTURE.png")

