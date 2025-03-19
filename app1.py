from flask import Flask, request, render_template, Response
from inference_sdk import InferenceHTTPClient
import cv2

# Initialize Flask app
app = Flask(__name__)

# Hard-coded API key and inference client
api_key = "fCUAHGl3HrcCS0LybLEx"
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)

def detect_ambulance(frame):
    """Detects an ambulance in a given frame. Returns True if an ambulance is detected, else False."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = CLIENT.infer(gray_frame, model_id="ambulance-sjpea/18")

    # Check if there are any predictions and if the class is 'ambulance'
    if 'predictions' in result and result['predictions']:
        for prediction in result['predictions']:
            label = prediction['class']
            if label.lower() == 'ambulance':
                return True  # Ambulance detected
    return False  # No ambulance detected

def process_video_amb(file_path):
    """Processes the video and returns True if an ambulance is detected, otherwise False."""
    cap = cv2.VideoCapture(file_path)
    ambulance_detected = False
    frame_interval = 1  # Process every 10th frame to optimize performance

    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only check for ambulance detection on every 10th frame
        if frame_count % frame_interval == 0:
            if detect_ambulance(frame):
                ambulance_detected = True
                break

        frame_count += 1

    cap.release()
    return ambulance_detected  # Returns True if ambulance is detected in any frame
